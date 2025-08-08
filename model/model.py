import torch
from torch import nn
from torch.nn import functional as F
from torch.func import vmap, functional_call, grad
from typing import Optional
from dataclasses import dataclass, replace
import numpy as np
import einops
from tqdm.notebook import trange
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Config:
  n_features: int
  n_hidden: int
  n_experts: int # total number of experts
  n_active_experts:int  # no of active experts
  load_balancing_loss: bool

# training helpers
def linear_lr(step, steps):
  return (1 - (step / steps))

def constant_lr(*_):
  return 1.0

def cosine_decay_lr(step, steps):
  return np.cos(0.5 * np.pi * step / (steps - 1))

# model
class MoEModel(nn.Module):
  def __init__(self, 
               config, 
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,               
               device='cuda'):
    super().__init__()
    self.config = config
    self.W_experts = nn.Parameter(torch.empty((config.n_experts, config.n_features, config.n_hidden), device=device))
    nn.init.xavier_normal_(self.W_experts)
    self.b_final = nn.Parameter(torch.zeros((config.n_experts, config.n_features), device=device))
    self.gate = nn.Parameter(torch.zeros((config.n_experts, config.n_features), device=device))
    nn.init.xavier_normal_(self.gate)

    if feature_probability is None:
      feature_probability = torch.ones(())
    self.feature_probability = feature_probability.to(device)
    if importance is None:
      importance = torch.ones(())
    self.importance = importance.to(device)

  def compute_active_experts(self, features):   
    # features: [..., n_features]
    # gate: [n_experts, n_features]     
    gate_scores = torch.einsum("...f,ef->...e", features, self.gate)
    gate_probs = F.softmax(gate_scores, dim=-1)
    
    top_k_values, top_k_indices = torch.topk(gate_probs, k=self.config.n_active_experts, dim=-1)
    active_mask = torch.zeros_like(gate_probs)
    active_mask = active_mask.scatter(-1, top_k_indices, 1.0)    
    
    load_balance_loss = None
    if self.config.load_balancing_loss:
      # P_i: average router probability for expert i (before top-k selection)
      P_i = torch.mean(gate_probs, dim=tuple(range(gate_probs.dim() - 1)))
      
      # f_i: fraction of tokens actually dispatched to expert i (after top-k selection)
      f_i = torch.mean(active_mask, dim=tuple(range(active_mask.dim() - 1)))
      
      N = self.config.n_experts
      alpha = 0.01
      load_balance_loss = alpha * N * torch.sum(f_i * P_i)
    
    # renormalize gating weights for active experts only
    # sum of probabilities for active experts
    active_sum = torch.sum(gate_probs * active_mask, dim=-1, keepdim=True)
    
    renormalized_weights = torch.where(
        active_mask.bool(),
        gate_probs / active_sum,
        torch.zeros_like(gate_probs)
    )
    return renormalized_weights, top_k_indices, load_balance_loss


  def forward(self, features):
    # features: [..., n_features]    

    expert_weights, top_k_indices, load_balance_loss = self.compute_active_experts(features)
    
    # hidden: [..., n_experts, n_hidden] - compression
    hidden = torch.einsum("...f,efh->...eh", features, self.W_experts)
    
    # expert_outputs: [..., n_experts, n_features]
    expert_outputs = torch.einsum("...eh,efh->...ef", hidden, self.W_experts)
    expert_outputs = expert_outputs + self.b_final
    expert_outputs = F.relu(expert_outputs)
  
    # final_output: [..., n_features] - recons
    final_output = torch.einsum("...e,...ef->...f", expert_weights, expert_outputs)
    return final_output, load_balance_loss

  def generate_batch(self, n_batch):
    feat = torch.rand((n_batch, self.config.n_features), device=self.W_experts.device)
    batch = torch.where(
        torch.rand((n_batch, self.config.n_features), device=self.W_experts.device) <= self.feature_probability,
        feat,
        torch.zeros((), device=self.W_experts.device),
    )
    return batch
  


# training
def optimize(model, 
             render=False, 
             n_batch=1024,
             steps=10_000,
             print_freq=100,
             lr=1e-3,
             lr_scale=constant_lr,
             hooks=[]):
  """train a moe"""
  cfg = model.config

  opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

  start = time.time()
  # Replace trange with regular range
  for step in range(steps):
    step_lr = lr * lr_scale(step, steps)
    for group in opt.param_groups:
      group['lr'] = step_lr
    opt.zero_grad(set_to_none=True)
    batch = model.generate_batch(n_batch)
    out, load_balance_loss = model(batch)
    error = (model.importance*(batch.abs() - out)**2)
    reconstruction_loss = einops.reduce(error, 'b f -> f', 'mean').sum()
    
    loss = reconstruction_loss
    if load_balance_loss is not None:
      loss = loss + load_balance_loss
    
    loss.backward()
    opt.step()
  
    if hooks:
      hook_data = dict(model=model,
                       step=step, 
                       opt=opt,
                       error=error,
                       loss=loss,
                       reconstruction_loss=reconstruction_loss,
                       load_balance_loss=load_balance_loss,
                       lr=step_lr)
      for h in hooks:
        h(hook_data)
    if step % print_freq == 0 or (step + 1 == steps):
      print(f"Step {step}: loss={loss.item():.6f}, lr={step_lr:.6f}")


# vmap helpers
def make_functional_model(config, device, importance, feature_probability):
    """separates the model's computation (functions) from its params for vmap"""
    model = MoEModel(config, device=device, importance=importance, feature_probability=feature_probability)
    
    # Extract parameters and buffers as dictionaries
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    
    def func_model(params_dict, buffers_dict, *inputs):
        state_dict = {**params_dict, **buffers_dict}
        return functional_call(model, state_dict, inputs[0] if len(inputs) == 1 else inputs)
    
    return func_model, params, buffers

def vectorized_forward(params_batch, buffers_batch, features_batch, func_model):
  """in_dims tells vmap that first dim of each input is batch"""
  return vmap(func_model, in_dims=(0, 0, 0))(params_batch, buffers_batch, features_batch)

def generate_vectorized_batch(configs, feature_probs, n_batch, device):
    batches = []
    for config, feat_prob in zip(configs, feature_probs):
        feat = torch.rand((n_batch, config.n_features), device=device)
        batch = torch.where(
            torch.rand((n_batch, config.n_features), device=device) <= feat_prob,
            feat,
            torch.zeros((), device=device)
        )
        batches.append(batch)
    return torch.stack(batches)  # Shape: [n_models, n_batch, n_features]

def stack_state_dicts(state_dicts):
    """stack a list of state dictionaries into a single state dict with batched tensors"""
    if not state_dicts:
        return {}
    
    stacked = {}
    for key in state_dicts[0].keys():
        stacked_tensor = torch.stack([sd[key] for sd in state_dicts])
        stacked[key] = stacked_tensor.detach().requires_grad_(True)
    return stacked


# vectorized training
def optimize_vectorized(configs, feature_probs, importances, 
                        device=DEVICE,
                        n_batch=1024, 
                        steps=10_000, 
                        print_freq=100, 
                        lr=1e-3, 
                        lr_scale=constant_lr, 
                        hooks=[]):
    
    func_models = []
    all_params = []
    all_buffers = []

    for config, feat_prob, importance in zip(configs, feature_probs, importances):
        func_model, params, buffers = make_functional_model(config, device, importance, feat_prob)
        func_models.append(func_model)
        all_params.append(params)
        all_buffers.append(buffers)

    stacked_params = stack_state_dicts(all_params)
    stacked_buffers = stack_state_dicts(all_buffers)

    flat_params = list(stacked_params.values())
    
    opt = torch.optim.AdamW(flat_params, lr=lr)

    start = time.time()

    # replace trange with regular range
    for step in range(steps):
        step_lr = lr * lr_scale(step, steps)
        for group in opt.param_groups:
            group['lr'] = step_lr
        opt.zero_grad(set_to_none=True)

        batch = generate_vectorized_batch(configs, feature_probs, n_batch, device)
        # use the first func_model since they should all have the same signature
        out, load_balance_loss = vectorized_forward(stacked_params, stacked_buffers, batch, func_models[0])

        stacked_importance = torch.stack(importances)
        error = stacked_importance.unsqueeze(1) * (batch.abs() - out)**2

        reconstruction_losses = einops.reduce(error, 'models b f -> models', 'mean')
        losses = reconstruction_losses
        if load_balance_loss is not None:
            losses = losses + load_balance_loss

        total_loss = losses.sum()

        total_loss.backward()
        opt.step()
        
        if hooks:
            hook_data = dict(models=func_models,
                            step=step, 
                            opt=opt,
                            errors=error,
                            losses=losses,
                            total_loss=total_loss,
                            reconstruction_losses=reconstruction_losses,
                            load_balance_losses=load_balance_loss,
                            lr=step_lr)
            
            for h in hooks:
                h(hook_data)
        if step % print_freq == 0 or (step + 1 == steps):
            print(f"Step {step}: avg_loss={losses.mean().item():.6f}, lr={step_lr:.6f}")
import torch
import torch.nn.functional as F

def expert_feat_dimensionality(model):
    """fraction of a dimension within an expert that is occupied by a feature."""
    n_experts = model.config.n_experts
    n_features = model.config.n_features
    
    # Initialize tensor to store results for all experts
    all_dimensionalities = torch.zeros(n_experts, n_features)
    
    for expert_idx in range(n_experts):
        W_expert = model.W_experts[expert_idx]  # shape: [n_features, n_hidden]
        W_norm_squared = torch.sum(W_expert**2, dim=1)  # shape: [n_features], ||W_i||^2
        W_hat = W_expert / torch.sqrt(W_norm_squared).unsqueeze(1) # W_hat_i = W_i / ||W_i||
        dot_products = torch.mm(W_hat, W_expert.T) # W_hat_i · W_j
        squared = dot_products**2
        interference = torch.sum(squared, dim=1) # sum of squared dot products for each feature
        dimensionality = W_norm_squared / interference # D_i^(e) = ||W_i^(e)||^2 / sum_j(W_hat_i · W_j)^2
        all_dimensionalities[expert_idx] = dimensionality
    
    return torch.round(all_dimensionalities, decimals=3)

def global_feature_dimensionality(model, input_features=None, n_batch=4096):
    """global feature dimensionality weighted across all experts."""
    
    # get per-expert dimensionalities (you already have this)
    expert_dimensionalities = expert_feat_dimensionality(model)  # shape: [n_experts, n_features]
    
    if input_features is None:
        input_features = model.generate_batch(n_batch) 
    
    # compute gating probabilities
    gate_scores = torch.einsum("...f,ef->...e", input_features, model.gate)
    gate_probs = F.softmax(gate_scores, dim=-1)  # shape: [batch, n_experts]

    #Find the top expert for each point
    selected = gate_probs.argmax(dim=-1)
    counts = torch.bincount(selected.flatten())
    if counts.shape[0] < model.config.n_experts:
        counts = torch.cat([counts, torch.zeros(model.config.n_experts - counts.shape[0])])
    usage = counts / counts.sum()
    #print(f"Usage: {usage}")
    
    # average probabilities across batch to get p_e
    #p_e = torch.mean(gate_probs, dim=0)  # shape: [n_experts]
    #print(f"p_e: {p_e}")
    
    # D_i^global = Σ_e p_e · D_i^(e)
    #global_dims_old = torch.einsum("e,ef->f", p_e, expert_dimensionalities)
    global_dims_new = torch.einsum("e,ef->f", usage, expert_dimensionalities)
    return usage, torch.round(global_dims_new, decimals=3)
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_feat_dims import global_feature_dimensionality
from model.model import Config, MoEModel, optimize_vectorized, make_functional_model, vectorized_forward, generate_vectorized_batch, stack_state_dicts
from helpers.dimensions_per_feature import compute_dimensions_per_feature_single
from helpers.expert_classification import classify_all_experts_feature_allocation
from helpers.expert_classification import norm_and_superposition

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GridExperimentConfig:
    """Configuration for the sparsity-importance grid experiment."""
    n_features: int = 32
    n_hidden: int = 8
    n_experts: int = 2
    n_active_experts: int = 1
    n_batch: int = 512
    steps: int = 3000
    lr: float = 1e-3
    tolerance: float = 0.1
    n_models_per_cell: int = 1  # Number of models to train per grid cell
    
    # Grid parameters
    sparsity_values: List[float] = None  # Feature sparsity values
    importance_values: List[float] = None  # Last feature importance values
    base_importance: float = 1.0  # Base importance for all features except last
    skip_last_feature: bool = False # treat last feature like the rest
    # n_experts_values: List[int] = None  # Number of experts values, replaces importance if specified
    
    def __post_init__(self):
        if self.sparsity_values is None:
            self.sparsity_values = np.arange(0.01, 1.0, 0.05)
        if self.importance_values is None:
            self.importance_values = np.arange(0.1, 5, 0.25)


def create_grid_configs(config: GridExperimentConfig) -> Tuple[List[Config], List[torch.Tensor], List[torch.Tensor]]:
    """
    Create configurations for all grid points.
    
    Returns:
        configs: List of model configurations
        feature_probs: List of feature probability tensors
        importances: List of importance tensors
    """
    configs = []
    feature_probs = []
    importances = []
    experts = []

    if config.n_experts_values is not None:
        print(f"Using {config.n_experts_values} as importance values")
    
    for sparsity in config.sparsity_values:
        for second_option in (config.n_experts_values or config.importance_values):
            # Create model config
            model_config = Config(
                n_features=config.n_features,
                n_hidden=config.n_hidden,
                n_experts=config.n_experts if config.n_experts_values is None else second_option,
                n_active_experts=config.n_active_experts,
                load_balancing_loss=True
            )
            configs.append(model_config)
            
            # Create feature probability tensor (same sparsity for all features)
            feature_prob = 1 - torch.tensor(sparsity, device=DEVICE)
            feature_probs.append(feature_prob)
            
            # Create importance tensor (base importance for all except last, custom for last)
            importance_tensor = torch.ones(config.n_features, device=DEVICE) * config.base_importance
            if ((not config.skip_last_feature) and (config.n_experts_values is None)): 
                importance_tensor[-1] = second_option  # Set last feature importance
            importances.append(importance_tensor)
            
            experts.append(config.n_experts if config.n_experts_values is None else second_option)
    
    return configs, feature_probs, importances, experts


def train_grid_models(config: GridExperimentConfig, custom_gate_init=False) -> Dict[str, Any]:
    """
    Train multiple independent runs using vmap, then select best models per cell.
    
    Returns:
        Dictionary with training results and model states
    """
    print(f"Creating grid with {len(config.sparsity_values)} sparsity values and {len(config.importance_values)} importance values")
    print(f"Total grid points: {len(config.sparsity_values) * len(config.importance_values)}")
    print(f"Training {config.n_models_per_cell} independent runs...")
    
    # Create base configurations for all grid points
    base_configs, feature_probs, importances, experts = create_grid_configs(config)
    
    # Store all runs for each grid cell
    all_runs = []
    
    start_time = time.time()
    
    # Train multiple independent runs
    for run_idx in range(config.n_models_per_cell):
        print(f"Training run {run_idx + 1}/{config.n_models_per_cell}...")
        
        # Set different random seed for each run
        torch.manual_seed(42 + run_idx * 10000)
        np.random.seed(42 + run_idx * 10000)
        
        # Train all grid points for this run using vmap
        final_losses, final_models = optimize_vectorized(
            base_configs,
            feature_probs,
            importances,
            device=DEVICE,
            n_batch=config.n_batch,
            steps=config.steps,
            lr=config.lr,
            print_freq=config.steps // 4 if run_idx == 0 else config.steps + 1  # Only print for first run
        )
        
        # Convert stacked parameters back to individual model parameters
        individual_models = []
        for i in range(len(base_configs)):
            model_params = {}
            for key in final_models.keys():
                model_params[key] = final_models[key][i]
            individual_models.append(model_params)
        
        all_runs.append({
            'losses': final_losses,
            'models': individual_models
        })
    
    # Select best model for each grid cell
    best_models = []
    best_losses = []
    
    print("Selecting best models per grid cell...")
    for grid_idx in range(len(base_configs)):
        # Compare all runs for this grid cell
        cell_losses = [run['losses'][grid_idx].item() for run in all_runs]
        cell_models = [run['models'][grid_idx] for run in all_runs]
        
        # Select best model
        best_run_idx = np.argmin(cell_losses)
        best_loss = cell_losses[best_run_idx]
        best_model = cell_models[best_run_idx]
        
        best_models.append(best_model)
        best_losses.append(best_loss)
        
        # Print progress for first few cells
        if grid_idx < 5:
            sparsity_idx = grid_idx // len(config.importance_values)
            importance_idx = grid_idx % len(config.importance_values)
            sparsity = config.sparsity_values[sparsity_idx]
            importance = config.importance_values[importance_idx]
            print(f"  Cell ({sparsity:.2f}, {importance:.1f}): best loss {best_loss:.6f} (run {best_run_idx + 1})")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Total models trained: {len(base_configs) * config.n_models_per_cell}")
    
    return {
        'configs': base_configs,
        'feature_probs': feature_probs,
        'importances': importances,
        'final_losses': torch.tensor(best_losses),
        'final_models': best_models,
        'sparsity_values': config.sparsity_values,
        'importance_values': config.importance_values,
        'all_runs': all_runs  # Keep all runs for potential analysis
    }

def plot_global_feature_dimensionality_grid(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Create a 2D grid plot showing global feature dimensionality.
    """
    sparsity_values = config.sparsity_values
    importance_values = config.importance_values

    # plot global feature dimensionality
    global_dims = np.zeros((len(sparsity_values), len(importance_values)))
    for i, sparsity in enumerate(sparsity_values):
        for j, importance in enumerate(importance_values):
            result = grid_results[(sparsity, importance)]
            global_dims[i, j] = global_feature_dimensionality(result)

    # plot global feature dimensionality
    im = plt.imshow(global_dims, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Global Feature Dimensionality')
    plt.xlabel('Last Feature Importance')
    plt.ylabel('Feature Sparsity')
    plt.title('Global Feature Dimensionality')
    plt.show()
    

def analyze_grid_models(results: Dict[str, Any], tolerance=0.1) -> Dict[str, Any]:
    """
    Analyze all trained models and classify their expert allocations.
    
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing expert allocations for all grid points...")
    
    grid_results = {}
    sparsity_values = results['sparsity_values']
    importance_values = results['importance_values']
    
    for i, sparsity in enumerate(sparsity_values):
        for j, importance in enumerate(importance_values):
            grid_idx = i * len(importance_values) + j
            
            # Get model parameters
            model_params = results['final_models'][grid_idx]
            
            # Extract expert weights
            expert_weights = model_params['W_experts']  # Shape: [n_experts, n_features, n_hidden]
            
            # Classify expert allocations
            classifications = classify_all_experts_feature_allocation(expert_weights, tolerance=tolerance)
            
            # Get allocation types for plotting
            allocation_types = []
            for expert_key in sorted(classifications.keys()):
                allocation_types.append(classifications[expert_key]['allocation_type'])
            
            # Store results
            grid_results[(sparsity, importance)] = {
                'classifications': classifications,
                'allocation_types': allocation_types,
                'final_loss': results['final_losses'][grid_idx].item(),
                'expert_weights': expert_weights.cpu().detach().numpy(),
                'gate_params': model_params['gate'].cpu().detach().numpy(),
                'grid_idx': grid_idx,
                'model': model_params
            }
    
    return grid_results


def plot_allocation_grid(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Create a 2D grid plot showing expert allocation types.
    """
    sparsity_values = config.sparsity_values
    importance_values = config.importance_values
    
    # Create figure with subplots for each expert
    n_experts = config.n_experts
    fig, axes = plt.subplots(1, n_experts, figsize=(5*n_experts, 5))
    if n_experts == 1:
        axes = [axes]
    
    # Define color mapping for allocation types
    allocation_types = set()
    for result in grid_results.values():
        allocation_types.update(result['allocation_types'])
    
    # Create color map
    unique_types = sorted(list(allocation_types))
    type_to_idx = {allocation_type: idx for idx, allocation_type in enumerate(unique_types)}
    
    # Plot for each expert
    for expert_idx in range(n_experts):
        ax = axes[expert_idx]
        
        # Create grid data with numerical indices
        grid_data = np.zeros((len(sparsity_values), len(importance_values)), dtype=int)
        
        for i, sparsity in enumerate(sparsity_values):
            for j, importance in enumerate(importance_values):
                result = grid_results[(sparsity, importance)]
                allocation_type = result['allocation_types'][expert_idx]
                grid_data[i, j] = type_to_idx[allocation_type]
        
        # Create heatmap
        im = ax.imshow(grid_data, cmap='Set3', aspect='auto', vmin=0, vmax=len(unique_types)-1)
        ax.set_xlabel('Last Feature Importance', fontsize=12, labelpad=10)
        ax.set_ylabel('Feature Sparsity', fontsize=12, labelpad=10)
        ax.set_title(f'Expert {expert_idx} Allocation Type', fontsize=14, pad=15)
        
        # Set tick labels with better spacing and fewer ticks
        # Reduce number of ticks for better legibility
        x_step = max(1, len(importance_values) // 8)  # Show ~8 ticks max
        y_step = max(1, len(sparsity_values) // 6)    # Show ~6 ticks max
        
        x_ticks = range(0, len(importance_values), x_step)
        y_ticks = range(0, len(sparsity_values), y_step)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{importance_values[i]:.1f}' for i in x_ticks], rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{sparsity_values[i]:.2f}' for i in y_ticks])
        
        # Only add colorbar to the last subplot to avoid duplication
        if expert_idx == n_experts - 1:
            cbar = plt.colorbar(im, ax=ax, ticks=range(len(unique_types)))
            cbar.set_ticklabels(unique_types, ha='left')
            cbar.set_label('Allocation Type', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    plt.show()
    
    # Print summary statistics
    print("\n=== Grid Experiment Summary ===")
    print(f"Grid size: {len(sparsity_values)} x {len(importance_values)} = {len(grid_results)} points")
    print(f"Total experts analyzed: {len(grid_results) * n_experts}")
    
    # Count allocation types across all experts
    type_counts = {}
    for result in grid_results.values():
        for allocation_type in result['allocation_types']:
            if allocation_type not in type_counts:
                type_counts[allocation_type] = 0
            type_counts[allocation_type] += 1
    
    print("\nAllocation type distribution across all experts:")
    for allocation_type, count in sorted(type_counts.items()):
        percentage = (count / (len(grid_results) * n_experts)) * 100
        print(f"  {allocation_type}: {count} experts ({percentage:.1f}%)")


def plot_dimensions_per_feature_grid(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Create a 2D grid plot showing dimensions per feature analysis for each grid point.
    """
    sparsity_values = config.sparsity_values
    importance_values = config.importance_values
    n_experts = config.n_experts
    
    # Create figure with subplots for each expert
    fig, axes = plt.subplots(1, n_experts, figsize=(5*n_experts, 5))
    if n_experts == 1:
        axes = [axes]

    # Plot for each expert
    for expert_idx in range(n_experts):
        ax = axes[expert_idx]
        
        # Create grid data for dimensions per feature
        dims_grid = np.zeros((len(sparsity_values), len(importance_values)))
        
        for i, sparsity in enumerate(sparsity_values):
            for j, importance in enumerate(importance_values):
                result = grid_results[(sparsity, importance)]
                # Get the expert weights for this expert
                expert_weights = result['expert_weights'][expert_idx]  # Shape: [n_features, n_hidden]
                
                # Compute dimensions per feature for this expert
                # Convert numpy array to tensor if needed
                if isinstance(expert_weights, np.ndarray):
                    expert_weights_tensor = torch.tensor(expert_weights, dtype=torch.float32)
                else:
                    expert_weights_tensor = expert_weights
                
                analysis = compute_dimensions_per_feature_single(expert_weights_tensor)
                dims_grid[i, j] = analysis['dimensions_per_feature'].item()
        
        # Create heatmap
        im = ax.imshow(dims_grid, cmap='viridis', aspect='auto')
        ax.set_xlabel('Last Feature Importance', fontsize=12, labelpad=10)
        ax.set_ylabel('Feature Sparsity', fontsize=12, labelpad=10)
        ax.set_title(f'Expert {expert_idx} Dimensions per Feature', fontsize=14, pad=15)
        
        # Set tick labels with better spacing and fewer ticks
        x_step = max(1, len(importance_values) // 8)
        y_step = max(1, len(sparsity_values) // 6)
        
        x_ticks = range(0, len(importance_values), x_step)
        y_ticks = range(0, len(sparsity_values), y_step)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{importance_values[i]:.1f}' for i in x_ticks], rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{sparsity_values[i]:.2f}' for i in y_ticks])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dimensions per Feature', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
    # Print summary statistics for dimensions per feature
    print("\n=== Dimensions per Feature Summary ===")
    
    # Calculate statistics across all experts
    all_dims = []
    for result in grid_results.values():
        for expert_idx in range(n_experts):
            expert_weights = result['expert_weights'][expert_idx]
            # Convert numpy array to tensor if needed
            if isinstance(expert_weights, np.ndarray):
                expert_weights_tensor = torch.tensor(expert_weights, dtype=torch.float32)
            else:
                expert_weights_tensor = expert_weights
            
            analysis = compute_dimensions_per_feature_single(expert_weights_tensor)
            all_dims.append(analysis['dimensions_per_feature'].item())
    
    all_dims = np.array(all_dims)
    print(f"Mean dimensions per feature: {np.mean(all_dims):.3f}")
    print(f"Std dimensions per feature: {np.std(all_dims):.3f}")
    print(f"Min dimensions per feature: {np.min(all_dims):.3f}")
    print(f"Max dimensions per feature: {np.max(all_dims):.3f}")
    
    # Analyze by allocation type
    print("\nDimensions per feature by allocation type:")
    
    # Calculate type counts for this analysis
    type_counts = {}
    for result in grid_results.values():
        for allocation_type in result['allocation_types']:
            if allocation_type not in type_counts:
                type_counts[allocation_type] = 0
            type_counts[allocation_type] += 1
    
    for allocation_type in sorted(type_counts.keys()):
        type_dims = []
        for result in grid_results.values():
            for expert_idx in range(n_experts):
                if result['allocation_types'][expert_idx] == allocation_type:
                    expert_weights = result['expert_weights'][expert_idx]
                    # Convert numpy array to tensor if needed
                    if isinstance(expert_weights, np.ndarray):
                        expert_weights_tensor = torch.tensor(expert_weights, dtype=torch.float32)
                    else:
                        expert_weights_tensor = expert_weights
                    
                    analysis = compute_dimensions_per_feature_single(expert_weights_tensor)
                    type_dims.append(analysis['dimensions_per_feature'].item())
        
        if type_dims:
            type_dims = np.array(type_dims)
            print(f"  {allocation_type}: mean={np.mean(type_dims):.3f}, std={np.std(type_dims):.3f}")


def plot_loss_grid_sparsity_last_feature(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Create a 2D grid plot showing final training losses.
    """
    sparsity_values = config.sparsity_values
    importance_values = config.importance_values
    
    # Create loss grid
    loss_grid = np.zeros((len(sparsity_values), len(importance_values)))
    
    for i, sparsity in enumerate(sparsity_values):
        for j, importance in enumerate(importance_values):
            result = grid_results[(sparsity, importance)]
            loss_grid[i, j] = result['final_loss']
    
    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(loss_grid, cmap='viridis', aspect='auto')
    cbar = plt.colorbar(label='Final Loss')
    cbar.ax.tick_params(labelsize=10)
    
    plt.xlabel('Last Feature Importance', fontsize=12, labelpad=10)
    plt.ylabel('Feature Sparsity', fontsize=12, labelpad=10)
    plt.title('Final Training Loss Across Grid', fontsize=14, pad=15)
    
    # Set tick labels with better spacing and fewer ticks
    # Reduce number of ticks for better legibility
    x_step = max(1, len(importance_values) // 8)  # Show ~8 ticks max
    y_step = max(1, len(sparsity_values) // 6)    # Show ~6 ticks max
    
    x_ticks = range(0, len(importance_values), x_step)
    y_ticks = range(0, len(sparsity_values), y_step)
    
    plt.xticks(x_ticks, [f'{importance_values[i]:.1f}' for i in x_ticks], rotation=45, ha='right', fontsize=10)
    plt.yticks(y_ticks, [f'{sparsity_values[i]:.2f}' for i in y_ticks], fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    plt.show()

def plot_loss_grid_sparsity_experts(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Create a 2D grid plot showing final training losses.
    """
    sparsity_values = config.sparsity_values
    n_experts = config.n_experts
    
    # Create loss grid
    loss_grid = np.zeros((len(sparsity_values), n_experts))
    
    for i, sparsity in enumerate(sparsity_values):
        for j in range(n_experts):
            result = grid_results[(sparsity, j)]
            loss_grid[i, j] = result['final_loss']
    
    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(loss_grid, cmap='viridis', aspect='auto')
    cbar = plt.colorbar(label='Final Loss')
    cbar.ax.tick_params(labelsize=10)
    
    plt.xlabel('Last Feature Importance', fontsize=12, labelpad=10)
    plt.ylabel('Feature Sparsity', fontsize=12, labelpad=10)
    plt.title('Final Training Loss Across Grid', fontsize=14, pad=15)
    
    # Set tick labels with better spacing and fewer ticks
    # Reduce number of ticks for better legibility
    x_step = max(1, n_experts // 8)  # Show ~8 ticks max
    y_step = max(1, len(sparsity_values) // 6)    # Show ~6 ticks max
    
    x_ticks = range(0, n_experts, x_step)
    y_ticks = range(0, len(sparsity_values), y_step)
    
    plt.xticks(x_ticks, [f'{j}' for j in x_ticks], rotation=45, ha='right', fontsize=10)
    plt.yticks(y_ticks, [f'{sparsity_values[i]:.2f}' for i in y_ticks], fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    plt.show()


def save_grid_results(grid_results: Dict[str, Any], config: GridExperimentConfig, filename: str = None):
    """
    Save grid results to a JSON file.
    """
    if filename is None:
        timestamp = int(time.time())
        filename = f"grid_experiment_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, result in grid_results.items():
        serializable_results[str(key)] = {
            'allocation_types': result['allocation_types'],
            'final_loss': float(result['final_loss']),
            'grid_idx': result['grid_idx']
        }
    
    data_to_save = {
        'config': {
            'n_features': config.n_features,
            'n_hidden': config.n_hidden,
            'n_experts': config.n_experts,
            'sparsity_values': config.sparsity_values,
            'importance_values': config.importance_values,
            'base_importance': config.base_importance
        },
        'results': serializable_results
    }
    
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Results saved to {filename}")


def analyze_expert_bias(grid_results: Dict[str, Any], config: GridExperimentConfig):
    """
    Analyze potential systematic biases in expert assignment.
    """
    print("\n=== Expert Assignment Bias Analysis ===")
    
    sparsity_values = config.sparsity_values
    importance_values = config.importance_values
    n_experts = config.n_experts
    
    # Analyze gate parameters
    print("\n1. Gate Parameter Analysis:")
    gate_biases = []
    
    for i, sparsity in enumerate(sparsity_values):
        for j, importance in enumerate(importance_values):
            result = grid_results[(sparsity, importance)]
            
            # Get gate parameters (if available)
            if 'gate_params' in result:
                gate_params = result['gate_params']
                # Analyze gate parameter patterns
                gate_means = np.mean(gate_params, axis=1)  # Mean across features for each expert
                gate_biases.append(gate_means)
    
    if gate_biases:
        gate_biases = np.array(gate_biases)
        print(f"Gate parameter means across all models:")
        for expert_idx in range(n_experts):
            expert_gates = gate_biases[:, expert_idx]
            print(f"  Expert {expert_idx}: mean={np.mean(expert_gates):.4f}, std={np.std(expert_gates):.4f}")
    
    # Analyze allocation patterns
    print("\n2. Allocation Pattern Analysis:")
    
    # Count allocation types by grid position
    allocation_counts = {}
    for result in grid_results.values():
        for expert_idx, allocation_type in enumerate(result['allocation_types']):
            key = f"Expert_{expert_idx}_{allocation_type}"
            if key not in allocation_counts:
                allocation_counts[key] = 0
            allocation_counts[key] += 1
    
    print("Allocation type counts:")
    for key, count in sorted(allocation_counts.items()):
        percentage = (count / len(grid_results)) * 100
        print(f"  {key}: {count} ({percentage:.1f}%)")
    
    # Analyze spatial patterns
    print("\n3. Spatial Pattern Analysis:")
    
    # Create matrices to track which expert gets which allocation type
    expert_0_patterns = np.zeros((len(sparsity_values), len(importance_values)), dtype=bool)
    expert_1_patterns = np.zeros((len(sparsity_values), len(importance_values)), dtype=bool)
    
    for i, sparsity in enumerate(sparsity_values):
        for j, importance in enumerate(importance_values):
            result = grid_results[(sparsity, importance)]
            
            # Check if expert 0 gets a specific allocation type (e.g., "dedicated")
            if "dedicated" in result['allocation_types'][0]:
                expert_0_patterns[i, j] = True
            
            # Check if expert 1 gets a specific allocation type
            if "dedicated" in result['allocation_types'][1]:
                expert_1_patterns[i, j] = True
    
    # Calculate spatial correlation
    expert_0_dedicated_rate = np.mean(expert_0_patterns)
    expert_1_dedicated_rate = np.mean(expert_1_patterns)
    
    print(f"Expert 0 'dedicated' allocation rate: {expert_0_dedicated_rate:.3f}")
    print(f"Expert 1 'dedicated' allocation rate: {expert_1_dedicated_rate:.3f}")
    
    # Check for systematic bias
    if abs(expert_0_dedicated_rate - expert_1_dedicated_rate) > 0.1:
        print("⚠️  WARNING: Significant bias detected in expert assignment!")
        if expert_0_dedicated_rate > expert_1_dedicated_rate:
            print("  Expert 0 is systematically favored for 'dedicated' allocations")
        else:
            print("  Expert 1 is systematically favored for 'dedicated' allocations")
    else:
        print("✅ Expert assignment appears relatively balanced")
    
    # Analyze by parameter regions
    print("\n4. Parameter Region Analysis:")
    
    # Split grid into regions
    mid_sparsity = len(sparsity_values) // 2
    mid_importance = len(importance_values) // 2
    
    regions = {
        'low_sparsity_low_importance': (0, mid_sparsity, 0, mid_importance),
        'low_sparsity_high_importance': (0, mid_sparsity, mid_importance, len(importance_values)),
        'high_sparsity_low_importance': (mid_sparsity, len(sparsity_values), 0, mid_importance),
        'high_sparsity_high_importance': (mid_sparsity, len(sparsity_values), mid_importance, len(importance_values))
    }
    
    for region_name, (i_start, i_end, j_start, j_end) in regions.items():
        region_expert_0_dedicated = 0
        region_expert_1_dedicated = 0
        region_total = 0
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if i < len(sparsity_values) and j < len(importance_values):
                    sparsity = sparsity_values[i]
                    importance = importance_values[j]
                    result = grid_results[(sparsity, importance)]
                    
                    if "dedicated" in result['allocation_types'][0]:
                        region_expert_0_dedicated += 1
                    if "dedicated" in result['allocation_types'][1]:
                        region_expert_1_dedicated += 1
                    region_total += 1
        
        if region_total > 0:
            print(f"  {region_name}:")
            print(f"    Expert 0 dedicated: {region_expert_0_dedicated}/{region_total} ({region_expert_0_dedicated/region_total:.1%})")
            print(f"    Expert 1 dedicated: {region_expert_1_dedicated}/{region_total} ({region_expert_1_dedicated/region_total:.1%})")
    
    print("\n=== End Bias Analysis ===")


def main():
    """
    Run the complete sparsity-importance grid experiment.
    """
    print("=== Sparsity-Importance Grid Experiment ===")
    print(f"Device: {DEVICE}")
    
    # Create experiment configuration
    config = GridExperimentConfig()
    
    # Train all models
    print("\n1. Training models...")
    training_results = train_grid_models(config)
    
    # Analyze expert allocations
    print("\n2. Analyzing expert allocations...")
    grid_results = analyze_grid_models(training_results, tolerance=config.tolerance)
    
    # Create plots
    print("\n3. Creating plots...")
    plot_allocation_grid(grid_results, config)
    plot_loss_grid_sparsity_last_feature(grid_results, config)
    plot_dimensions_per_feature_grid(grid_results, config)
    
    # Save results
    print("\n4. Saving results...")
    save_grid_results(grid_results, config)
    
    # Analyze potential systematic biases
    analyze_expert_bias(grid_results, config)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main() 



from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import torch
from model.model import DEVICE

@dataclass
class SparsityExpertConfig(GridExperimentConfig):
    """Configuration for sparsity vs expert count experiments"""
    
    # Override grid parameters
    sparsity_values: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 0.9, 9))
    expert_count_values: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    
    # Keep other parameters but set good defaults
    n_features: int = 4  # More features to make expert allocation interesting
    n_hidden: int = 2    # Slightly larger hidden dimension
    n_active_experts: int = 2  # Keep routing simple
    steps: int = 500     # Shorter training for faster experiments
    n_models_per_cell: int = 3  # Train multiple models per cell
    
    def __post_init__(self):
        # Don't call parent post_init since we're replacing the grid structure
        pass

def train_grid_models_sparsity_expert(config: SparsityExpertConfig) -> Dict[tuple, Any]:
    """Train models across a grid of sparsity values and expert counts"""
    
    results = {}
    
    for sparsity in config.sparsity_values:
        for n_experts in config.expert_count_values:
            print(f"Training models for sparsity={sparsity:.2f}, experts={n_experts}")
            
            # Set up model config for this cell
            config.n_experts = n_experts
            
            # Create feature probabilities and importance tensor
            feature_prob = 1 - torch.tensor(sparsity, device=DEVICE)
            feature_probs = torch.full((config.n_features,), feature_prob, device=DEVICE)
            importance_tensor = torch.ones(config.n_features, device=DEVICE)
            
            # Train multiple models for this cell
            best_loss = float('inf')
            best_result = None
            
            for model_idx in range(config.n_models_per_cell):
                # Train with unique random seed
                torch.manual_seed(np.random.randint(0, 10000))
                
                result = train_models_vectorized(
                    config,
                    feature_probs=feature_probs,
                    importance=importance_tensor,
                    print_freq=config.steps + 1  # Disable printing
                )
                
                # Keep best model based on loss
                if result['loss'].item() < best_loss:
                    best_loss = result['loss'].item()
                    best_result = result
            
            # Store best model's results
            results[(sparsity, n_experts)] = best_result
            
    return results

def analyze_grid_models_sparsity_expert(training_results: Dict[tuple, Any]) -> Dict[tuple, Any]:
    """Analyze trained models from sparsity vs expert count grid"""
    
    grid_results = {}
    
    for (sparsity, n_experts), result in training_results.items():
        # Extract model and its parameters
        model = result['model']
        model_params = {name: param.detach() for name, param in model.named_parameters()}
        
        # Get expert weights
        expert_weights = model.W_experts
        
        # Compute metrics for each expert
        expert_metrics = []
        for expert_idx in range(n_experts):
            weights = expert_weights[expert_idx].squeeze()
            
            # Compute norm and superposition
            norms, superposition = norm_and_superposition(weights.cpu().numpy())
            
            # Get expert selection patterns
            if hasattr(model, 'router_history'):
                selection_freq = model.router_history[expert_idx].mean().item()
            else:
                selection_freq = 0.0
                
            expert_metrics.append({
                'norms': norms,
                'superposition': superposition,
                'selection_freq': selection_freq
            })
        
        # Store results for this grid cell
        grid_results[(sparsity, n_experts)] = {
            'loss': result['loss'].item(),
            'expert_weights': expert_weights.cpu().detach().numpy(),
            'expert_metrics': expert_metrics,
            'gate_params': model_params['gate'].cpu().detach().numpy() if 'gate' in model_params else None,
            'model': model
        }
    
    return grid_results

def plot_metrics_grid(grid_results: Dict[tuple, Any], config: SparsityExpertConfig):
    """Plot comprehensive metrics for the sparsity vs expert count experiment"""
    
    # Create grids for different metrics
    loss_grid = np.zeros((len(config.sparsity_values), len(config.expert_count_values)))
    avg_superposition_grid = np.zeros_like(loss_grid)
    avg_norm_grid = np.zeros_like(loss_grid)
    
    # Fill grids
    for i, sparsity in enumerate(config.sparsity_values):
        for j, n_experts in enumerate(config.expert_count_values):
            result = grid_results[(sparsity, n_experts)]
            
            # Store loss
            loss_grid[i, j] = result['loss']
            
            # Average metrics across experts
            avg_super = np.mean([m['superposition'] for m in result['expert_metrics']])
            avg_norm = np.mean([np.mean(m['norms']) for m in result['expert_metrics']])
            
            avg_superposition_grid[i, j] = avg_super
            avg_norm_grid[i, j] = avg_norm
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Loss
    im1 = axes[0].imshow(loss_grid, cmap='viridis', aspect='auto')
    axes[0].set_title('Training Loss')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Average Superposition
    im2 = axes[1].imshow(avg_superposition_grid, cmap='viridis', aspect='auto')
    axes[1].set_title('Average Superposition')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Average Norm
    im3 = axes[2].imshow(avg_norm_grid, cmap='viridis', aspect='auto')
    axes[2].set_title('Average L2 Norm')
    plt.colorbar(im3, ax=axes[2])
    
    # Format all plots
    for ax in axes:
        ax.set_xlabel('Number of Experts')
        ax.set_ylabel('Feature Sparsity')
        
        # Set tick labels
        ax.set_xticks(range(len(config.expert_count_values)))
        ax.set_xticklabels(config.expert_count_values)
        
        y_step = max(1, len(config.sparsity_values) // 6)
        y_ticks = range(0, len(config.sparsity_values), y_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{config.sparsity_values[i]:.2f}' for i in y_ticks])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average loss: {np.mean(loss_grid):.3f} ± {np.std(loss_grid):.3f}")
    print(f"Average superposition: {np.mean(avg_superposition_grid):.3f} ± {np.std(avg_superposition_grid):.3f}")
    print(f"Average norm: {np.mean(avg_norm_grid):.3f} ± {np.std(avg_norm_grid):.3f}")
    
    # Find optimal configurations
    min_loss_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    best_sparsity = config.sparsity_values[min_loss_idx[0]]
    best_experts = config.expert_count_values[min_loss_idx[1]]
    
    print(f"\nBest configuration:")
    print(f"Sparsity: {best_sparsity:.2f}")
    print(f"Number of experts: {best_experts}")
    print(f"Loss: {np.min(loss_grid):.3f}")

# Example usage:
if __name__ == "__main__":
    # Create config
    config = SparsityExpertConfig()
    
    # Train models
    print("Training grid models...")
    training_results = train_grid_models_sparsity_expert(config)
    
    # Analyze results
    print("\nAnalyzing results...")
    grid_results = analyze_grid_models_sparsity_expert(training_results)
    
    # Plot metrics
    print("\nPlotting metrics...")
    plot_metrics_grid(grid_results, config)
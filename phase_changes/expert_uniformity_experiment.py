import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import time

# Add parent directory to path to import from model and helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Config, MoEModel, optimize, make_functional_model, vectorized_forward, generate_vectorized_batch, stack_state_dicts, optimize_vectorized
from helpers.helpers import compute_all_expert_probabilities

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# @dataclass
# class ExperimentConfig:
#     n_features: int = 16
#     n_hidden: int = 8
#     n_active_experts: int = 1
#     expert_counts: List[int] = None
#     feature_probability: float = 0.1
#     n_batch: int = 512
#     steps: int = 3000
#     lr: float = 1e-3
#     n_samples: int = 10000
    
#     def __post_init__(self):
#         if self.expert_counts is None:
#             self.expert_counts = [1] + list(range(4, 32+1, 2))
@dataclass
class ExperimentConfig:
    n_features: int = 2
    n_hidden: int = 1
    n_active_experts: int = 1
    expert_counts: List[int] = None
    feature_probability: float = 0.1
    n_batch: int = 512
    steps: int = 4000
    lr: float = 1e-3
    n_samples: int = 10000
    
    def __post_init__(self):
        if self.expert_counts is None:
            self.expert_counts = [1] + list(range(1, 4+1, 1))

def compute_uniformity_metric(expert_probabilities: np.ndarray) -> float:
    """
    Compute uniformity metric for expert distribution.
    
    This measures how far the expert probabilities deviate from perfect uniformity.
    
    Examples:
    - Perfect uniformity: [0.25, 0.25, 0.25, 0.25] -> uniformity = 0.0
    - Slight non-uniformity: [0.3, 0.25, 0.25, 0.2] -> uniformity ≈ 0.2
    - Highly non-uniform: [0.8, 0.1, 0.05, 0.05] -> uniformity ≈ 0.9
    - Maximum non-uniformity: [1.0, 0.0, 0.0, 0.0] -> uniformity = 1.0
    
    Args:
        expert_probabilities: Array of shape [n_experts] with probabilities for each expert
    
    Returns:
        Uniformity metric (0 = perfectly uniform, 1 = perfectly non-uniform)
    """
    n_experts = len(expert_probabilities)
    expected_uniform = 1.0 / n_experts
    
    # Compute variance from uniform distribution
    variance = np.var(expert_probabilities)
    max_variance = (1.0 - expected_uniform) ** 2 * (n_experts - 1) / n_experts
    
    # Normalize to [0, 1] where 0 = perfectly uniform, 1 = perfectly non-uniform
    uniformity = variance / max_variance if max_variance > 0 else 0.0
    
    return uniformity

def train_and_analyze_model(n_experts: int, config: ExperimentConfig) -> Dict[str, Any]:
    """
    Train a single model with n_experts and analyze its router behavior.
    
    Args:
        n_experts: Number of experts in the model
        config: Experiment configuration
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Training model with {n_experts} experts...")
    
    # Create model configuration
    model_config = Config(
        n_features=config.n_features,
        n_hidden=config.n_hidden,
        n_experts=n_experts,
        n_active_experts=config.n_active_experts,
        load_balancing_loss=True
    )
    
    # Create model
    model = MoEModel(
        config=model_config,
        device=DEVICE,
        importance=torch.ones(config.n_features),
        feature_probability=torch.tensor(config.feature_probability)
    )
    
    # Train the model
    optimize(
        model, 
        n_batch=config.n_batch, 
        steps=config.steps, 
        print_freq=config.steps//10, 
        lr=config.lr
    )
    
    # Analyze router behavior
    gate_matrix = model.gate.detach()
    feature_prob = torch.tensor(config.feature_probability)
    
    # Compute expert probabilities
    probs_with_data, probs_without_data = compute_all_expert_probabilities(
        gate_matrix, 
        feature_probability=feature_prob, 
        n_samples=config.n_samples
    )
    
    print(f"Debug - Raw probs with data: {probs_with_data}")
    print(f"Debug - Raw probs without data: {probs_without_data}")
    
    # Debug: Let's also look at the gate matrix to understand the router behavior
    print(f"Debug - Gate matrix:\n{gate_matrix}")
    print(f"Debug - Feature probability: {feature_prob}")
    
    # Compute uniformity metrics
    uniformity_with_data = compute_uniformity_metric(probs_with_data)
    uniformity_without_data = compute_uniformity_metric(probs_without_data)
    
    print(f"Debug - Uniformity with data: {uniformity_with_data}")
    print(f"Debug - Uniformity without data: {uniformity_without_data}")
    
    # Debug: Let's also check what the expected uniform distribution would be
    n_experts = len(probs_with_data)
    expected_uniform = 1.0 / n_experts
    print(f"Debug - Expected uniform probability: {expected_uniform}")
    print(f"Debug - Deviation from uniform (sparse): {np.abs(probs_with_data - expected_uniform)}")
    print(f"Debug - Deviation from uniform (less sparse): {np.abs(probs_without_data - expected_uniform)}")
    
    # Compute additional statistics
    mean_prob_with_data = np.mean(probs_with_data)
    mean_prob_without_data = np.mean(probs_without_data)
    std_prob_with_data = np.std(probs_with_data)
    std_prob_without_data = np.std(probs_without_data)
    
    return {
        'n_experts': n_experts,
        'uniformity_with_data': uniformity_with_data,
        'uniformity_without_data': uniformity_without_data,
        'probs_with_data': probs_with_data,
        'probs_without_data': probs_without_data,
        'mean_prob_with_data': mean_prob_with_data,
        'mean_prob_without_data': mean_prob_without_data,
        'std_prob_with_data': std_prob_with_data,
        'std_prob_without_data': std_prob_without_data,
        'gate_matrix': gate_matrix.cpu().numpy()
    }

def run_uniformity_experiment(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """
    Run the uniformity experiment across all expert counts.
    
    Args:
        config: Experiment configuration
    
    Returns:
        List of results for each expert count
    """
    results = []
    
    print(f"Running uniformity experiment with expert counts: {config.expert_counts}")
    print(f"Parameters: n_features={config.n_features}, n_hidden={config.n_hidden}, k={config.n_active_experts}")
    
    for i, n_experts in enumerate(config.expert_counts):
        print(f"\n--- Experiment {i+1}/{len(config.expert_counts)} ---")
        
        try:
            result = train_and_analyze_model(n_experts, config)
            results.append(result)
            
            print(f"Results for {n_experts} experts:")
            print(f"  Uniformity (with data): {result['uniformity_with_data']:.4f}")
            print(f"  Uniformity (without data): {result['uniformity_without_data']:.4f}")
            print(f"  Mean prob (with data): {result['mean_prob_with_data']:.4f}")
            print(f"  Std prob (with data): {result['std_prob_with_data']:.4f}")
            
        except Exception as e:
            print(f"Error training model with {n_experts} experts: {e}")
            # Add a placeholder result
            results.append({
                'n_experts': n_experts,
                'uniformity_with_data': np.nan,
                'uniformity_without_data': np.nan,
                'probs_with_data': np.array([np.nan] * n_experts),
                'probs_without_data': np.array([np.nan] * n_experts),
                'mean_prob_with_data': np.nan,
                'mean_prob_without_data': np.nan,
                'std_prob_with_data': np.nan,
                'std_prob_without_data': np.nan,
                'gate_matrix': np.array([])
            })
    
    return results

def plot_uniformity_results(results: List[Dict[str, Any]], config: ExperimentConfig):
    """
    Create 2D plots of the uniformity experiment results.
    
    Args:
        results: List of results from the experiment
        config: Experiment configuration
    """
    # Extract data
    expert_counts = [r['n_experts'] for r in results]
    uniformities_with_data = [r['uniformity_with_data'] for r in results]
    uniformities_without_data = [r['uniformity_without_data'] for r in results]
    
    # Debug: Print the data to see what's happening
    print(f"\nDebug - Expert counts: {expert_counts}")
    print(f"Debug - Uniformities with data: {uniformities_with_data}")
    print(f"Debug - Uniformities without data: {uniformities_without_data}")
    
    # Filter out NaN values - check both data sources
    valid_indices = []
    for i in range(len(results)):
        if not np.isnan(uniformities_with_data[i]) or not np.isnan(uniformities_without_data[i]):
            valid_indices.append(i)
    
    expert_counts_valid = [expert_counts[i] for i in valid_indices]
    uniformities_with_data_valid = [uniformities_with_data[i] for i in valid_indices]
    uniformities_without_data_valid = [uniformities_without_data[i] for i in valid_indices]
    
    print(f"Debug - Valid indices: {valid_indices}")
    print(f"Debug - Valid expert counts: {expert_counts_valid}")
    print(f"Debug - Valid uniformities with data: {uniformities_with_data_valid}")
    print(f"Debug - Valid uniformities without data: {uniformities_without_data_valid}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Uniformity vs Number of Experts
    ax1 = axes[0, 0]
    ax1.plot(expert_counts_valid, uniformities_with_data_valid, 'o-', label='Sparse data (training-like)', linewidth=2, markersize=8)
    ax1.plot(expert_counts_valid, uniformities_without_data_valid, 's-', label='Less sparse data', linewidth=2, markersize=8)
    
    # Add vertical line at n_experts = n_features
    if config.n_features in expert_counts_valid:
        ax1.axvline(x=config.n_features, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'n_experts = n_features ({config.n_features})')
    
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Uniformity Metric')
    ax1.set_title('Expert Distribution Uniformity vs Number of Experts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean probability vs Number of Experts
    ax2 = axes[0, 1]
    mean_probs_with_data = [r['mean_prob_with_data'] for i, r in enumerate(results) if i in valid_indices]
    mean_probs_without_data = [r['mean_prob_without_data'] for i, r in enumerate(results) if i in valid_indices]
    
    print(f"Debug - Mean probs with data: {mean_probs_with_data}")
    print(f"Debug - Mean probs without data: {mean_probs_without_data}")
    
    ax2.plot(expert_counts_valid, mean_probs_with_data, 'o-', label='Sparse data (training-like)', linewidth=2, markersize=8)
    ax2.plot(expert_counts_valid, mean_probs_without_data, 's-', label='Less sparse data', linewidth=2, markersize=8)
    
    # Plot perfect uniformity reference (1/n_experts)
    perfect_uniform = [1.0/n for n in expert_counts_valid]
    ax2.plot(expert_counts_valid, perfect_uniform, 'r--', label='Perfect uniform (1/n_experts)', linewidth=2)
    
    # Add vertical line at n_experts = n_features
    if config.n_features in expert_counts_valid:
        ax2.axvline(x=config.n_features, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'n_experts = n_features ({config.n_features})')
    
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Mean Expert Probability')
    ax2.set_title('Mean Expert Probability vs Number of Experts')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Standard deviation vs Number of Experts
    ax3 = axes[1, 0]
    std_probs_with_data = [r['std_prob_with_data'] for i, r in enumerate(results) if i in valid_indices]
    std_probs_without_data = [r['std_prob_without_data'] for i, r in enumerate(results) if i in valid_indices]
    
    ax3.plot(expert_counts_valid, std_probs_with_data, 'o-', label='Sparse data (training-like)', linewidth=2, markersize=8)
    ax3.plot(expert_counts_valid, std_probs_without_data, 's-', label='Less sparse data', linewidth=2, markersize=8)
    
    # Plot theoretical std dev for perfect uniformity (should be 0)
    perfect_std = [0.0] * len(expert_counts_valid)
    ax3.plot(expert_counts_valid, perfect_std, 'r--', label='Perfect uniform (std=0)', linewidth=2)
    
    # Add vertical line at n_experts = n_features
    if config.n_features in expert_counts_valid:
        ax3.axvline(x=config.n_features, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'n_experts = n_features ({config.n_features})')
    
    ax3.set_xlabel('Number of Experts')
    ax3.set_ylabel('Standard Deviation of Expert Probability')
    ax3.set_title('Expert Probability Standard Deviation vs Number of Experts\n(Lower = more uniform distribution)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of expert probabilities for largest model
    ax4 = axes[1, 1]
    largest_model_idx = valid_indices[-1] if valid_indices else 0
    largest_result = results[largest_model_idx]
    
    if len(largest_result['probs_with_data']) > 0 and not np.isnan(largest_result['probs_with_data'][0]):
        probs_matrix = np.array([
            largest_result['probs_with_data'],
            largest_result['probs_without_data']
        ])
        
        im = ax4.imshow(probs_matrix, cmap='viridis', aspect='auto')
        ax4.set_xlabel('Expert Index')
        ax4.set_ylabel('Data Type')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['With Data', 'Without Data'])
        ax4.set_title(f'Expert Probabilities (n_experts={largest_result["n_experts"]})')
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'No valid data for heatmap', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Expert Probabilities Heatmap')
    
    plt.suptitle(f'Expert Uniformity Experiment\nn_features={config.n_features}, n_hidden={config.n_hidden}, k={config.n_active_experts}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('phase_changes/results', exist_ok=True)
    plt.savefig('phase_changes/results/expert_uniformity_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_results(results: List[Dict[str, Any]], config: ExperimentConfig):
    """
    Save experiment results to file.
    
    Args:
        results: List of results from the experiment
        config: Experiment configuration
    """
    import json
    
    # Prepare data for saving
    save_data = {
        'config': {
            'n_features': config.n_features,
            'n_hidden': config.n_hidden,
            'n_active_experts': config.n_active_experts,
            'expert_counts': config.expert_counts,
            'feature_probability': config.feature_probability,
            'n_batch': config.n_batch,
            'steps': config.steps,
            'lr': config.lr,
            'n_samples': config.n_samples
        },
        'results': []
    }
    
    for result in results:
        # Convert numpy arrays to lists for JSON serialization
        save_result = {
            'n_experts': result['n_experts'],
            'uniformity_with_data': float(result['uniformity_with_data']),
            'uniformity_without_data': float(result['uniformity_without_data']),
            'probs_with_data': result['probs_with_data'].tolist(),
            'probs_without_data': result['probs_without_data'].tolist(),
            'mean_prob_with_data': float(result['mean_prob_with_data']),
            'mean_prob_without_data': float(result['mean_prob_without_data']),
            'std_prob_with_data': float(result['std_prob_with_data']),
            'std_prob_without_data': float(result['std_prob_without_data'])
        }
        save_data['results'].append(save_result)
    
    # Save to file
    os.makedirs('phase_changes/results', exist_ok=True)
    with open('phase_changes/results/expert_uniformity_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to phase_changes/results/expert_uniformity_results.json")

def test_uniformity_concepts():
    """
    Test and demonstrate the uniformity concepts with examples.
    """
    print("\n" + "="*50)
    print("UNIFORMITY CONCEPT DEMONSTRATION")
    print("="*50)
    
    # Test cases
    test_cases = [
        ("Perfect uniformity (4 experts)", [0.25, 0.25, 0.25, 0.25]),
        ("Slight non-uniformity (4 experts)", [0.3, 0.25, 0.25, 0.2]),
        ("Moderate non-uniformity (4 experts)", [0.5, 0.3, 0.15, 0.05]),
        ("High non-uniformity (4 experts)", [0.8, 0.1, 0.05, 0.05]),
        ("Maximum non-uniformity (4 experts)", [1.0, 0.0, 0.0, 0.0]),
        ("Perfect uniformity (8 experts)", [0.125] * 8),
        ("Non-uniformity (8 experts)", [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    ]
    
    for name, probs in test_cases:
        uniformity = compute_uniformity_metric(np.array(probs))
        std_dev = np.std(probs)
        mean_prob = np.mean(probs)
        expected_uniform = 1.0 / len(probs)
        
        print(f"\n{name}:")
        print(f"  Probabilities: {probs}")
        print(f"  Mean: {mean_prob:.4f} (expected: {expected_uniform:.4f})")
        print(f"  Std Dev: {std_dev:.4f}")
        print(f"  Uniformity: {uniformity:.4f}")

def print_experiment_explanation():
    """
    Print explanation of the experiment concepts.
    """
    print("\n" + "="*60)
    print("EXPERIMENT CONCEPT EXPLANATION")
    print("="*60)
    
    print("\n1. DATA DISTRIBUTIONS:")
    print("   - 'Sparse data (training-like)': Uses the SAME distribution as training data")
    print("   -   (feature_probability=0.1, matches model.generate_batch())")
    print("   - 'Less sparse data': Same distribution but with feature_probability=0.2")
    print("   - This shows how router behavior changes with data density\n")
    
    print("2. PERFECT UNIFORMITY:")
    print("   - For n experts, each should have probability 1/n")
    print("   - Example: 4 experts → [0.25, 0.25, 0.25, 0.25]")
    print("   - Example: 8 experts → [0.125, 0.125, ..., 0.125]")
    
    print("\n3. UNIFORMITY METRIC:")
    print("   - 0.0 = perfectly uniform (all experts equally likely)")
    print("   - 1.0 = perfectly non-uniform (one expert gets everything)")
    print("   - Examples:")
    print("     [0.25, 0.25, 0.25, 0.25] → uniformity = 0.0")
    print("     [0.8, 0.1, 0.05, 0.05] → uniformity ≈ 0.9")
    
    print("\n4. STANDARD DEVIATION:")
    print("   - Measures how 'spread out' the expert probabilities are")
    print("   - 0.0 = all experts have same probability (perfect uniformity)")
    print("   - Higher values = some experts much more likely than others")
    print("   - Example: [0.25, 0.25, 0.25, 0.25] → std = 0.0")
    print("   - Example: [0.8, 0.1, 0.05, 0.05] → std ≈ 0.35")
    
    print("\n5. ROUTER SELECTION HOOK vs UNIFORMITY METRIC:")
    print("   - Router Hook: Tracks ACTUAL selections during training with REAL data")
    print("   - Uniformity Metric: Predicts selections with SYNTHETIC data")
    print("   - They should now match better since we use the same data distribution")
    print("   - The hook shows what happened, the metric shows what would happen")
    
    print("\n6. WHAT TO EXPECT:")
    print("   - More experts → harder to maintain uniformity")
    print("   - Sparse data → less uniform (experts specialize)")
    print("   - Less sparse data → more uniform (experts balance)")
    print("="*60 + "\n")

def main():
    """
    Main function to run the expert uniformity experiment.
    """
    print("=== Expert Uniformity Experiment ===")
    print(f"Device: {DEVICE}")
    
    # Print explanation and test concepts
    print_experiment_explanation()
    test_uniformity_concepts()
    
    # Create experiment configuration
    config = ExperimentConfig()
    
    # Run the experiment
    start_time = time.time()
    results = run_uniformity_experiment(config)
    end_time = time.time()
    
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")
    
    # Plot results
    plot_uniformity_results(results, config)
    
    # Save results
    save_results(results, config)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    for result in results:
        if not np.isnan(result['uniformity_with_data']):
            print(f"n_experts={result['n_experts']:2d}: "
                  f"uniformity_with_data={result['uniformity_with_data']:.4f}, "
                  f"uniformity_without_data={result['uniformity_without_data']:.4f}")

if __name__ == "__main__":
    main() 
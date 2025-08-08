# Examples in the dimensions_per_feature.ipynb notebook

from __future__ import annotations
import torch
import numpy as np
from matplotlib import pyplot as plt
from helpers.dimensions_per_feature import compute_dimensions_per_feature_single
from typing import Sequence, Literal, List
from typing import Tuple

def norm_and_superposition(
    W: Sequence[Sequence[float]] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    W
        An (n_features, d) array-like of feature / weight vectors.
        If you pass a single 1-D vector, it will be treated as shape (1, d).

    Returns
    -------
    norms         : shape (n_features,)   L2 norms  ‖Wᵢ‖₂
    superposition : shape (n_features,)   ∑ⱼ (x̂ᵢ·x̂ⱼ)²  with j ≠ i
    """
    if torch.is_tensor(W):
        W = W.detach().cpu().numpy()
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:                       # allow a single vector
        W = W[None, :]

    # --- 1. Norms -----------------------------------------------------------
    norms = np.linalg.norm(W, axis=1)     # ‖Wᵢ‖₂ for every row

    # --- 2. Normalise & build cosine-similarity matrix ----------------------
    # Handle zero vectors safely:
    W_hat = np.divide(W, norms[:, None], where=norms[:, None] != 0)

    cos = W_hat @ W.T                 # cos[i, j] = x̂ᵢ·x̂ⱼ

    # superᵢ = Σⱼ (cos[i, j])² but exclude j = i
    superposition = (cos**2).sum(axis=1) - 1.0   # subtract self-term (1²)

    return norms, superposition


import matplotlib.pyplot as plt

MoEClass = Literal[
    "ignore_all",       # (≈0, ≈0, ≈0)
    "monosemantic",     # one dominant coord (e.g. 0.9, 0.05, 0.01)
    "superimpose_two",  # two significant coords (e.g. 0.6, 0.6, 0.02)
    "superimpose_three" # three significant coords (e.g. 0.4, 0.4, 0.3)
]

def classify_moe_vector(
    v: Sequence[float],
    *,
    abs_tol: float = 1e-6,      # "close to zero" threshold
    rel_sig: float = 0.10,      # coord is "significant" if ≥ rel_sig · max(|v|)
    dominance: float = 0.80     # monosemantic if the largest coord ≥ dominance · ‖v‖₁
) -> MoEClass:
    """
    Classify a 3-D weight vector for a toy MoE neuron.

    Parameters
    ----------
    v : length-3 iterable of floats
    abs_tol : absolute tolerance for treating a weight as zero
    rel_sig : relative threshold (fraction of max |v|) for a coord to count
    dominance : fraction of L1 norm the biggest coord must hold to be "mono"

    Returns
    -------
    One of: "ignore_all", "monosemantic", "superimpose_two", "superimpose_three"
    """
    if len(v) != 3:
        raise ValueError("Expected a length-3 vector.")

    # Absolute-zero check
    if all(abs(x) <= abs_tol for x in v):
        return "ignore_all"

    # Significant coordinates
    m = max(abs(x) for x in v)
    sig = [abs(x) >= rel_sig * m for x in v]
    n_sig = sum(sig)

    # Monosemantic test (one coord owns the bulk of the mass)
    l1 = sum(abs(x) for x in v)
    if max(abs(x) for x in v) >= dominance * l1:
        return "monosemantic"

    # Otherwise decide by count of significant coords
    if n_sig == 2:
        return "superimpose_two"
    else:                # n_sig will be 3 here by elimination
        return "superimpose_three"



def compute_expert_selection_n3_n1(model):
    """
    Compute expert selection for a 3-feature, 1-hidden model.
    
    Returns:
        List of classifications for each expert using the MoE classification scheme.
    """
    device = next(model.parameters()).device
    n_experts = model.config.n_experts
    
    expert_classifications = []
    
    # Classify each expert based on their weight vector
    for expert_idx in range(n_experts):
        # Get expert weights for this expert
        expert_weights = model.W_experts[expert_idx].squeeze().detach().cpu().numpy()  # Shape: [3]
        
        # Classify the expert using the MoE classification function
        classification = classify_moe_vector(expert_weights)
        
        expert_classifications.append(classification)
        
        # Debug info
        print(f"Expert {expert_idx}:")
        print(f"  Weights: {expert_weights}")
        print(f"  Classification: {classification}")
        print()
    
    return expert_classifications


def compute_expert_selection_n3_n1_special_third(model):
    """
    Compute expert selection for a 3-feature, 1-hidden model with SPECIAL third coordinate.
    
    Returns:
        List of classifications for each expert using the special third coordinate scheme.
    """
    device = next(model.parameters()).device
    n_experts = model.config.n_experts
    
    expert_classifications = []
    
    # Classify each expert based on their weight vector
    for expert_idx in range(n_experts):
        # Get expert weights for this expert
        expert_weights = model.W_experts[expert_idx].squeeze().detach().cpu().numpy()  # Shape: [3]
        
        # Classify the expert using the special third coordinate classification function
        classification = classify_moe_vector_special_third(expert_weights)
        
        expert_classifications.append(classification)
        
        # Debug info
        print(f"Expert {expert_idx} (special third):")
        print(f"  Weights: {expert_weights}")
        print(f"  Classification: {classification}")
        print()
    
    return expert_classifications


MoEClassSpecial = Literal[
    "ignore_all",            # (≈0, ≈0, ≈0)
    "mono_1_or_2",           # dominant feature is 1 or 2
    "mono_3",                # dominant feature is 3
    "super_12",              # significant on 1 & 2 only
    "super_with_3",          # significant on 3 plus one of {1,2}
    "super_123"              # all three significant, none dominant
]

def _significant_mask_special(vals: Sequence[float], rel_sig: float) -> List[bool]:
    """Return a boolean mask marking coords that are ≥ rel_sig · max(|v|)."""
    m = max(abs(x) for x in vals)
    return [abs(x) >= rel_sig * m for x in vals]

def classify_moe_vector_special_third(
    v: Sequence[float],
    *,
    abs_tol: float = 1e-6,   # "close to zero"
    rel_sig: float = 0.10,   # coord counts if ≥ rel_sig · max(|v|)
    dominance: float = 0.80  # monosem if biggest coord ≥ dominance · ‖v‖₁
) -> MoEClassSpecial:
    """
    Classify a 3-D weight vector with a SPECIAL third coordinate.

    Returns one of:
      - ignore_all
      - mono_1_or_2   (mono and winner is coord 0 or 1)
      - mono_3        (mono and winner is coord 2)
      - super_12      (sig on 1&2 only)
      - super_with_3  (sig on 3 plus 1 or 2)
      - super_123     (sig on all three)
    """
    if len(v) != 3:
        raise ValueError("Expected a length-3 vector.")

    # 1) Ignore-all check
    if all(abs(x) <= abs_tol for x in v):
        return "ignore_all"

    # 2) Monosemantic?
    l1 = sum(abs(x) for x in v)
    max_val = max(abs(x) for x in v)
    max_idx = max(range(3), key=lambda i: abs(v[i]))
    if max_val >= dominance * l1:
        return "mono_3" if max_idx == 2 else "mono_1_or_2"

    # 3) Superposition cases
    sig = _significant_mask_special(v, rel_sig)
    sig_idx = [i for i, s in enumerate(sig) if s]

    if len(sig_idx) == 2:
        if set(sig_idx) == {0, 1}:          # 1 & 2 significant, 3 tiny
            return "super_12"
        else:                               # involves coordinate 3
            return "super_with_3"
    else:                                   # len == 3 by elimination
        return "super_123"


def plot_expert_selection_m2_e2(model, importance_vector, step=0.01, f1=0, f2=-1):
    """
    Plot expert selection comparing first and last features based on importance vector.
    
    Args:
        model: The MoE model
        importance_vector: Vector of feature importances (e.g., [1, 1, 1] for 3 features)
        step: Step size for the grid
    """
    # Get model parameters
    n_features = model.config.n_features
    device = next(model.parameters()).device
    
    # Convert importance to tensor if it's not already
    if not isinstance(importance_vector, torch.Tensor):
        importance_vector = torch.tensor(importance_vector, device=device, dtype=torch.float32)
    
    # Determine the maximum values based on importance
    # Use importance as a scaling factor for the maximum
    max_val = max(importance_vector.max().item(), 1.0)  # At least 1.0
    
    # Create grid for first and last features
    feature_1_values = np.arange(0, max_val, step)
    feature_last_values = np.arange(0, max_val, step)
    feature_1_grid, feature_last_grid = np.meshgrid(feature_1_values, feature_last_values)
    
    # Compute the expert selection for each combination
    expert_selection = np.zeros((len(feature_1_values), len(feature_last_values)))
    
    for i in range(len(feature_1_values)):
        for j in range(len(feature_last_values)):
            # Create feature vector with zeros for middle features
            features = torch.zeros(1, n_features, device=device, dtype=torch.float32)
            features[0, 0] = feature_1_grid[i, j]  # First feature
            features[0, -1] = feature_last_grid[i, j]  # Last feature
            
            # Get expert selection probabilities
            expert_weights, top_k_indices, _ = model.compute_active_experts(features)
            
            # For k=1, we want the probability of the first expert being selected
            expert_selection[i, j] = expert_weights[0, 0].item()  # Probability of expert 0
            
            # Debug: print some values to see what's happening
            # if i == 0 and j == 0:
            #     print(f"Debug - Features: {features}")
            #     print(f"Debug - Expert weights shape: {expert_weights.shape}")
            #     print(f"Debug - Expert weights: {expert_weights}")
            #     print(f"Debug - Top k indices: {top_k_indices}")
            #     print(f"Debug - Selected probability: {expert_selection[i, j]}")
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    plt.contourf(feature_1_grid, feature_last_grid, expert_selection, levels=20, cmap='viridis')
    cbar = plt.colorbar(label='Expert 0 Selection Probability')
    cbar.ax.tick_params(labelsize=10)
    
    plt.xlabel(f'Feature 1 (max={max_val:.2f})', fontsize=12, labelpad=10)
    plt.ylabel(f'Feature {n_features} (max={max_val:.2f})', fontsize=12, labelpad=10)
    plt.title(f'Expert 0 Selection Probability: Feature 1 vs Feature {n_features}', fontsize=14, pad=15)
    
    # Improve tick label spacing with fewer ticks
    # Reduce number of ticks for better legibility
    x_ticks = np.linspace(0, len(feature_1_values)-1, min(8, len(feature_1_values)), dtype=int)
    y_ticks = np.linspace(0, len(feature_last_values)-1, min(6, len(feature_last_values)), dtype=int)
    
    plt.xticks(x_ticks, [f'{feature_1_values[i]:.2f}' for i in x_ticks], fontsize=10)
    plt.yticks(y_ticks, [f'{feature_last_values[i]:.2f}' for i in y_ticks], fontsize=10)
    
    plt.tight_layout(pad=2.0)
    plt.show()


def plot_data_distribution_m2_e2(model, min_m1, min_m2, max_m1, max_m2, sparsity=0.1, n_samples=10000, step=0.01):
    """
    Plot distribution of data points (features=2) for different values of m1 and m2.
    """
    # Make bins for m1 and m2
    m1_bins = np.arange(min_m1, max_m1, step)
    m2_bins = np.arange(min_m2, max_m2, step)

    # Make a grid of m1 and m2 values 
    m1_grid, m2_grid = np.meshgrid(m1_bins, m2_bins)

    # Generate data and add to grid
    

    # Make a grid of m1 and m2 values 
    # m1_values = np.arange(min_m1, max_m1, step)
    # m2_values = np.arange(min_m2, max_m2, step)
    # m1_grid, m2_grid = np.meshgrid(m1_values, m2_values)

    # # Compute the expert selection for each value of m1 and m2
    # expert_selection = np.zeros((len(m1_values), len(m2_values)))
    # for i in range(len(m1_values)):
    #     for j in range(len(m2_values)):
    #         expert_selection[i, j] = model.compute_active_experts(m1_grid[i, j])
    
    # # Create a figure
    # plt.figure(figsize=(10, 6))
    # plt.contourf(m1_grid, expert_selection, levels=20, cmap='viridis')
    # plt.colorbar(label='Expert Selection')
    # plt.xlabel('m1')
    # plt.ylabel('m2')
    # plt.title('Expert Selection for Different Values of m1 and m2')
    # plt.show()
    pass
    
def plot_expert_selection_distribution_m2_e2(model, min_m1, min_m2, max_m1, max_m2, step=0.01):
    """
    Plot distribution of expert selection (features=2, experts=2) for different values of m1 and m2.
    """
    # Make a grid of m1 and m2 values 
    # m1_values = np.arange(min_m1, max_m1, step)
    # m2_values = np.arange(min_m2, max_m2, step)
    # m1_grid, m2_grid = np.meshgrid(m1_values, m2_values)

    # # Compute the expert selection for each value of m1 and m2
    # expert_selection = np.zeros((len(m1_values), len(m2_values)))
    # for i in range(len(m1_values)):
    #     for j in range(len(m2_values)):
    #         expert_selection[i, j] = model.compute_active_experts(m1_grid[i, j])
    
    # # Create a figure
    # plt.figure(figsize=(10, 6))
    # plt.contourf(m1_grid, expert_selection, levels=20, cmap='viridis')
    # plt.colorbar(label='Expert Selection')
    # plt.xlabel('m1')
    # plt.ylabel('m2')
    # plt.title('Expert Selection for Different Values of m1 and m2')
    # plt.show()
    pass
    
def classify_expert_feature_allocation(analysis_result, tolerance=0.1):
    """
    Classify how an expert has allocated its hidden dimensions among features.
    
    Args:
        analysis_result: Output from compute_dimensions_per_feature_single
        tolerance: Tolerance for determining if features are dedicated/superimposed
    
    Returns:
        Dictionary with classification details:
        - 'allocation_type': Main classification
        - 'subtype': More specific details
        - 'feature_assignments': How each feature is allocated
        - 'superposition_groups': Groups of features that are superimposed
        - 'ignored_features': Features that are effectively ignored
        - 'dedicated_features': Features with dedicated dimensions
    """
    feature_dims = analysis_result['feature_dimensionalities']
    feature_norms = analysis_result['feature_norms']
    feature_geometry = analysis_result['feature_geometry']
    n_features = analysis_result['n_features']
    n_hidden = analysis_result['n_hidden']
    
    # Convert to numpy for easier analysis
    if torch.is_tensor(feature_dims[0]):
        feature_dims = [d.item() if torch.is_tensor(d) else d for d in feature_dims]
    if torch.is_tensor(feature_norms[0]):
        feature_norms = [n.item() if torch.is_tensor(n) else n for n in feature_norms]
    
    # Determine feature assignments
    feature_assignments = []
    dedicated_features = []
    ignored_features = []
    superposition_groups = []
    
    # Check for ignored features (very low norm)
    for i, norm in enumerate(feature_norms):
        if norm < tolerance:
            ignored_features.append(i)
            feature_assignments.append(f'ignored_{i}')
        else:
            feature_assignments.append('unknown')
    
    # Check for dedicated features (dimensionality close to 1.0)
    for i, dim in enumerate(feature_dims):
        if i not in ignored_features and abs(dim - 1.0) < tolerance:
            dedicated_features.append(i)
            feature_assignments[i] = f'dedicated_{i}'
    
    # Find superposition groups
    remaining_features = [i for i in range(n_features) if i not in ignored_features and i not in dedicated_features]
    
    if len(remaining_features) > 1:
        # Check for superposition patterns
        for i in remaining_features:
            for j in remaining_features[i+1:]:
                # Check if features i and j are superimposed
                if torch.is_tensor(feature_geometry):
                    geometry_ij = feature_geometry[i, j].item()
                else:
                    geometry_ij = feature_geometry[i, j]
                
                if geometry_ij > 0.8:  # High correlation indicates superposition
                    # Check if this pair is already in a group
                    found_group = False
                    for group in superposition_groups:
                        if i in group or j in group:
                            if i not in group:
                                group.append(i)
                            if j not in group:
                                group.append(j)
                            found_group = True
                            break
                    
                    if not found_group:
                        superposition_groups.append([i, j])
    
    # Update feature assignments for superimposed features
    for group in superposition_groups:
        group_str = f'superimposed_{"_".join(map(str, sorted(group)))}'
        for feature_idx in group:
            feature_assignments[feature_idx] = group_str
    
    # Determine main allocation type
    n_dedicated = len(dedicated_features)
    n_ignored = len(ignored_features)
    n_superimposed = sum(len(group) for group in superposition_groups)
    n_unassigned = n_features - n_dedicated - n_ignored - n_superimposed
    
    # Classification logic
    if n_ignored == n_features:
        allocation_type = 'completely_ignored'
        subtype = f'all_{n_features}_features_ignored'
    elif n_dedicated == n_features:
        allocation_type = 'fully_dedicated'
        subtype = f'all_{n_features}_features_dedicated'
    elif n_dedicated > 0 and n_superimposed == 0 and n_ignored == 0:
        allocation_type = 'partially_dedicated'
        subtype = f'{n_dedicated}_dedicated_{n_unassigned}_unassigned'
    elif len(superposition_groups) > 0 and n_dedicated == 0 and n_ignored == 0:
        allocation_type = 'fully_superimposed'
        subtype = f'{len(superposition_groups)}_superposition_groups'
    elif len(superposition_groups) > 0 and n_dedicated > 0:
        allocation_type = 'mixed_dedicated_superimposed'
        subtype = f'{n_dedicated}_dedicated_{len(superposition_groups)}_superposition_groups'
    elif n_ignored > 0 and n_dedicated > 0:
        allocation_type = 'mixed_dedicated_ignored'
        subtype = f'{n_dedicated}_dedicated_{n_ignored}_ignored'
    elif n_ignored > 0 and len(superposition_groups) > 0:
        allocation_type = 'mixed_superimposed_ignored'
        subtype = f'{len(superposition_groups)}_superposition_groups_{n_ignored}_ignored'
    elif n_ignored > 0 and n_dedicated == 0 and len(superposition_groups) == 0:
        allocation_type = 'partially_ignored'
        subtype = f'{n_ignored}_ignored_{n_unassigned}_unassigned'
    else:
        allocation_type = 'complex_mixed'
        subtype = f'{n_dedicated}_dedicated_{len(superposition_groups)}_superposition_groups_{n_ignored}_ignored'
    
    return {
        'allocation_type': allocation_type,
        'subtype': subtype,
        'feature_assignments': feature_assignments,
        'superposition_groups': superposition_groups,
        'ignored_features': ignored_features,
        'dedicated_features': dedicated_features,
        'unassigned_features': [i for i in range(n_features) if feature_assignments[i] == 'unknown'],
        'n_dedicated': n_dedicated,
        'n_ignored': n_ignored,
        'n_superimposed': n_superimposed,
        'n_unassigned': n_unassigned,
        'total_features': n_features
    }


def classify_all_experts_feature_allocation(expert_weights, tolerance=0.1):
    """
    Classify feature allocation for all experts in a model.
    
    Args:
        expert_weights: Shape [n_experts, n_features, n_hidden]
        tolerance: Tolerance for classification
    
    Returns:
        Dictionary with classifications for each expert
    """
    n_experts = expert_weights.shape[0]
    classifications = {}
    
    for expert_id in range(n_experts):
        expert_analysis = compute_dimensions_per_feature_single(expert_weights[expert_id])
        expert_classification = classify_expert_feature_allocation(expert_analysis, tolerance)
        classifications[f'expert_{expert_id}'] = expert_classification
    
    return classifications


def print_expert_allocation_summary(classifications):
    """
    Print a summary of expert feature allocations.
    
    Args:
        classifications: Output from classify_all_experts_feature_allocation
    """
    print("=== Expert Feature Allocation Summary ===")
    
    allocation_counts = {}
    for expert_key, classification in classifications.items():
        allocation_type = classification['allocation_type']
        if allocation_type not in allocation_counts:
            allocation_counts[allocation_type] = 0
        allocation_counts[allocation_type] += 1
    
    print(f"Total experts: {len(classifications)}")
    print("\nAllocation type distribution:")
    for allocation_type, count in allocation_counts.items():
        percentage = (count / len(classifications)) * 100
        print(f"  {allocation_type}: {count} experts ({percentage:.1f}%)")
    
    print("\nDetailed breakdown:")
    for expert_key, classification in classifications.items():
        expert_num = expert_key.split('_')[1]
        print(f"\nExpert {expert_num}:")
        print(f"  Type: {classification['allocation_type']}")
        print(f"  Subtype: {classification['subtype']}")
        print(f"  Dedicated features: {classification['dedicated_features']}")
        print(f"  Ignored features: {classification['ignored_features']}")
        print(f"  Superposition groups: {classification['superposition_groups']}")
        print(f"  Unassigned features: {classification['unassigned_features']}")


def get_allocation_type_for_plotting(classifications):
    """
    Get allocation types in a format suitable for plotting.
    
    Args:
        classifications: Output from classify_all_experts_feature_allocation
    
    Returns:
        List of allocation types for each expert
    """
    allocation_types = []
    for expert_key in sorted(classifications.keys()):
        allocation_types.append(classifications[expert_key]['allocation_type'])
    return allocation_types 
    
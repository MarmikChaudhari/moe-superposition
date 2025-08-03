# Examples in the dimensions_per_feature.ipynb notebook

import torch
import numpy as np
from matplotlib import pyplot as plt
from helpers.dimensions_per_feature import compute_dimensions_per_feature_single

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
    plt.colorbar(label='Expert 0 Selection Probability')
    plt.xlabel(f'Feature 1 (max={max_val:.2f})')
    plt.ylabel(f'Feature {n_features} (max={max_val:.2f})')
    plt.title(f'Expert 0 Selection Probability: Feature 1 vs Feature {n_features}')
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
    
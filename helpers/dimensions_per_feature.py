# Examples in the dimensions_per_feature.ipynb notebook

import torch
import matplotlib.pyplot as plt

def compute_dimensions_per_feature(expert_weights):
    """
    Compute dimensions per feature for a given expert weight matrix.
    
    Args:
        expert_weights: Shape [n_experts, n_features, n_hidden] or [n_features, n_hidden]
    
    Returns:
        Dictionary with various dimensionality metrics
    """
    # Handle different input shapes
    if expert_weights.dim() == 3:
        # [n_experts, n_features, n_hidden] - analyze each expert separately
        n_experts, n_features, n_hidden = expert_weights.shape
        results = {}
        
        for expert_id in range(n_experts):
            expert_result = compute_dimensions_per_feature_single(expert_weights[expert_id])
            results[f'expert_{expert_id}'] = expert_result
        
        return results
    else:
        # [n_features, n_hidden] - single expert
        return compute_dimensions_per_feature_single(expert_weights)


def compute_dimensions_per_feature_single(weight_matrix):
    """
    Compute dimensions per feature for a single weight matrix.
    """
    n_features, n_hidden = weight_matrix.shape
    
    # 1. Compute Frobenius norm (total features learned)
    frobenius_norm_squared = torch.sum(weight_matrix ** 2)
    
    # 2. Compute dimensions per feature (D* = m / ||W||_F^2)
    dimensions_per_feature = n_hidden / frobenius_norm_squared
    
    # 3. Compute individual feature dimensionalities
    feature_dimensionalities = []
    feature_norms = []
    feature_unit_vectors = []
    
    for i in range(n_features):
        # Get weight vector for this feature
        w_i = weight_matrix[i, :]  # Shape: [n_hidden]
        
        # Compute norm of this feature
        feature_norm = torch.norm(w_i)
        feature_norms.append(feature_norm)
        
        # Compute unit vector
        if feature_norm > 1e-8:  # Avoid division by zero
            w_i_unit = w_i / feature_norm
        else:
            w_i_unit = torch.zeros_like(w_i)
        feature_unit_vectors.append(w_i_unit)
        
        # Compute feature dimensionality D_i
        numerator = feature_norm ** 2
        
        # Denominator: sum of squared dot products with all features
        denominator = 0
        for j in range(n_features):
            w_j = weight_matrix[j, :]
            dot_product = torch.dot(w_i_unit, w_j)
            denominator += dot_product ** 2
        
        if denominator > 1e-8:
            feature_dimensionality = numerator / denominator
        else:
            feature_dimensionality = torch.tensor(0.0)
        
        feature_dimensionalities.append(feature_dimensionality)
    
    # 4. Compute feature geometry (dot products between features)
    feature_geometry = torch.zeros(n_features, n_features)
    for i in range(n_features):
        for j in range(n_features):
            w_i = weight_matrix[i, :]
            w_j = weight_matrix[j, :]
            feature_geometry[i, j] = torch.abs(torch.dot(w_i, w_j))
    
    # 5. Check for antipodal pairs
    antipodal_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            w_i = weight_matrix[i, :]
            w_j = weight_matrix[j, :]
            
            # Check if features are antipodal (negative of each other)
            norm_i = torch.norm(w_i)
            norm_j = torch.norm(w_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                cosine_similarity = torch.dot(w_i, w_j) / (norm_i * norm_j)
                if torch.abs(cosine_similarity + 1.0) < 0.1:  # Nearly antipodal
                    antipodal_pairs.append((i, j))
    
    return {
        'frobenius_norm_squared': frobenius_norm_squared,
        'dimensions_per_feature': dimensions_per_feature,
        'feature_dimensionalities': feature_dimensionalities,
        'feature_norms': feature_norms,
        'feature_geometry': feature_geometry,
        'antipodal_pairs': antipodal_pairs,
        'n_features': n_features,
        'n_hidden': n_hidden
    }

def plot_feature_dimensionality_analysis(analysis, title="Feature Dimensionality Analysis"):
    """
    Plot the feature dimensionality analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert tensors to numpy for plotting
    feature_dims = [d.item() if torch.is_tensor(d) else d for d in analysis['feature_dimensionalities']]
    feature_norms = [n.item() if torch.is_tensor(n) else n for n in analysis['feature_norms']]
    geometry = analysis['feature_geometry'].cpu().detach().numpy() if torch.is_tensor(analysis['feature_geometry']) else analysis['feature_geometry']
    frobenius_norm = analysis['frobenius_norm_squared'].item() if torch.is_tensor(analysis['frobenius_norm_squared']) else analysis['frobenius_norm_squared']
    dimensions_per_feature = analysis['dimensions_per_feature'].item() if torch.is_tensor(analysis['dimensions_per_feature']) else analysis['dimensions_per_feature']
    
    # Plot 1: Feature dimensionalities
    ax1 = axes[0, 0]
    ax1.bar(range(len(feature_dims)), feature_dims)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Dimensionality')
    ax1.set_title('Individual Feature Dimensionalities')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Antipodal (0.5)')
    ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Orthogonal (1.0)')
    ax1.legend()
    
    # Plot 2: Feature norms
    ax2 = axes[0, 1]
    ax2.bar(range(len(feature_norms)), feature_norms)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Feature Norm')
    ax2.set_title('Feature Norms (||W_i||)')
    
    # Plot 3: Feature geometry heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(geometry, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Feature Index')
    ax3.set_title('Feature Geometry (|W_i · W_j|)')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.8, f"Dimensions per feature: {dimensions_per_feature:.3f}", fontsize=12)
    ax4.text(0.1, 0.7, f"Frobenius norm²: {frobenius_norm:.3f}", fontsize=12)
    ax4.text(0.1, 0.6, f"Antipodal pairs: {len(analysis['antipodal_pairs'])}", fontsize=12)
    ax4.text(0.1, 0.5, f"Features: {analysis['n_features']}", fontsize=12)
    ax4.text(0.1, 0.4, f"Hidden dims: {analysis['n_hidden']}", fontsize=12)
    
    # List antipodal pairs
    if analysis['antipodal_pairs']:
        pairs_text = "Antipodal pairs:\n"
        for i, j in analysis['antipodal_pairs']:
            pairs_text += f"  ({i}, {j})\n"
        ax4.text(0.1, 0.3, pairs_text, fontsize=10)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
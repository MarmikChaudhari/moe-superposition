import torch
import numpy as np
import torch.nn.functional as F
from model.model import MoEModel, Config

# Set device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def create_model(n_features=32, n_hidden=8, n_experts=4, n_active_experts=1, one_minus_sparsity=0.1, importance_vector=None):
    """Train model with router selection tracking."""
    config = Config(
        n_features=n_features,
        n_hidden=n_hidden,
        n_experts=n_experts,
        n_active_experts=n_active_experts,
        load_balancing_loss=True,
    )

    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    model = MoEModel(
        config=config,
        device=DEVICE,
        importance = torch.tensor(importance_vector if importance_vector is not None else [1 for _ in range(config.n_features)]),
        feature_probability=torch.tensor(one_minus_sparsity)
    )

    return model, config


def classify_expert_weights_2d(expert_weights: torch.Tensor, tolerance: float = 0.1):
    """Classify expert weight matrices for 2-feature models."""
    n_experts, n_features, n_hidden = expert_weights.shape
    classifications = {}
    
    for expert_id in range(n_experts):
        weights = expert_weights[expert_id]  # Shape: [n_features, n_hidden]
        
        # For each hidden dimension, classify the feature weights
        hidden_classifications = []
        for hidden_dim in range(n_hidden):
            # Handle the case where n_hidden=1 (tensor gets squeezed)
            if n_hidden == 1:
                feature_weights = weights.squeeze()  # Shape: [n_features]
            else:
                feature_weights = weights[:, hidden_dim]  # Shape: [n_features]
            
            weights_norm = feature_weights / torch.norm(feature_weights)
            weights_np = weights_norm.cpu().detach().numpy()
            
            # Check for superposition patterns (2 features)
            superposition_patterns = [
                np.array([1.0, 1.0]),   # Both features positive
                np.array([1.0, -1.0]),  # Features opposite
                np.array([-1.0, 1.0]),  # Features opposite (reversed)
                np.array([-1.0, -1.0])  # Both features negative
            ]
            
            # Single features (orthogonal)
            single_patterns = [
                np.array([1.0, 0.0]),  # Feature 0 only
                np.array([0.0, 1.0])   # Feature 1 only
            ]
            
            # Test all patterns
            max_similarity = 0
            best_pattern = None
            best_pattern_type = None
            
            # Test superposition patterns
            for pattern in superposition_patterns:
                pattern_norm = pattern / np.linalg.norm(pattern)
                similarity = np.abs(np.dot(weights_np, pattern_norm))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
                    best_pattern_type = 'superposition'
            
            # Test single patterns
            for pattern in single_patterns:
                pattern_norm = pattern / np.linalg.norm(pattern)
                similarity = np.abs(np.dot(weights_np, pattern_norm))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
                    best_pattern_type = 'single'
            
            # Determine classification
            if max_similarity > (1.0 - tolerance):
                if best_pattern_type == 'superposition':
                    # Identify which features are superimposed
                    active_features = np.where(np.abs(best_pattern) > 0.1)[0]
                    signs = best_pattern[active_features]
                    if np.all(signs > 0):
                        classification = f'superposition_both_positive'
                    elif np.all(signs < 0):
                        classification = f'superposition_both_negative'
                    else:
                        classification = f'superposition_opposite'
                elif best_pattern_type == 'single':
                    # Identify which single feature
                    active_feature = np.where(np.abs(best_pattern) > 0.1)[0][0]
                    classification = f'orthogonal_feature_{active_feature}'
            else:
                classification = 'incomprehensible'
            
            hidden_classifications.append({
                'classification': classification,
                'similarity': max_similarity,
                'weights': feature_weights.tolist(),
                'pattern': best_pattern.tolist() if best_pattern is not None else None
            })
        
        classifications[f'expert_{expert_id}'] = {
            'hidden_dimensions': hidden_classifications,
            'raw_weights': weights.tolist()
        }
    
    return classifications


# Classify expert weights into different types (superposition, orthogonal, incomprehensible)
# This is a simple classification based on the weights of the expert.

# Classify expert weights
def classify_expert_weights_n3(expert_weights: torch.Tensor, tolerance: float = 0.1):
    """Classify expert weight matrices into different types."""
    n_experts, n_features, n_hidden = expert_weights.shape
    classifications = {}
    
    for expert_id in range(n_experts):
        weights = expert_weights[expert_id].squeeze()  # Shape: [n_features, n_hidden]
        
        # For each hidden dimension, classify the feature weights
        hidden_classifications = []
        for hidden_dim in range(n_hidden):
            feature_weights = weights[:, hidden_dim]  # Shape: [n_features]
            weights_norm = feature_weights / torch.norm(feature_weights)
            weights_np = weights_norm.cpu().detach().numpy()
            
            # Check for superposition patterns
            # All three features superimposed
            all_three_patterns = [
                np.array([1.0, 1.0, 1.0]),
                np.array([1.0, 1.0, -1.0]),
                np.array([1.0, -1.0, 1.0]),
                np.array([-1.0, 1.0, 1.0]),
                np.array([1.0, -1.0, -1.0]),
                np.array([-1.0, 1.0, -1.0]),
                np.array([-1.0, -1.0, 1.0]),
                np.array([-1.0, -1.0, -1.0])
            ]
            
            # Pairs of features superimposed
            pair_patterns = [
                np.array([1.0, 1.0, 0.0]),  # Features 0,1 superimposed
                np.array([1.0, 0.0, 1.0]),  # Features 0,2 superimposed
                np.array([0.0, 1.0, 1.0]),  # Features 1,2 superimposed
                np.array([1.0, -1.0, 0.0]), # Features 0,1 superimposed (opposite)
                np.array([1.0, 0.0, -1.0]), # Features 0,2 superimposed (opposite)
                np.array([0.0, 1.0, -1.0]), # Features 1,2 superimposed (opposite)
                np.array([-1.0, 1.0, 0.0]), # Features 0,1 superimposed (opposite)
                np.array([-1.0, 0.0, 1.0]), # Features 0,2 superimposed (opposite)
                np.array([0.0, -1.0, 1.0])  # Features 1,2 superimposed (opposite)
            ]
            
            # Single features (orthogonal)
            single_patterns = [
                np.array([1.0, 0.0, 0.0]),  # Feature 0 only
                np.array([0.0, 1.0, 0.0]),  # Feature 1 only
                np.array([0.0, 0.0, 1.0])   # Feature 2 only
            ]
            
            # Test all patterns
            max_similarity = 0
            best_pattern = None
            best_pattern_type = None
            
            # Test all-three patterns
            for pattern in all_three_patterns:
                pattern_norm = pattern / np.linalg.norm(pattern)
                similarity = np.abs(np.dot(weights_np, pattern_norm))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
                    best_pattern_type = 'all_three'
            
            # Test pair patterns
            for pattern in pair_patterns:
                pattern_norm = pattern / np.linalg.norm(pattern)
                similarity = np.abs(np.dot(weights_np, pattern_norm))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
                    best_pattern_type = 'pair'
            
            # Test single patterns
            for pattern in single_patterns:
                pattern_norm = pattern / np.linalg.norm(pattern)
                similarity = np.abs(np.dot(weights_np, pattern_norm))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern
                    best_pattern_type = 'single'
            
            # Determine classification
            if max_similarity > (1.0 - tolerance):
                if best_pattern_type == 'all_three':
                    # Identify which features are superimposed
                    active_features = np.where(np.abs(best_pattern) > 0.1)[0]
                    signs = best_pattern[active_features]
                    if np.all(signs > 0):
                        classification = f'superposition_all_three_positive'
                    elif np.all(signs < 0):
                        classification = f'superposition_all_three_negative'
                    else:
                        classification = f'superposition_all_three_mixed'
                elif best_pattern_type == 'pair':
                    # Identify which pair is superimposed
                    active_features = np.where(np.abs(best_pattern) > 0.1)[0]
                    signs = best_pattern[active_features]
                    feature_names = [f'feature_{i}' for i in active_features]
                    if np.all(signs > 0):
                        classification = f'superposition_pair_{"_".join(feature_names)}_positive'
                    elif np.all(signs < 0):
                        classification = f'superposition_pair_{"_".join(feature_names)}_negative'
                    else:
                        classification = f'superposition_pair_{"_".join(feature_names)}_mixed'
                elif best_pattern_type == 'single':
                    # Identify which single feature
                    active_feature = np.where(np.abs(best_pattern) > 0.1)[0][0]
                    classification = f'orthogonal_feature_{active_feature}'
            else:
                classification = 'incomprehensible'
            
            hidden_classifications.append({
                'classification': classification,
                'similarity': max_similarity,
                'weights': feature_weights.tolist(),
                'pattern': best_pattern.tolist() if best_pattern is not None else None
            })
        
        classifications[f'expert_{expert_id}'] = {
            'hidden_dimensions': hidden_classifications,
            'raw_weights': weights.tolist()
        }
    
    return classifications


def compute_router_probabilities(gate_matrix, feature_probability=None, n_samples=10000):
    """
    Compute probability that router picks first expert.
    
    Args:
        gate_matrix: Shape [n_experts, n_features] - the gate weights
        feature_probability: Probability of feature being active (None for uniform)
        n_samples: Number of samples to estimate probability
    
    Returns:
        prob_with_data: Probability considering data sparsity
        prob_without_data: Probability assuming uniform data
    """
    device = gate_matrix.device
    n_experts, n_features = gate_matrix.shape
    
    # 1. Probability WITHOUT considering data properties (uniform data)
    # Generate uniform random features
    uniform_features = torch.rand(n_samples, n_features, device=device)
    
    # Compute gate scores and probabilities
    gate_scores_uniform = torch.einsum("bf,ef->be", uniform_features, gate_matrix)
    gate_probs_uniform = F.softmax(gate_scores_uniform, dim=-1)
    
    # Probability of selecting first expert
    prob_without_data = gate_probs_uniform[:, 0].mean().item()
    
    # 2. Probability WITH considering data properties (sparse data)
    if feature_probability is not None:
        # Generate sparse data according to feature_probability
        sparse_features = torch.where(
            torch.rand(n_samples, n_features, device=device) <= feature_probability,
            torch.rand(n_samples, n_features, device=device),
            torch.zeros(n_samples, n_features, device=device)
        )
        
        # Compute gate scores and probabilities for sparse data
        gate_scores_sparse = torch.einsum("bf,ef->be", sparse_features, gate_matrix)
        gate_probs_sparse = F.softmax(gate_scores_sparse, dim=-1)
        
        # Probability of selecting first expert
        prob_with_data = gate_probs_sparse[:, 0].mean().item()
    else:
        prob_with_data = prob_without_data
    
    return prob_with_data, prob_without_data

# For multiple experts, you can also compute for each expert:
def compute_all_expert_probabilities(gate_matrix, feature_probability=None, n_samples=10000):
    """
    Compute probability for each expert being selected.
    """
    device = gate_matrix.device
    n_experts, n_features = gate_matrix.shape
    
    # Uniform data
    uniform_features = torch.rand(n_samples, n_features, device=device)
    gate_scores_uniform = torch.einsum("bf,ef->be", uniform_features, gate_matrix)
    gate_probs_uniform = F.softmax(gate_scores_uniform, dim=-1)
    probs_without_data = gate_probs_uniform.mean(dim=0).cpu().detach().numpy()
    
    # Sparse data
    if feature_probability is not None:
        sparse_features = torch.where(
            torch.rand(n_samples, n_features, device=device) <= feature_probability,
            torch.rand(n_samples, n_features, device=device),
            torch.zeros(n_samples, n_features, device=device)
        )
        gate_scores_sparse = torch.einsum("bf,ef->be", sparse_features, gate_matrix)
        gate_probs_sparse = F.softmax(gate_scores_sparse, dim=-1)
        probs_with_data = gate_probs_sparse.mean(dim=0).cpu().detach().numpy()
    else:
        probs_with_data = probs_without_data
    
    return probs_with_data, probs_without_data


# Tools to track gate expert selection statistics (during training)
# Currently only works for k=1

import torch
import numpy as np

class RouterSelectionHook:
    def __init__(self, debug=False):
        self.expert_selections = {}  # {expert_id: count}
        self.total_selections = 0
        self.batch_size = 0
        self.call_count = 0  # Track how many times hook is called
        self.debug = False
        
    def __call__(self, hook_data):
        """Hook function called during training."""
        model = hook_data['model']
        batch = hook_data.get('batch', None)
        
        if batch is None:
            # Generate a batch to analyze router behavior
            batch = model.generate_batch(1000)
        
        # Get router probabilities for this batch
        with torch.no_grad():
            expert_weights, top_k_indices, _ = model.compute_active_experts(batch)
            
            # For k=1, track which expert was selected for each input
            if model.config.n_active_experts == 1:
                selected_experts = top_k_indices.squeeze(-1)  # [batch_size]
                
                # Validate expert indices
                n_experts = model.config.n_experts
                valid_experts = selected_experts[(selected_experts >= 0) & (selected_experts < n_experts)]
                
                # Count selections for each expert
                for expert_id in valid_experts.cpu().numpy():
                    if expert_id not in self.expert_selections:
                        self.expert_selections[expert_id] = 0
                    self.expert_selections[expert_id] += 1
                
                self.total_selections += len(valid_experts)
                self.batch_size = len(selected_experts)
                self.call_count += 1
                
                # Debug info (optional)
                # if self.debug and self.call_count % 10 == 0:  # Print every 10 calls
                #     print(f"Hook call {self.call_count}: {len(valid_experts)} valid selections out of {len(selected_experts)} total")
    
    def reset(self):
        """Reset all counters."""
        self.expert_selections = {}
        self.total_selections = 0
        self.batch_size = 0
        self.call_count = 0
    
    def get_statistics(self):
        """Get router selection statistics."""
        if self.total_selections == 0:
            return {}
        
        stats = {
            'total_selections': self.total_selections,
            'expert_counts': dict(self.expert_selections),
            'expert_percentages': {},
            'call_count': self.call_count
        }
        
        for expert_id, count in self.expert_selections.items():
            stats['expert_percentages'][expert_id] = (count / self.total_selections) * 100
        
        return stats
    
    def print_statistics(self):
        """Print current router selection statistics."""
        stats = self.get_statistics()
        
        if not stats:
            print("No router selections recorded yet.")
            return
        
        print(f"\n=== Router Selection Statistics ===")
        print(f"Total selections: {stats['total_selections']}")
        print(f"Hook calls: {stats['call_count']}")
        print(f"Average selections per call: {stats['total_selections'] / stats['call_count']:.1f}")
        print(f"Expert counts: {stats['expert_counts']}")
        print(f"Expert percentages:")
        for expert_id, percentage in stats['expert_percentages'].items():
            print(f"  Expert {expert_id}: {percentage:.2f}%")
        
        # Check for load balancing
        n_experts = len(stats['expert_counts'])
        if n_experts > 0:
            expected_percentage = 100.0 / n_experts
            print(f"Expected percentage per expert: {expected_percentage:.2f}%")
            
            # Calculate load balancing metric
            percentages = list(stats['expert_percentages'].values())
            variance = np.var(percentages)
            print(f"Load balancing variance: {variance:.2f}")

# Alternative: Hook that tracks router behavior over time
class RouterEvolutionHook:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step = 0
        self.router_history = []  # List of (step, expert_percentages) tuples
        
    def __call__(self, hook_data):
        self.step += 1
        
        if self.step % self.log_interval == 0:
            # Create temporary hook to get current statistics
            temp_hook = RouterSelectionHook()
            temp_hook(hook_data)
            
            stats = temp_hook.get_statistics()
            if stats:
                self.router_history.append((self.step, stats['expert_percentages']))
                
                print(f"\nStep {self.step} - Router evolution:")
                for expert_id, percentage in stats['expert_percentages'].items():
                    print(f"  Expert {expert_id}: {percentage:.2f}%")
    
    def plot_evolution(self):
        """Plot router evolution over time."""
        if not self.router_history:
            print("No router history to plot.")
            return
        
        import matplotlib.pyplot as plt
        
        steps = [step for step, _ in self.router_history]
        expert_ids = set()
        for _, percentages in self.router_history:
            expert_ids.update(percentages.keys())
        
        plt.figure(figsize=(10, 6))
        for expert_id in sorted(expert_ids):
            percentages = [data.get(expert_id, 0) for _, data in self.router_history]
            plt.plot(steps, percentages, label=f'Expert {expert_id}', marker='o')
        
        plt.xlabel('Training Step')
        plt.ylabel('Router Selection Percentage')
        plt.title('Router Evolution During Training')
        plt.legend()
        plt.grid(True)
        plt.show()

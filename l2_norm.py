import torch


def compute_feature_difference(query_features, normal_features):
    """
    Compute the feature difference between query and normal features.
    Args:
        query_features (torch.Tensor): Query feature tensor of shape (B, N, D).
        normal_features (torch.Tensor): Normal feature tensor of shape (B, N, D).
    Returns:
        difference_map (torch.Tensor): Feature difference map of shape (B, N).
    """
    # Compute L2 Norm of the feature difference
    difference = torch.norm(query_features - normal_features, dim=-1)
    return difference


def compute_anomaly_map(query_features, normal_features, similarity_map):
    """
    Combine similarity map and feature difference to compute anomaly map.
    Args:
        query_features (torch.Tensor): Query feature tensor of shape (B, N, D).
        normal_features (torch.Tensor): Normal feature tensor of shape (B, N, D).
        similarity_map (torch.Tensor): Similarity map of shape (B, 1, H, W).
    Returns:
        anomaly_map (torch.Tensor): Combined anomaly map of shape (B, 1, H, W).
    """
    # Feature Difference
    feature_diff = compute_feature_difference(query_features, normal_features)  # Shape: (B, N)

    # Reshape and normalize feature_diff to match similarity_map size
    feature_diff = feature_diff.view(similarity_map.shape)  # Reshape to (B, 1, H, W)
    feature_diff = feature_diff / (feature_diff.max() + 1e-8)  # Normalize to [0, 1]

    # Combine similarity and feature difference
    combined_map = (1 - similarity_map + feature_diff) / 2  # Weighted Average
    return combined_map
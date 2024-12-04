import torch


def contrast_normalization(similarity_map):
    """
    Contrast normalization to emphasize the differences in similarity scores.
    Args:
        similarity_map (torch.Tensor): Similarity map of shape (B, 1, H, W).
    Returns:
        normalized_map (torch.Tensor): Normalized similarity map of the same shape.
    """
    # Normalize to [0, 1]
    min_val = similarity_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = similarity_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    normalized_map = (similarity_map - min_val) / (max_val - min_val + 1e-8)

    # Return anomaly map (1 - normalized similarity)
    anomaly_map = 1 - normalized_map
    return anomaly_map
import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, embed_dim):
        """
        Cross-attention mechanism between query and normal features.
        Args:
            embed_dim (int): Dimensionality of the feature embeddings.
        """
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_features, normal_features):
        """
        Compute cross-attention between query and normal features.
        Args:
            query_features (torch.Tensor): Query features (B, N, D).
            normal_features (torch.Tensor): Normal features (B, N, D).
        Returns:
            attended_features (torch.Tensor): Attention-weighted features (B, N, D).
        """
        Q = self.query_proj(query_features)  # (B, N, D)
        K = self.key_proj(normal_features)  # (B, N, D)
        V = self.value_proj(normal_features)  # (B, N, D)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B, N, N)
        attention_scores = self.softmax(attention_scores / (query_features.size(-1) ** 0.5))

        attended_features = torch.matmul(attention_scores, V)  # (B, N, D)
        return attended_features


def compute_anomaly_with_attention(query_features, normal_features, similarity_map):
    """
    Use cross-attention to refine anomaly map.
    Args:
        query_features (torch.Tensor): Query feature tensor of shape (B, N, D).
        normal_features (torch.Tensor): Normal feature tensor of shape (B, N, D).
        similarity_map (torch.Tensor): Similarity map of shape (B, 1, H, W).
    Returns:
        refined_anomaly_map (torch.Tensor): Attention-refined anomaly map.
    """
    attention_layer = CrossAttention(embed_dim=query_features.size(-1))
    attended_features = attention_layer(query_features, normal_features)

    feature_diff = torch.norm(attended_features - query_features, dim=-1)  # Shape: (B, N)
    feature_diff = feature_diff.view(similarity_map.shape)  # (B, 1, H, W)
    feature_diff = feature_diff / (feature_diff.max() + 1e-8)

    refined_anomaly_map = (1 - similarity_map + feature_diff) / 2
    return refined_anomaly_map
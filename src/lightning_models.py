from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdbscan import HDBSCAN
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity

import wandb

from .models.swin_transformer import SwinTransformer
from .utils.loss import BinaryDiceLoss, FocalLoss, InfoNCELoss
from .utils.metric import MetricsComputer


class FOCALightning(L.LightningModule):
    """
    This implementation is based on the official FOCAL framework: 
    "Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering."

    - Official Repository: https://github.com/HighwayWu/FOCAL
    - Paper: https://arxiv.org/pdf/2308.09307

    This adaptation reimplements FOCAL for industrial anomaly detection, where the goal is to identify 
    defective or anomalous regions in industrial products. It utilizes a SwinTransformer for feature 
    extraction, contrastive learning via InfoNCE loss, and unsupervised clustering for anomaly localization.

    Key Features:
    - Encoder: Swin Transformer is used for high-quality feature extraction.
    - Loss Function: InfoNCE loss is applied to separate normal and anomalous regions in feature space.
    - Clustering: K-means (default) is used for anomaly map generation, with support for other clustering 
      algorithms like HDBSCAN.
    - Validation Metrics: F1-score, IoU, and other metrics are computed to evaluate anomaly localization 
      performance.
    - Logging: Weights & Biases (wandb) integration for visualizing input images, anomaly maps, and ground 
      truth masks.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the FOCALightning module with configuration.

        Args:
            cfg (Any): Configuration object containing experiment and model settings.
        """
        super(FOCALightning, self).__init__()
        self.cfg = cfg
        
        self.encoder = SwinTransformer(
            model_name=cfg.model.model_name,
            pretrained=True,
            features_only=True
        )
        self.infoNCE = InfoNCELoss()
        self.metrics = MetricsComputer()
        self.clustering = self._init_clustering(cfg.experiment)

        self.layers_to_extract = cfg.experiment.layers_to_extract
        self.max_samples = cfg.experiment.max_samples
        self.lr = cfg.experiment.learning_rate
        self.max_epochs = cfg.experiment.max_epochs
        self.optimizer = cfg.experiment.optimizer
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, Any]): Batch of data containing images and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed training loss.
        """
        
        inputs = torch.stack(batch['images'][1::2], dim=0) # [abnormal_1, abnormal_2, ...]
        masks = torch.stack(batch['masks'][1::2], dim=0) # [mask_abnormal_1, mask_abnormal_2, ...]

        features = self.encoder(inputs) # L x (B, W, H, C)
        features_merged = self._merge_feature_maps(features, self.layers_to_extract) # (B, W, H, C)
        features_merged = F.normalize(features_merged, dim=3)

        loss = self._compute_loss(features_merged, masks)
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Perform a single validation step.

        Create anomaly map using Clustering,
        K-means now, but to be updated as using HDBSCAN or anything.

        Also, unlike the official implementation, this code does not utilize model ensembling.
        (to be updated)

        Args:
            batch (Dict[str, Any]): Batch of data containing images and masks.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, Any]: Computed metrics for the batch.
        """
        inputs = torch.stack(batch['images'], dim=0)
        masks = torch.stack(batch['masks'], dim=0)
        
        features = self.encoder(inputs) # L x (B, H, W, C)
        features_merged = self._merge_feature_maps(features, self.layers_to_extract) # (B, H, W, C)
        features_merged = F.normalize(features_merged, dim=3)
        features_flattened = torch.flatten(features_merged, start_dim=1, end_dim=2)
        
        masks = F.interpolate(masks, size=(features_merged[0, :, :, 0].shape), mode='nearest')
        
        # assume using the K-means clustering now
        results = []
        clusters = self.clustering(features_flattened, k=2).labels
        for cluster in clusters:
            cluster = cluster if torch.sum(cluster) <= torch.sum(1 - cluster) else 1 - cluster
            cluster = cluster.view(features_merged.shape[1:3])[None, :, :, None] # (1, H, W, 1)
            results.append(cluster)

        anomaly_map = torch.cat(results, dim=0).permute(0, 3, 1, 2) # (B, 1, H, W)

        masks_flat = masks.view(masks.size(0), -1).cpu().numpy()
        anomaly_map_flat = anomaly_map.view(anomaly_map.size(0), -1).cpu().numpy()

        i_pred = anomaly_map_flat.max(axis=1)
        i_label = masks_flat.max(axis=1)
        
        metrics = self.metrics.compute_all_metrics(masks_flat, anomaly_map_flat, i_pred, i_label)

        num_samples_to_log = min(4, anomaly_map.size(0))  # Log up to 4 samples
        inputs_resized = F.interpolate(inputs, size=anomaly_map.shape[-2:], mode="bilinear", align_corners=False)  # (B, C, H, W)
        for idx in range(num_samples_to_log):
            wandb.log({
                f"val_sample {batch_idx}_{idx}": [
                    wandb.Image(inputs_resized[idx].cpu().numpy().transpose(1, 2, 0), caption="Input Image"),
                    wandb.Image(anomaly_map[idx][0].cpu().numpy(), caption="Anomaly Map"),
                    wandb.Image(masks[idx][0].cpu().numpy(), caption="Ground Truth"),
                ]
            })

        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value)

        return metrics

    def _init_clustering(self, cfg: DictConfig) -> Union[KMeans, HDBSCAN]:
        """
        Initialize the clustering method based on experiment settings.

        Args:
            cfg (Any): Configuration object containing clustering settings.

        Returns:
            Union[KMeans, HDBSCAN]: Clustering algorithm instance.
        """
        if cfg.clustering == 'k_means':
            return KMeans(n_clusters=2, verbose=False)
        elif cfg.clustering == 'HDBSCAN':
            return HDBSCAN(min_cluster_size=10, metric='cosine')
        else:
            raise ValueError(f"Unsupported clustering method: {cfg.clustering}")
    
    def _compute_loss(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the InfoNCE loss for training.

        Args:
            features (torch.Tensor): Feature maps from the encoder.
            masks (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss = []

        masks = F.interpolate(masks, size=(features[0, :, :, 0].shape), mode='nearest').squeeze(1)
        for feature, mask in zip(features, masks):
            query = feature[mask == 0]
            negative = feature[mask == 1]
            
            # skip normal samples
            if query.size(0) == 0 or negative.size(0) == 0:
                continue
            
            query_sample = query[torch.randperm(query.shape[0])][:self.max_samples]
            negative_sample = negative[torch.randperm(negative.shape[0])][:self.max_samples]
            loss.append(self.infoNCE(query_sample, query_sample, negative_sample))

        return torch.mean(torch.stack(loss).squeeze())
        
    def _merge_feature_maps(
        self,
        features: List[torch.Tensor],
        layers: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Merge feature maps from specified layers of encoder.

        Args:
            features (List[torch.Tensor]): List of feature maps.
            layers (Optional[str]): 
                Comma-separated string of layer indices to use, e.g., "0,1,2,3".
                If None, all feature maps from the layers are used.
            size (Optional[Tuple[int, int]]):
                Desired spatial size (height, width) for the output feature maps.
                If None, the size of the first selected feature map is used as the base size.

        Returns:
            torch.Tensor: Combined feature map with uniform spatial size.

        Raises:
            ValueError: If no valid layer indices are provided, or if indices are out of range.
        """
        if layers is None:
            layers = "0,1,2,3"

        layer_indices = [int(idx) for idx in layers.split(",") if idx.strip().isdigit()]
        
        if not layer_indices:
            raise ValueError("No valid layer indices provided in 'layers'.")

        for idx in layer_indices:
            if idx < 0 or idx > len(features) - 1:
                raise ValueError(f"Layer index {idx} is out of range. Available range: 0 to {len(features) - 1}")

        base_size = features[layer_indices[0]].shape[1:3] if size is None else size # (H, W)

        combined = []
        for idx in layer_indices:
            feature = features[idx]
            if feature.shape[1:3] != base_size:
                feature = F.interpolate(
                    feature.permute(0, 3, 1, 2), size=base_size, mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)  # Back to (B, H, W, C)
            combined.append(feature)

        return torch.cat(combined, dim=-1)
    
    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]:
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Raises:
            ValueError: If an unsupported optimizer type is provided in the configuration.
        Returns:
            Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]: 
            Optimizer and scheduler configuration.
        """
        if self.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "learning_rate"
            }
        }
    

class FewShotLightning(L.LightningModule):
    """
    FewShotLightning module
    ------------------------
    This module is designed for few-shot anomaly detection tasks.

    Current Status:
    - Only the `predict_step` method is implemented at this stage.
    - Training and validation methods are yet to be developed and will be updated in future iterations.

    Known Issues:
    - GPU memory overflow issue during distributed training has been identified.
    - The issue is under investigation and a fix will be applied in a future update.

    Purpose:
    - This module aims to provide a flexible framework for implementing few-shot anomaly detection algorithms.
    """
    def __init__(self, args):
        super(FewShotLightning, self).__init__()
        self.args = args
        self.encoder = SwinTransformer(
            model_name=args.model_name,
            pretrained=True,
            features_only=True
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return features

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            normal_inputs = torch.stack(batch['images'][0::2], dim=0)
            query_inputs = torch.stack(batch['images'][1::2], dim=0)

            query_patches = self.encoder(query_inputs) # L x [B, H, W, D]
            normal_patches = self.encoder(normal_inputs) # L x [B, H, W, D]

            similarity_map = self.get_similarity_map(query_patches, normal_patches)
                
            del query_patches, normal_patches
            # empty_cache()
            import gc
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            return similarity_map

    def get_similarity_map(self, query_patches, normal_patches):
        sims = []

        for i in range(len(query_patches)):
            B, H, W, C = query_patches[i].shape
            query_patches_tokens = query_patches[i].view(B, H*W, 1, C)
            normal_patches_tokens = normal_patches[i].reshape(B, 1, -1, C)
            cosine_similarity_matrix = F.cosine_similarity(query_patches_tokens, normal_patches_tokens, dim=-1)
            sim_max, _ = torch.max(cosine_similarity_matrix, dim=-1)
            sims.append(sim_max)
        
        max_resolution = sims[0].shape[-1]
        resized_sims = [
            F.interpolate(sim.view(B, 1, int(sim.shape[1]**0.5), int(sim.shape[1]**0.5)), 
                        size=(int(max_resolution**0.5), int(max_resolution**0.5)), 
                        mode="bilinear", 
                        align_corners=False).view(sim.shape[0], -1)
            for sim in sims
        ]

        sim = torch.mean(torch.stack(resized_sims, dim=0), dim=0).reshape(B, 1, 56, 56)
        sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
        similarity_map = 1 - sim
        
        return similarity_map.cpu()
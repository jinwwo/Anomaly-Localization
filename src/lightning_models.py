from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hdbscan import HDBSCAN
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity

from models.swin_transformer import SwinTransformer
from utils.loss import BinaryDiceLoss, FocalLoss, InfoNCELoss
from utils.metric import MetricsComputer

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class FewShotLightning(L.LightningModule):
    def __init__(self, args):
        super(FewShotLightning, self).__init__()
        self.args = args
        self.encoder = SwinTransformer(
            model_name=args.model_name,
            pretrained=True
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
    

class FOCALightning(L.LightningModule):
    def __init__(self, args):
        super(FOCALightning, self).__init__()
        # model
        self.args = args
        self.encoder = SwinTransformer(
            model_name=args.model_name,
            pretrained=True,
            features_only=True
        )
        self.max_samples = args.max_samples
        self.infoNCE = InfoNCELoss()
        self.metrics = MetricsComputer()

        if args.clustering == 'k_means':
            # self.clustering = KMeans(n_clusters=2, distance=CosineSimilarity, verbose=False)
            self.clustering = KMeans(n_clusters=2, verbose=False)
        elif args.clustering == 'HDBSCAN':
            self.clustering = HDBSCAN(min_cluster_size=10, metric='cosine')
    
    def training_step(self, batch, batch_idx):
        # [abnormal_1, abnormal_2, ...]
        inputs = torch.stack(batch['images'][1::2], dim=0)
        
        # [mask_abnormal_1, mask_abnormal_2, ...]
        masks = torch.stack(batch['masks'][1::2], dim=0)
        features = self.encoder(inputs) # L x (B, W, H, C)
        features_merged = self._merge_feature_maps(features, self.args.layers_to_extract) # (B, W, H, C)
        features_merged = F.normalize(features_merged, dim=3)

        loss = self._compute_loss(features_merged, masks)
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Create localization result map using Clustering,
        K-means now, but to be updated as using HDBSCAN or anything.
        
        Also, unlike the official implementation, this code does not utilize model ensembling.
        (to be updated)
        """
        inputs = torch.stack(batch['images'], dim=0)
        masks = torch.stack(batch['masks'], dim=0)
        
        features = self.encoder(inputs) # L x (B, H, W, C)
        features_merged = self._merge_feature_maps(features, self.args.layers_to_extract) # (B, H, W, C)
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

        # Metric (f1, IOU) and logging (wandb)            
        anomaly_map = torch.cat(results, dim=0).permute(0, 3, 1, 2) # (B, 1, H, W)
        masks # (B, 1, H, W)

        masks_flat = masks.view(masks.size(0), -1).cpu().numpy()
        anomaly_map_flat = anomaly_map.view(anomaly_map.size(0), -1).cpu().numpy()

        i_pred = anomaly_map_flat.max(axis=1)
        i_label = masks_flat.max(axis=1)
        
        metrics = self.metrics.compute_all_metrics(masks_flat, anomaly_map_flat, i_pred, i_label)

        num_samples_to_log = min(4, anomaly_map.size(0))  # Log up to 4 samples
        for idx in range(num_samples_to_log):
            inputs_resized = F.interpolate(inputs, size=anomaly_map.shape[-2:], mode="bilinear", align_corners=False)  # (B, C, H, W)
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

    # def test_step(self, batch, batch_idx):

    def _compute_loss(self, features, masks):
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
        Merge feature maps from specified layers of SwinTransformer.

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

        # print(f"Extracting feature maps from layers: {layer_indices}")
        
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
            A dictionary containing the optimizer and learning rate scheduler configuration.
            The dictionary has the following keys:
            - "optimizer": The configured optimizer instance (e.g., Adam or AdamW).
            - "lr_scheduler": A dictionary with:
                - "scheduler": The learning rate scheduler instance (CosineAnnealingLR).
                - "interval": The frequency for scheduler updates (e.g., "epoch").
                - "name": Name of the learning rate scheduler ("learning_rate").
        """
        if self.args.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epochs, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "learning_rate"
            }
        }
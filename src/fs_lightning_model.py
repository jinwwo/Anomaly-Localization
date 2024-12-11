import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_transformer import SwinTransformer
from utils.loss import BinaryDiceLoss, FocalLoss


class FewShotLightning(L.LightningModule):

    def __init__(
        self,
        args,
    ):
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
        normal_inputs = torch.stack(batch['images'][0::2], dim=0)
        query_inputs = torch.stack(batch['images'][1::2], dim=0)

        query_patches = self.encoder(query_inputs) # L x [B, H, W, D]
        normal_patches = self.encoder(normal_inputs) # L x [B, H, W, D]

        similarity_map = self.get_similarity_map(query_patches, normal_patches)
        
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
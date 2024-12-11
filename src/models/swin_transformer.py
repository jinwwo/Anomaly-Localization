import timm
import torch.nn as nn


class SwinTransformer(nn.Module):
    def __init__(
        self, 
        model_name: str = None,
        pretrained: bool = False
    ):
        super().__init__()
        
        if model_name is None:
            model_name = 'swin_large_patch4_window7_224.ms_in22k'

        self.model = timm.create_model(
            model_name=model_name, 
            pretrained=pretrained,
            features_only=True, # for feature map extraction
        )

    def forward(self, x):
        x = self.model(x)
        return x
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Union


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array suitable for visualization.
    
    Args:
        tensor (torch.Tensor): A PyTorch tensor with shape [C, H, W].
        
    Returns:
        np.ndarray: A NumPy array with shape [H, W, C] normalized to range [0, 1].
    """
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1,2,0)) # [C, H, W] to [H, W, C]
    if not np.all(tensor == 0.0):
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


def visualize(images: List[Union[np.ndarray, torch.Tensor]]) -> None:
    """
    Visualizes a list of images using Matplotlib.
    
    Images can be provided as NumPy arrays or PyTorch tensors.
    PyTorch tensors will be converted to NumPy arrays automatically.
    
    Args:
        images (List[Union[np.ndarray, torch.Tensor]]): A list of images where each image 
            is either a NumPy array or a PyTorch tensor. Tensors should have shape [C, H, W], 
            while NumPy arrays should have shape [H, W, C] or [H, W].
    """
    num_imgs = len(images)
    
    plt.figure(figsize=(10,5))
    for i in range(num_imgs):
        if isinstance(images[i], torch.Tensor):
            images[i] = tensor_to_np(images[i])
        cmap = None
        if images[i].shape[-1] == 1:
            cmap='gray'
        plt.subplot(1, num_imgs, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis('off')
    plt.show()
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def login_wandb(key: Optional[str] = None) -> bool:
    """
    Log in to Weights & Biases (wandb) using an API key.

    Args:
        key (Optional[str]): The API key for logging in. If None, uses the existing login credentials.

    Returns:
        bool: True if login was successful, False otherwise.
    """
    import wandb

    return wandb.login(key=key)


def to_yaml(target: DictConfig) -> None:
    """
    Save a `DictConfig` configuration object to a YAML file.

    This function converts a `DictConfig` object to a YAML format string and prints it.
    It also saves the configuration to a YAML file in the `configs/experiment` directory,
    with a filename based on the `experiment_name` and `learning_rate` attributes of the `DictConfig` object.

    Args:
        target (DictConfig): The configuration object to convert and save.
    """
    print(OmegaConf.to_yaml(target))

    output_dir = os.getcwd() + '/configs/experiment'
    yaml_filename = f"{target.experiment_name}_lr_{target.learning_rate}.yaml"
    yaml_path = os.path.join(output_dir, yaml_filename)

    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(target))

    print(f"Config saved to: {yaml_path}")


def to_container(target: DictConfig, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert a `DictConfig` object to a Python dictionary.

    This function converts an OmegaConf `DictConfig` object to a standard Python dictionary.
    By default, it resolves interpolations in the configuration during the conversion process.

    Args:
        target (DictConfig): The configuration object to convert.
        resolve (bool, optional): Whether to resolve interpolations in the `DictConfig` during conversion.
                                  Defaults to `True`.

    Returns:
        Dict[str, Any]: The converted dictionary representation of the `DictConfig`.
    """
    return OmegaConf.to_container(target, resolve=resolve)


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
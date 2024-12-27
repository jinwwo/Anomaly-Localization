import os
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .self_sup_tasks import patch_ex

WIDTH_BOUNDS_PCT = {
    "bottle": ((0.03, 0.4), (0.03, 0.4)),
    "cable": ((0.05, 0.4), (0.05, 0.4)),
    "capsule": ((0.03, 0.15), (0.03, 0.4)),
    "hazelnut": ((0.03, 0.35), (0.03, 0.35)),
    "metal_nut": ((0.03, 0.4), (0.03, 0.4)),
    "pill": ((0.03, 0.2), (0.03, 0.4)),
    "screw": ((0.03, 0.12), (0.03, 0.12)),
    "toothbrush": ((0.03, 0.4), (0.03, 0.2)),
    "transistor": ((0.03, 0.4), (0.03, 0.4)),
    "zipper": ((0.03, 0.4), (0.03, 0.2)),
    "carpet": ((0.03, 0.4), (0.03, 0.4)),
    "grid": ((0.03, 0.4), (0.03, 0.4)),
    "leather": ((0.03, 0.4), (0.03, 0.4)),
    "tile": ((0.03, 0.4), (0.03, 0.4)),
    "wood": ((0.03, 0.4), (0.03, 0.4)),
}

INTENSITY_LOGISTIC_PARAMS = {
    "bottle": (1 / 12, 24),
    "cable": (1 / 12, 24),
    "capsule": (1 / 2, 4),
    "hazelnut": (1 / 12, 24),
    "metal_nut": (1 / 3, 7),
    "pill": (1 / 3, 7),
    "screw": (1, 3),
    "toothbrush": (1 / 6, 15),
    "transistor": (1 / 6, 15),
    "zipper": (1 / 6, 15),
    "carpet": (1 / 3, 7),
    "grid": (1 / 3, 7),
    "leather": (1 / 3, 7),
    "tile": (1 / 3, 7),
    "wood": (1 / 6, 15),
}

BACKGROUND = {
    "bottle": (200, 60),
    "screw": (200, 60),
    "capsule": (200, 60),
    "zipper": (200, 60),
    "hazelnut": (20, 20),
    "pill": (20, 20),
    "toothbrush": (20, 20),
    "metal_nut": (20, 20),
}

OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]


class MVtecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            root_dir (str): Root directory of the dataset.
            transform (Optional[Any]): Transformations to be applied to images. Defaults to resizing to 224x224.
            random_seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.norm_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        if transform:
            self.transform = transform

        self.paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and "png" in file:
                    self.paths.append(file_path)

        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Fetches an item from the dataset.

        This method retrieves a normal image from the dataset and uses the `patch_ex` function
        to synthesize an abnormal image by adding a patch from another image (selected randomly).
        The method returns the normal image, the synthesized abnormal image, the corresponding
        binary mask (indicating abnormal regions), and the image file path.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
                - Normal image tensor (C, H, W): The original, unmodified image.
                - Abnormal image tensor (C, H, W): The synthesized image with an abnormal patch.
                - Mask tensor (1, H, W): Binary mask indicating the abnormal patch region.
                - Image path (str): Path to the image file.
        """
        img_path = self.paths[index]
        img_normal = self.transform(Image.open(img_path).convert("RGB"))
        class_name = img_path.split("/")[-4]
        unique_seed = index

        self_sup_args = {
            "width_bounds_pct": WIDTH_BOUNDS_PCT.get(class_name),
            "intensity_logistic_params": INTENSITY_LOGISTIC_PARAMS.get(class_name),
            "num_patches": 2,  # if single_patch else NUM_PATCHES.get(class_name),
            "min_object_pct": 0,
            "min_overlap_pct": 0.25,
            "gamma_params": (2, 0.05, 0.03),
            "resize": True,
            "shift": True,
            "same": False,
            "mode": cv2.NORMAL_CLONE,
            "label_mode": "logistic-intensity",
            "skip_background": BACKGROUND.get(class_name),
            "random_seed": unique_seed
        }
        if class_name in TEXTURES:
            self_sup_args["resize_bounds"] = (0.5, 2)

        img_normal = np.asarray(img_normal)

        prev = Image.open(self.paths[self.prev_idx]).convert("RGB")
        if self.transform is not None:
            prev = self.transform(prev)
        prev = np.asarray(prev)

        img_abnormal, mask = patch_ex(img_normal, prev, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        mask[mask > 0.15], mask[mask <= 0.15] = 1, 0

        self.prev_idx = index

        img_normal = self.norm_transform(img_normal.copy())
        img_abnormal = self.norm_transform(img_abnormal.copy())

        if np.all(mask.numpy() == 0.0):
            img_abnormal = img_normal

        return img_normal, img_abnormal, mask, img_path

    def collate(
        self, instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
    ) -> Dict[str, List[torch.Tensor]]:
        images = []
        masks = []
        img_paths = []

        for instance in instances:
            images.append(instance[0])
            masks.append(torch.zeros_like(instance[2]))
            img_paths.append(instance[3])

            images.append(instance[1])
            masks.append(instance[2])
            img_paths.append(instance[3])

        return dict(images=images, masks=masks, img_paths=img_paths)
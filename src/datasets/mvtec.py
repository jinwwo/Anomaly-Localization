import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .self_sup_tasks import patch_ex

WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
                    'zipper':((0.03, 0.4), (0.03, 0.2)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}

INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


class MVtecDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        # self.transform = transform
        self.transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.norm_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
        ])
        self.paths = []
        self.img_normals = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and 'png' in file:
                    self.paths.append(file_path)
                    self.img_normals.append(self.transform(Image.open(file_path).convert('RGB')))

        self.prev_idx = np.random.randint(len(self.paths))
        
    def __len__(self):
        return len(self.paths) 

    def __getitem__(self, index):
        img_path, img_normal = self.paths[index], self.img_normals[index]
        class_name = img_path.split('/')[-4]
        
        self_sup_args={
            'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
            'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
            'num_patches': 2, #if single_patch else NUM_PATCHES.get(class_name),
            'min_object_pct': 0,
            'min_overlap_pct': 0.25,
            'gamma_params':(2, 0.05, 0.03), 'resize':True, 
            'shift':True, 
            'same':False, 
            'mode':cv2.NORMAL_CLONE,
            'label_mode':'logistic-intensity',
            'skip_background': BACKGROUND.get(class_name)
        }
        if class_name in TEXTURES:
            self_sup_args['resize_bounds'] = (.5, 2)
            
        img_normal = np.asarray(img_normal)
        
        prev = self.img_normals[self.prev_idx]
        if self.transform is not None:
            prev = self.transform(prev)
        prev = np.asarray(prev)    
        
        img_abnormal, mask = patch_ex(img_normal, prev, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        mask[mask > 0.3], mask[mask <= 0.3] = 1, 0
        
        self.prev_idx = index
        
        img_normal = self.norm_transform(img_normal.copy())
        img_abnormal = self.norm_transform(img_abnormal.copy())
        
        return img_normal, img_abnormal, mask, img_path
    
    def collate(self, instances):
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
            
        return dict(
            images=images,
            masks=masks,
            img_paths=img_paths
        )
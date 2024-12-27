import argparse
import gc
import os
from typing import List, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset

from .datasets.mvtec import MVtecDataset
from .lightning_models import FewShotLightning


def set_cuda_devices(device_ids: str) -> str:
    """
    Configure the CUDA_VISIBLE_DEVICES environment variable based on provided GPU IDs.

    Args:
        device_ids (str): Comma-separated string of GPU IDs (e.g., "0,1,2") or "cpu".

    Returns:
        str: The type of device being used ("gpu" or "cpu").

    Raises:
        ValueError: If the GPU IDs are invalid or not available.
    """
    available_gpus = list(map(int, device_ids.split(",")))

    if device_ids.lower() == "cpu" or not device_ids.strip():
        print("No GPU specified. Using CPU.")
        return "cpu"
    try:
        selected_gpus = list(map(int, device_ids.split(",")))
    except ValueError:
        raise ValueError(f"Invalid GPU IDs format: {device_ids}. Expected comma-separated integers (e.g., '0,1,2').")

    invalid_gpus = [gpu for gpu in selected_gpus if gpu not in available_gpus]
    if invalid_gpus:
        raise ValueError(f"Invalid GPU IDs: {invalid_gpus}. Available GPUs: {available_gpus}")

    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    print(f"CUDA_VISIBLE_DEVICES set to: {device_ids}")
    return "gpu"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for testing the few-shot anomaly detection model.

    Returns:
        argparse.Namespace: Parsed arguments including model name, batch size, output directory,
                            dataset directory, and device configuration.
    """
    parser = argparse.ArgumentParser(
        description="""
            Test pretrained few-shot anomaly detection model
        """
    )
    parser.add_argument('--model_name', type=str, default=None, required=False, help="Feature encoder name")
    parser.add_argument('--batch_size', type=int, default=1, help="Input batch size")
    parser.add_argument('--max_samples', type=int, default=None, required=False, help="data subset size")
    parser.add_argument('--output', type=str, default='./fs_output', help="Output directory")
    parser.add_argument('--dataset', type=str, help="Dataset directory")
    parser.add_argument('--device', type=str, help="List of GPU IDs to use, e.g., '0,1,2,3'")

    args = parser.parse_args()
    device_type = set_cuda_devices(args.device)
    args.device_type = device_type

    return args


def test() -> None:
    """
    Test the pretrained few-shot anomaly detection model on the provided dataset.

    This function prepares the dataset and model, sets up a DataLoader, and performs
    inference using PyTorch Lightning's Trainer.

    Current Status:
        - Only supports few-shot anomaly detection (similarity-based).

    Future Updates:
        1. Support for contrastive anomaly detection testing.
        2. Testing code for integrated models combining few-shot and contrastive learning.

    Returns:
        None
    """
    args = parse_args()
    num_gpus = len(args.device.split(","))

    dataset = MVtecDataset(root_dir=args.dataset)
    model = FewShotLightning(args)

    if args.max_samples:
        subset_indices = list(range(args.max_samples))
        dataset = Subset(dataset, subset_indices)    

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        num_workers=4,
        collate_fn=dataset.collate,
        shuffle=False,
    )

    trainer = L.Trainer(
        accelerator=args.device_type,
        devices=num_gpus,
        strategy="ddp"
    )

    preds = trainer.predict(model, dataloader)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    test()
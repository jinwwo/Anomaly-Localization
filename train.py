from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, random_split

import wandb

from .datasets.mvtec import MVtecDataset
from .lightning_models import FOCALightning
from .utils.utils import to_container


def prepare_dataset(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation DataLoaders."""
    dataset = MVtecDataset(root_dir=cfg.data.root_dir)
    collate_fn = dataset.collate if isinstance(dataset, MVtecDataset) else None

    if cfg.data.test_subset_size > 0:
        indices = torch.randperm(len(dataset))[: cfg.data.test_subset_size]
        dataset = Subset(dataset, indices)

    train_size = int(len(dataset) * cfg.data.train_ratio)
    valid_size = len(dataset) - train_size

    generator = torch.Generator()
    generator.manual_seed(cfg.data.seed)

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader


def prepare_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """Prepare Lightning callbacks."""
    checkpoint_callback = ModelCheckpoint(**to_container(cfg.callbacks.checkpoint))
    earlystopping_callback = EarlyStopping(**to_container(cfg.callbacks.early_stopping))
    lr_monitor = LearningRateMonitor(**to_container(cfg.callbacks.lr_monitor))

    return [checkpoint_callback, earlystopping_callback, lr_monitor]


def prepare_logger(cfg: DictConfig) -> WandbLogger:
    """Prepare WandB logger."""
    return WandbLogger(
        project=f"{cfg.project_name}_{cfg.experiment_name}",
        name=f"SwinT_{cfg.experiment_name}",
    )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    train_loader, valid_loader = prepare_dataset(cfg)

    print(OmegaConf.to_yaml(cfg))
    model = FOCALightning(cfg)
    wandb_logger = prepare_logger(cfg)
    callbacks = prepare_callbacks(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.experiment.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        devices=1
    )

    trainer.fit(model, train_loader, valid_loader)
    wandb.finish()


if __name__ == "__main__":
    main()
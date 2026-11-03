from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import argparse
import pathlib
import friendlywords
from opensr_model.data.DOP20Dataset import DOP20Dataset, MultiSpectralDataset, ImageDataset
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any

from lightning.pytorch.utilities import rank_zero_only

from opensr_model.transformer.swin2sr_enh import SWIN2SRLightning

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])


@rank_zero_only
def save_config(log_dir, config):
    print(f"Logging data to: {log_name}")
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, log_dir / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load", type=pathlib.Path)
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_transformer_repro_org.yaml"
    )
    args = parser.parse_args()
    log_name = pathlib.Path(str(args.config.name).split(".")[0] + "_" + friendlywords.generate(2, separator="-"))

    config = OmegaConf.load(args.config)
    ddconfig = config.transformer_config
    embed_dim = 4
    training_config = config.training_config
    device = training_config.device
    seed = training_config.seed

    L.seed_everything(seed)

    model = SWIN2SRLightning(ddconfig, learning_rate=training_config.learning_rate)

    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    s2_dataset = MultiSpectralDataset(**s2_dataset_config)

    dop20_dataset_config = dataset_config.hr_dataset
    dop20_dataset = ImageDataset(**dop20_dataset_config)

    dataset = DOP20Dataset(
        s2_dataset=s2_dataset, dop20_dataset=dop20_dataset, harmonizer_config=config.get("harmonizer", None)
    )
    # train_dataset = torch.utils.data.Subset(dataset, range(int(0.9 * len(dataset))))
    # val_dataset = torch.utils.data.Subset(dataset, range(int(0.9 * len(dataset)), len(dataset)))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        shuffle=training_config.shuffle,
        pin_memory=training_config.pin_memory,
        num_workers=training_config.num_workers,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        pin_memory=training_config.pin_memory,
        num_workers=training_config.num_workers,
    )

    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=log_name, version="")

    save_config(logger.log_dir, config)
    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=device,
        max_epochs=training_config.epochs,
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.load,
        weights_only=False,
    )

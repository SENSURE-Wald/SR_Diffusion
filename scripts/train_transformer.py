from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import argparse
import pathlib
import friendlywords
from opensr_model.data.DOP20Dataset import create_dop_dataset_splits
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any

from lightning.pytorch.utilities import rank_zero_only

from opensr_model.transformer.swin2sr_enh import SWIN2SRLightning

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])

# A bit dirty but what can you do?
import warnings
import os

warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
torch.set_float32_matmul_precision("high")  # fixes the Tensor Core message
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*isinstance\\(treespec, LeafSpec\\).*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm.*")
warnings.filterwarnings("ignore", message=r".*torch\.meshgrid.*indexing argument.*")
warnings.filterwarnings("ignore", message=r".*Grad strides do not match bucket view strides.*")
warnings.filterwarnings("ignore", message=r".*AccumulateGrad node's stream.*")
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")

# Make child processes (DDP ranks) inherit the same filters
os.environ["PYTHONWARNINGS"] = ",".join(
    [
        "ignore::FutureWarning:timm",
        "ignore:.*torch.meshgrid.*:UserWarning",
        "ignore:.*Grad strides do not match.*:UserWarning",
        "ignore:.*AccumulateGrad node's stream.*:UserWarning",
    ]
)


@rank_zero_only
def save_config(log_dir, config):
    print(f"Logging data to: {log_name}")
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, log_dir / "config.yaml")
    print(
        f"Training starting using {len(train_dataset)} image pairs. This is {round(len(train_dataset) / dataset_total_length, 2) *100 }% of the dataset "
    )
    print(
        f"Validation will be done on {len(val_dataset)} image pairs. This is {round(len(val_dataset) / dataset_total_length, 2) *100 }% of the dataset"
    )
    print(
        f"Testing will be done on {len(test_dataset)} image pairs. This is {round(len(test_dataset) / dataset_total_length, 2) *100 }% of the dataset "
    )


@rank_zero_only
def get_log_name(config_name):
    friendly_word = friendlywords.generate(2, separator="-")
    log_name = pathlib.Path(str(config_name).split(".")[0] + "_" + friendly_word)

    return log_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load", type=pathlib.Path)
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_transformer_repro_org.yaml"
    )
    args = parser.parse_args()
    log_name = get_log_name(args.config.name)
    config = OmegaConf.load(args.config)
    ddconfig = config.transformer_config
    training_config = config.training_config
    device = training_config.device
    seed = training_config.seed

    L.seed_everything(seed)

    model = SWIN2SRLightning(ddconfig, metrics=training_config.metrics, learning_rate=training_config.learning_rate)

    train_dataset, val_dataset, test_dataset = create_dop_dataset_splits(config.dataset_config)

    dataset_total_length = len(train_dataset) + len(val_dataset) + len(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
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

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
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

    trainer.test(model=model, dataloaders=test_dataloader)

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

    parser.add_argument("-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_test_rgbi.yaml")
    args = parser.parse_args()
    log_name = pathlib.Path(str(args.config.name).split(".")[0] + "_" + friendlywords.generate(2, separator="-"))

    config = OmegaConf.load(args.config)
    ddconfig = config.transformer_config
    embed_dim = 4
    testing_config = config.testing_config
    seed = testing_config.seed
    chkpts_path = pathlib.Path(testing_config.checkpoint)
    device = torch.device(f"cuda:{testing_config.device[0]}")

    sorted_chkpts = sorted(chkpts_path.rglob("*.ckpt"), reverse=True)
    checkpoint = sorted_chkpts[0]

    L.seed_everything(seed)

    model = SWIN2SRLightning(ddconfig, metrics=testing_config.metrics)

    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    s2_dataset = MultiSpectralDataset(**s2_dataset_config)

    dop20_dataset_config = dataset_config.hr_dataset
    dop20_dataset = ImageDataset(**dop20_dataset_config)

    dataset = DOP20Dataset(
        s2_dataset=s2_dataset, dop20_dataset=dop20_dataset, harmonizer_config=config.get("harmonizer", None)
    )
    _, dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=testing_config.batch_size,
        shuffle=False,
        pin_memory=testing_config.pin_memory,
        num_workers=testing_config.num_workers,
    )

    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=log_name, version="")

    trainer = L.Trainer(devices=testing_config.device, logger=logger)

    save_config(logger.log_dir, config)

    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=checkpoint, weights_only=False)

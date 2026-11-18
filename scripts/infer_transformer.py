from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import argparse
import pathlib
import friendlywords
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any

import rasterio

from opensr_model.data.DOP20Dataset import MultiSpectralDataset
from opensr_model.transformer.swin2sr_enh import SWIN2SRLightning

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_test_rgbi.yaml")
    parser.add_argument("-i", "--input", type=pathlib.Path, default="florian_s2_tiles_float")
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
    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    dataset = MultiSpectralDataset(**s2_dataset_config)
    infer_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    L.seed_everything(seed)

    lightning_model = SWIN2SRLightning.load_from_checkpoint(
        checkpoint,
        map_location=device,
        weights_only=False,
        metrics=testing_config.metrics,
    )
    lightning_model.to(device).eval()

    for i, lr in enumerate(infer_dataloader):
        output, _ = lightning_model.model(lr.to(device))
        file = dataset.image_paths[i]
        with rasterio.open(file, "r") as tif:
            scale = tif.width / output.shape[-1]
            new_transform = tif.transform * rasterio.Affine.scale(scale)
            output = output.detach().cpu().numpy()[0]
            profile = tif.profile.copy()
            profile.update(
                {
                    "width": 400,
                    "height": 400,
                    "transform": new_transform,
                    "count": 4,
                }
            )

            with rasterio.open(f"florian_ricarda_rgbi/sr_{file.name}", "w", **profile) as dst:
                dst.write(output)

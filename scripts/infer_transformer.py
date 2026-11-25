"""Run SWIN2SR inference on LR tiles and write SR GeoTIFFs.

Usage (from the repo root, so `opensr_model` is importable):
    python -m scripts.infer_transformer [-c opensr_model/configs/<config_file>] <lr_tiles_dir> <out_dir>

<lr_tiles_dir> overrides dataset_config.lr_dataset.data_path from the config.
"""

from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
import lightning as L
import argparse
import pathlib
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any
from tqdm import tqdm

import rasterio

from opensr_model.data.DOP20Dataset import MultiSpectralDataset
from opensr_model.transformer.swin2sr_enh import SWIN2SRLightning

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_test_rgbi.yaml")
    parser.add_argument("INPUT", type=pathlib.Path)
    parser.add_argument("OUTPUT", type=pathlib.Path)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    ddconfig = config.transformer_config
    embed_dim = 4
    testing_config = config.testing_config
    seed = testing_config.seed
    chkpts_path = pathlib.Path(testing_config.checkpoint)
    device = torch.device(f"cuda:{testing_config.device[0]}")

    sorted_chkpts = sorted(chkpts_path.rglob("*.ckpt"), reverse=True)
    if not sorted_chkpts:
        raise FileNotFoundError(f"No .ckpt files under {chkpts_path.resolve()} (testing_config.checkpoint)")
    checkpoint = sorted_chkpts[0]
    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    s2_dataset_config.data_path = str(args.INPUT)

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
    args.OUTPUT.mkdir(parents=True, exist_ok=True)

    for i, lr in tqdm(list(enumerate(infer_dataloader))):
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

            with rasterio.open(args.OUTPUT / f"sr_{file.name}", "w", **profile) as dst:
                dst.write(output)

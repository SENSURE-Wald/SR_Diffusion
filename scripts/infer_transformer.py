from torch.utils.data import DataLoader
import torch
import torchvision
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from rasterio.warp import calculate_default_transform
import argparse
import pathlib
import friendlywords
from opensr_model.data.DOP20Dataset import DOP20Dataset, MultiSpectralDataset, ImageDataset
from omegaconf import DictConfig, ListConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata
from typing import Any
import rasterio
import numpy as np

from opensr_model.transformer.swin2sr_enh import Swin2SR

torch.serialization.add_safe_globals([DictConfig, ListConfig, AnyNode, ContainerMetadata, Any, dict])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load", type=pathlib.Path)
    parser.add_argument("-c", "--config", type=pathlib.Path, default="opensr_model/configs/config_florian_ricarda.yaml")
    args = parser.parse_args()
    log_name = pathlib.Path(str(args.config.name).split(".")[0] + "_" + friendlywords.generate(2, separator="-"))

    config = OmegaConf.load(args.config)
    ddconfig = config.transformer_config
    embed_dim = 4
    training_config = config.training_config
    device = training_config.device
    seed = training_config.seed
    other = config.other

    L.seed_everything(seed)

    # model = SWIN2SRLightning.load_from_checkpoint(other.pretrained)

    dataset_config = config.dataset_config
    s2_dataset_config = dataset_config.lr_dataset
    s2_dataset = MultiSpectralDataset(**s2_dataset_config)

    # dop20_dataset_config = dataset_config.hr_dataset
    # dop20_dataset = ImageDataset(**dop20_dataset_config)

    # dataset = DOP20Dataset(
    #    s2_dataset=s2_dataset, dop20_dataset=dop20_dataset, harmonizer_config=config.get("harmonizer", None)
    # )

    # s2, dop20 = dataset[0]
    # print(s2.min(), s2.max(), dop20.min(), dop20.max())
    model = Swin2SR(**ddconfig).to("cuda")
    checkpoint = torch.load(other.pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    """
    for file in pathlib.Path("florian_s2_tiles_float").rglob("*.tif"):
        with rasterio.open(file) as tif:
            arr = tif.read()[0:13] / 3
            tensor = torch.tensor(arr).to("cuda")
            output, _ = model(tensor.view(1, *tensor.shape))
            output = ((output + 1) / 2).clip(0, 1)
            output = output.detach().cpu().numpy()[0]
            refl_uint8 = (output * 255).astype(np.uint8)
            profile = tif.profile.copy()
            profile.update(
                {
                    "width": 400,
                    "height": 400,
                    "dtype": rasterio.uint8,
                    "count": 3,
                }
            )

        with rasterio.open(f"florian_sr_tiles/sr_{file.name}", "w", **profile) as dst:
            dst.write(refl_uint8)
    for data in s2_dataset:
        data = data.to("cuda")
        output, _ = model(data.view(1, *data.shape))

        output = output.detach().cpu().numpy()[0]
        print(output.min(), output.max())
        import sys
        from PIL import Image

        print(output.transpose(1, 2, 0).shape)
        image = torchvision.transforms.functional.to_pil_image(output.transpose(1, 2, 0))
        image.save("Foo.png")
        sys.exit(0)
    """
    for file in pathlib.Path("s2_output_exkursion_2024").rglob("*.tif"):
        with rasterio.open(file) as tif:
            reordered_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            arr = tif.read()[0:13][reordered_indices] * 3

            print(arr.max(), arr.min(), arr.mean(), arr.std())
            # arr = (arr - mean) / std
            # arr = arr.astype(np.float32)
            tensor = torch.tensor(arr).to("cuda")
            output, _ = model(tensor.view(1, *tensor.shape))
            scale = tif.width / output.shape[-1]
            new_transform = tif.transform * rasterio.Affine.scale(scale)

            output = output.detach().cpu().numpy()[0].clip(0, 1)
            profile = tif.profile.copy()
            profile.update(
                {
                    "width": 400,
                    "height": 400,
                    "dtype": rasterio.uint8,
                    "transform": new_transform,
                    "count": 3,
                }
            )
            image = torchvision.transforms.functional.to_pil_image(output.transpose(1, 2, 0))
            image.save(f"Foo{file.name}.png")
            arr = np.array(image).transpose(2, 0, 1)

            with rasterio.open(f"sr_output_exkursion_hessenforst_2024/sr_{file.name}", "w", **profile) as dst:
                dst.write(arr)
    for i, data in enumerate(s2_dataset):
        if i == 3:
            break
        data = data.to("cuda")
        output, _ = model(data.view(1, *data.shape))
        output = output.detach().cpu().numpy()[0].clip(0, 1)
        image = torchvision.transforms.functional.to_pil_image(output.transpose(1, 2, 0))
        print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
        image.save(f"Foo_{i}.png")

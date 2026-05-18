import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from opensr_model.data.DOP20Dataset import DOP20Dataset, MultiSpectralDataset, ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="opensr_model/configs/config_dop20.yaml", type=Path)
parser.add_argument("-o", "--output_path", type=Path, default="data/DOP20/dop20_tiles_harm")


args = parser.parse_args()

if not args.output_path.is_dir():
    args.output_path.mkdir()

config = OmegaConf.load(args.config)
dataset_config = config.dataset_config

s2_dataset_config = dataset_config.lr_dataset
s2_dataset = MultiSpectralDataset(**s2_dataset_config)

dop20_dataset_config = dataset_config.hr_dataset
dop20_dataset = ImageDataset(**dop20_dataset_config)

dataset = DOP20Dataset(s2_dataset, dop20_dataset, dataset_config.harmonizer)
s2, dop20 = dataset[0]

s2 = s2.detach().cpu().numpy()[0:3, :].transpose(1, 2, 0)
dop20 = dop20.detach().cpu().numpy().transpose(1, 2, 0)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[1].imshow(s2)
axs[1].set_title("S2 Reference")
axs[1].axis("off")

axs[0].imshow(dop20)
axs[0].set_title("DOP20 Reference")
axs[0].axis("off")
plt.tight_layout()
plt.savefig("bar2.png")


for i, (_, dop20) in enumerate(tqdm(dataset)):
    img = to_pil_image(dop20)
    file_name = dop20_dataset.image_paths[i].name
    img.save(args.output_path / file_name)

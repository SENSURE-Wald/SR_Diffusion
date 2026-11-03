from torch.utils.data import Dataset
import torch
import numpy as np
import pathlib
from typing import Tuple
from PIL import Image
from opensr_model.data.harmonization import get_harmonizer
from torchvision import transforms

from opensr_model.utils import load_tif, load_jpg


class MultiSpectralDataset(Dataset):

    def __init__(
        self,
        data_path: pathlib.Path,
        image_size: tuple[int, int],
        channels: list[int] = [0, 1, 2],
        rgb_channels: list[int] = [2, 1, 0],
        file_type="npy",
        gain=3,
        clamp=True,
        **kwargs,
    ):
        self.data_path = pathlib.Path(data_path)
        self.image_size = image_size
        self.channels = channels
        self.rgb_channels = rgb_channels
        self.file_type = file_type
        self.gain = gain
        self.clamp = clamp
        if file_type == "npy":
            self._load_function = np.load
        elif file_type == "jpg":
            self._load_function = load_jpg
        else:
            self._load_function = load_tif

        rgb = [channels[i] for i in rgb_channels]
        rest = [c for c in channels if c not in rgb]
        self.reordered_indices = rgb + rest
        # self.image_paths = sorted(list(self.data_path.rglob(f"**/{self.file_type}")))
        self.image_paths = sorted(list(self.data_path.rglob(f"*.{self.file_type}")))

    def __getitem__(self, i):
        path = self.image_paths[i]
        img = self._load(path)
        img = transforms.Resize(self.image_size)(img)
        # Reorder to be in RGB order
        img = img[self.reordered_indices, ...]
        return img

    def _load(self, path):
        img = (transforms.ToTensor()(self._load_function(path).transpose(1, 2, 0))).to(torch.float32) * self.gain
        return img.clamp(0, 1.0) if self.clamp else img

    def __len__(self):
        return len(self.image_paths)


class ImageDataset(MultiSpectralDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        image_size: tuple[int, int],
        channels: list[int] = [0, 1, 2],
        rgb_channels: list[int] = [0, 1, 2],
        file_type="jpg",
        gain=1.0,
        **kwargs,
    ):
        super().__init__(data_path, image_size, channels, rgb_channels, file_type, gain=gain)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        dataset_0: Dataset,
        dataset_1: Dataset,
    ):
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1

    def __getitem__(self, index):
        return self.dataset_0[index], self.dataset_1[index]

    def __len__(self):
        return self.dataset_0.__len__()


class DOP20Dataset(PairedImageDataset):
    def __init__(self, s2_dataset: MultiSpectralDataset, dop20_dataset: ImageDataset, harmonizer_config):
        super().__init__(dataset_0=s2_dataset, dataset_1=dop20_dataset)
        self.harmonizer = get_harmonizer(harmonizer_config)

    def __getitem__(self, index):
        lr, hr = self.dataset_0[index], self.dataset_1[index]
        hr_transformed = self.harmonizer(lr[:3, :], hr[:3, :])
        hr[0:3, :] = hr_transformed
        return lr, hr

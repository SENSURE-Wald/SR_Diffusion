from torch.utils.data import Dataset
import numpy as np
import pathlib
from typing import Tuple
from PIL import Image
from opensr_model.data.harmonization import get_harmonizer
from torchvision import transforms


class MultiSpectralDataset(Dataset):

    def __init__(
        self,
        data_path: pathlib.Path,
        image_size: tuple[int, int],
        channels: list[int] = [0, 1, 2],
        rgb_channels: list[int] = [2, 1, 0],
        file_type="npy",
        scaling_factor=3000.0,
        **kwargs,
    ):
        self.data_path = pathlib.Path(data_path)
        self.image_size = image_size
        self.channels = channels
        self.rgb_channels = rgb_channels
        self.file_type = file_type
        self.scaling_factor = scaling_factor
        self.reordered_indices = channels.copy()
        for i, idx in enumerate(self.rgb_channels):
            self.reordered_indices[i] = self.channels[idx]

        for i, idx in enumerate(self.rgb_channels):
            self.reordered_indices[i] = self.channels[idx]
        # self.image_paths = sorted(list(self.data_path.rglob(f"**/{self.file_type}")))
        self.image_paths = sorted(list(self.data_path.rglob(f"*.{self.file_type}")))

    def __getitem__(self, i):
        path = self.image_paths[i]
        img = self._load_function(path)
        img = transforms.Resize(self.image_size)(img)
        # Reorder to be in RGB order
        img = img[self.reordered_indices, ...]
        # Get only the channels that are relevant
        return img

    def _load_function(self, path):
        return transforms.ToTensor()(np.load(path).transpose(1, 2, 0)) / self.scaling_factor

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
        **kwargs,
    ):
        super().__init__(data_path, image_size, channels, rgb_channels, file_type)

    def _load_function(self, path):
        # with rasterio.open(path) as src:
        #    arr = src.read(masked=True)  # MaskedArray for all bands
        #    arr_filled = arr.filled(0)  #
        # return ToTensor()(arr_filled)
        return transforms.ToTensor()(Image.open(path))


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
        hr_transformed = self.harmonizer(lr[0:3, :], hr)
        return lr, hr_transformed

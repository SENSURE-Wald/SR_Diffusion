import torch
from torch.utils.data import DataLoader

from opensr_model.data.DOP20Dataset import (
    DOP20Dataset,
    ImageDataset,
    MultiSpectralDataset,
)


class DataPreparator:
    def __init__(self, config_loader, split_ratio=0.9):
        self.config_loader = config_loader
        self.split_ratio = split_ratio

    def prepare_data(self):
        lr_dataset = MultiSpectralDataset(**self.config_loader.lr_dataset_config)
    
        hr_dataset = ImageDataset(**self.config_loader.hr_dataset_config)

        dataset = DOP20Dataset(
            s2_dataset=lr_dataset, dop20_dataset=hr_dataset, harmonizer_config=self.config_loader.yaml.get("harmonizer", None)
        )

        _, dataset = torch.utils.data.random_split(dataset, [self.split_ratio, 1-self.split_ratio])

        return DataLoader(
            dataset=dataset,
            batch_size=self.config_loader.testing_config.batch_size,
            shuffle=False,
            pin_memory=self.config_loader.testing_config.pin_memory,
            num_workers=self.config_loader.testing_config.num_workers,
        )
from torch.utils.data import DataLoader

from opensr_model.data.DOP20Dataset import (
    DOP20Dataset,
    ImageDataset,
    MultiSpectralDataset,
)


class DataPreparator:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def prepare_data(self):
        lr_config = self.config_loader.lr_dataset_config
        hr_config = self.config_loader.hr_dataset_config
        print(
            f"[Benchmark] Loading evaluation data: LR={lr_config.data_path}, HR={hr_config.data_path}",
            flush=True,
        )
        dataset = DOP20Dataset(
            s2_dataset=MultiSpectralDataset(**lr_config),
            dop20_dataset=ImageDataset(**hr_config),
            harmonizer_config=None,
        )
        print(f"[Benchmark] Found {len(dataset)} test samples.", flush=True)
        if len(dataset) == 0:
            raise ValueError(
                f"No evaluation samples found in {lr_config.data_path}. "
                f"Check the configured dataset paths and LR file_type ({lr_config.file_type}). "
                f"Expected matching HR images in {hr_config.data_path}."
            )

        return DataLoader(
            dataset=dataset,
            batch_size=self.config_loader.testing_config.batch_size,
            shuffle=False,
            pin_memory=self.config_loader.testing_config.pin_memory,
            num_workers=self.config_loader.testing_config.num_workers,
        )

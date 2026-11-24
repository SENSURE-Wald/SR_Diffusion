from pathlib import Path

from torch.utils.data import DataLoader

from opensr_model.data.DOP20Dataset import create_split


class DataPreparator:
    def __init__(self, config_loader):
        self.config_loader = config_loader

    def prepare_data(self):
        lr_path = Path(self.config_loader.lr_dataset_config.data_path) / "test" / "s2"
        hr_path = Path(self.config_loader.hr_dataset_config.data_path) / "test" / "dop"
        print(f"[Benchmark] Loading test split: LR={lr_path}, HR={hr_path}", flush=True)
        dataset = create_split(
            lr_config=self.config_loader.lr_dataset_config,
            hr_config=self.config_loader.hr_dataset_config,
            split="test",
        )
        print(f"[Benchmark] Found {len(dataset)} test samples.", flush=True)
        if len(dataset) == 0:
            raise ValueError("The test split is empty. Check the dataset paths (<root>/test/s2 and <root>/test/dop).")

        return DataLoader(
            dataset=dataset,
            batch_size=self.config_loader.testing_config.batch_size,
            shuffle=False,
            pin_memory=self.config_loader.testing_config.pin_memory,
            num_workers=self.config_loader.testing_config.num_workers,
        )

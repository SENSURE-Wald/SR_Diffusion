from omegaconf import OmegaConf


class ConfigLoader:
    def __init__(self, config_path, testing_config=None):
        self.yaml = OmegaConf.load(config_path)
        if testing_config is not None:
            # Checkpoints belong to the model; evaluation settings belong to the benchmark.
            shared_settings = {
                key: value for key, value in testing_config.items()
                if key not in ("checkpoint", "weights_only")
            }
            self.yaml.testing_config = OmegaConf.merge(
                self.yaml.get("testing_config", {}), shared_settings
            )
        self.name = self.yaml.name

class SetupConfigLoader(ConfigLoader):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.dataset_config = self.yaml.dataset_config
        self.lr_dataset_config = self.yaml.dataset_config.lr_dataset
        self.hr_dataset_config = self.yaml.dataset_config.hr_dataset
        self.testing_config = self.yaml.testing_config

class OpenSRConfigLoader(ConfigLoader):
    def __init__(self, config_path, testing_config=None):
        super().__init__(config_path, testing_config)
        self.ckpt_version = self.yaml.ckpt_version
        self.testing_config = self.yaml.testing_config
        self.input_channel_indices = self.yaml.input_channel_indices

class Swin2SRConfigLoader(ConfigLoader):
    def __init__(self, config_path, testing_config=None):
        super().__init__(config_path, testing_config)
        self.transformer_config = self.yaml.transformer_config
        self.testing_config = self.yaml.testing_config

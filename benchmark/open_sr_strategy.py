import lightning as L
import torch

import opensr_model
from benchmark.config_loader import OpenSRConfigLoader
from benchmark.model_tester import OpenSRModelTester
from benchmark.strategy import Strategy


class OpenSRStrategy(Strategy):
    def execute(self):
        print ("Executing OpenSR strategy...")

        config = OpenSRConfigLoader(self.config_path, self.testing_config)
        L.seed_everything(config.testing_config.get("seed", 42), workers=True)
        devices = config.testing_config.get("device", [0])
        device_index = devices[0] if not isinstance(devices, int) else 0
        device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

        model = opensr_model.SRLatentDiffusion(config.yaml, device=device)  # create model
        model.load_pretrained(config.ckpt_version)  # load checkpoint
        assert model.training == False, "Model has to be in eval mode."

        tester = OpenSRModelTester(config_loader=config, data_loader=self.data_loader, model_wrapper=model)
        return tester.test()

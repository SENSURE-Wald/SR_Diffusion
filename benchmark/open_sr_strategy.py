import torch

import opensr_model
from benchmark.config_loader import OpenSRConfigLoader
from benchmark.model_tester import OpenSRModelTester
from benchmark.strategy import Strategy


class OpenSRStrategy(Strategy):
    def __init__(self, data_loader, config_path):
        super().__init__(data_loader, config_path)

    def execute(self):
        print ("Executing OpenSR strategy...")

        device = "cuda" if torch.cuda.is_available() else "cpu"  # try to run on GPU
        config = OpenSRConfigLoader(self.config_path) # load config

        model = opensr_model.SRLatentDiffusion(config.yaml, device=device)  # create model
        model.load_pretrained(config.ckpt_version)  # load checkpoint
        assert model.training == False, "Model has to be in eval mode."

        tester = OpenSRModelTester(config_loader=config, data_loader=self.data_loader, model_wrapper=model)
        tester.test()
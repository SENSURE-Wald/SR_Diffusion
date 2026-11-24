import lightning as L

from benchmark.config_loader import Swin2SRConfigLoader
from benchmark.model_tester import LightningModelTester, TransfomerModelWrapper
from benchmark.strategy import Strategy
from opensr_model.transformer.swin2sr_enh import SWIN2SRLightning


class Swin2SRStrategy(Strategy):
    def execute(self):
        print ("Executing Swin2SR strategy...")

        config_loader = Swin2SRConfigLoader(self.config_path, self.testing_config)
        ddconfig = config_loader.transformer_config
        testing_config = config_loader.testing_config
        seed = testing_config.seed
    
        L.seed_everything(seed, workers=True)
    
        model = SWIN2SRLightning(ddconfig, metrics=testing_config.metrics)

        model_wrapper = TransfomerModelWrapper(model=model, weights_only=testing_config.get("weights_only", False), ckpt_path="")
        model_tester = LightningModelTester(config_loader, self.data_loader, model_wrapper=model_wrapper)
        model_tester.prepare_checkpoint()
        return model_tester.test()

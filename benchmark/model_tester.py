import pathlib

import friendlywords
import lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from opensr_model.metrics import create_metric


@rank_zero_only
def save_config(log_dir, config, log_name):
    print(f"Logging data to: {log_name}")
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, log_dir / "config.yaml")

class TransfomerModelWrapper:
    def __init__(self, model, ckpt_path, weights_only=False):
        self.model = model
        self.ckpt_path = ckpt_path
        self.weights_only = weights_only

class ModelTester:
    def __init__(self, config_loader, data_loader, model_wrapper):
        self.config = config_loader
        self.data_loader = data_loader
        self.model_wrapper = model_wrapper

    def init_logger(self):
        log_name = pathlib.Path(str(self.config.yaml.name).split(".")[0] + "_" + friendlywords.generate(2, separator="-"))
        logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=log_name, version="")
        save_config(logger.log_dir, self.config.yaml, log_name=log_name)
        return logger

class LightningModelTester(ModelTester):
    def __init__(self, config_loader, data_loader, model_wrapper):
        super().__init__(config_loader, data_loader, model_wrapper)

    def prepare_checkpoint(self):
        chkpts_path = pathlib.Path(self.config.testing_config.checkpoint)
        sorted_chkpts = [chkpts_path] if chkpts_path.is_file() else sorted(chkpts_path.rglob("*.ckpt"), reverse=True)
        if not sorted_chkpts:
            raise FileNotFoundError(f"No checkpoint found at {chkpts_path}")
        self.model_wrapper.ckpt_path = sorted_chkpts[0]

    def test(self):
        trainer = L.Trainer(devices=self.config.testing_config.device, logger=self.init_logger())
        return trainer.test(model=self.model_wrapper.model, dataloaders=self.data_loader, ckpt_path=self.model_wrapper.ckpt_path, weights_only=self.model_wrapper.weights_only)

class OpenSRModelTester(ModelTester):
    def __init__(self, config_loader, data_loader, model_wrapper):
        super().__init__(config_loader, data_loader, model_wrapper)

    def test(self):
        model = self.model_wrapper
        model.eval()

        device = next(model.parameters()).device
        metric_names = self.config.testing_config.metrics

        metrics = [
            create_metric(name, device=device)
            for name in metric_names
        ]

        # Configure the TensorBoard logger.
        logger = self.init_logger()

        totals = {name: 0.0 for name in metric_names}
        num_images = 0

        with torch.inference_mode():
            for batch_idx, (lr, hr) in enumerate(self.data_loader):
                print(
                    f"[Benchmark] OpenSR: processing batch {batch_idx + 1}/{len(self.data_loader)} "
                    f"({num_images}/{len(self.data_loader.dataset)} samples completed).",
                    flush=True,
                )
                lr = lr.to(device=device, dtype=torch.float32)
                hr = hr.to(device=device, dtype=torch.float32)

                indices = list(self.config.input_channel_indices)
                lr_opensr = lr[:, indices, :, :].contiguous()

                # SRLatentDiffusion reads sampling settings from its own config.
                sr = model(lr_opensr)

                # look at SWIN2SRLightning class in swin2sr_enh.py
                sr = sr.clamp(0, 1).contiguous()
                hr = hr.contiguous()

                if sr.shape != hr.shape:
                    raise ValueError(
                        f"SR und HR müssen für den Metrikvergleich gleich groß sein: "
                        f"SR={tuple(sr.shape)}, HR={tuple(hr.shape)}"
                    )

                batch_size = hr.shape[0]
                num_images += batch_size

                for name, metric in zip(metric_names, metrics):
                    value = metric(hr, sr)
                    totals[name] += value.item() * batch_size

        results = {}
        if num_images == 0:
            raise ValueError("Der Testdatensatz ist leer.")

        results = {
            name: total / num_images
            for name, total in totals.items()
        }

        # Log one aggregated value per metric for this test run.
        for name, value in results.items():
            logger.experiment.add_scalar(f"test/{name}", value, global_step=0)

        logger.experiment.flush()
        return results

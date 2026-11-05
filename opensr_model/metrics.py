import torch
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from abc import ABC, abstractmethod


class Metric(torch.nn.Module, ABC):
    def __init__(self, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    @staticmethod
    def name():
        raise NotImplementedError("Abstract class Metric should not be used and cannot be initialised.")

    @abstractmethod
    def forward(self, gt, output):
        pass


class LPIPSMetric(Metric):

    def __init__(
        self,
        net_type="vgg",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(self.device)

    @staticmethod
    def name():
        return "LPIPS"

    def forward(self, gt, output):
        return self.lpips(gt[:, :3, :, :], output[:, :3, :, :])


class SSIMMetric(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    @staticmethod
    def name():
        return "SSIM"

    def forward(self, gt, output):
        return self.ssim(gt, output)


class PSNRMetric(Metric):

    def __init__(self, data_range=(0, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(self.device)

    @staticmethod
    def name():
        return "PSNR"

    def forward(self, gt, output):
        return self.psnr(gt, output)


def create_metric(name, *args, **kwargs):
    if name == LPIPSMetric.name():
        return LPIPSMetric(*args, **kwargs)
    elif name == SSIMMetric.name():
        return SSIMMetric(*args, **kwargs)
    elif name == PSNRMetric.name():
        return PSNRMetric(*args, **kwargs)

    else:
        raise NotImplementedError(f"Metric with name {name} is not implemented.")

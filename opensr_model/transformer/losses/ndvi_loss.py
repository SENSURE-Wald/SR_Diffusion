import torch

from opensr_model.transformer.utils import ndvi


class NDVILoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.MSELoss(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion

    def forward(self, output, hr):
        """
        output: (B, 4, H_out, W_out) - model prediction, RGBNIR
        hr:     (B, 4, H_hr,  W_hr)  - input GT, RGBNIR
        """
        hr_red = hr[:4, 0:1, :, :]
        hr_nir = hr[:4, 3:4, :, :]

        output_red = output[:4, 0:1, :, :]
        output_nir = output[:4, 3:4, :, :]

        hr_ndvi = ndvi(hr_red, hr_nir)
        output_ndvi = ndvi(output_red, output_nir)

        return self.criterion(hr_ndvi, output_ndvi)

from opensr_model.transformer.utils import rgb_to_ycbcr, GaussianBlur
import torch
import torch.nn.functional as F


class DownsampleSimilarityLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.L1Loss()

    def forward(self, output, lr):
        """
        output: (B, 3, H_out, W_out) - model prediction, RGB
        lr:     (B, 3, H_lr,  W_lr)  - input LR, RGB (already sliced to [:, :3, :, :] by caller)
        """
        scale_factor = output.shape[-1] // lr.shape[-1]  # assumes integer scale, e.g. 4
        output_downscaled = F.avg_pool2d(output, kernel_size=scale_factor, stride=scale_factor)
        return self.l1(output_downscaled, lr)


class ChromaSimilarityLoss(torch.nn.Module):
    def __init__(self, blur_sigma=2.0, kernel_size=9):
        super().__init__()
        self.blur = GaussianBlur(channels=2, kernel_size=kernel_size, sigma=blur_sigma)
        self.l1 = torch.nn.L1Loss()

    def forward(self, output, lr):
        """
        output: (B, 3, H_out, W_out) - model prediction, RGB
        lr:     (B, 3, H_lr,  W_lr)  - input LR, RGB (already sliced to [:, :3, :, :] by caller)
        """
        out_ycbcr = rgb_to_ycbcr(output)
        lr_ycbcr = rgb_to_ycbcr(lr)

        out_chroma = out_ycbcr[:, 1:3, :, :]  # Cb, Cr only
        lr_chroma = lr_ycbcr[:, 1:3, :, :]

        # bring output chroma down to LR's spatial size, blur both to
        # strip spatial detail and only compare color/tone
        out_chroma_ds = torch.nn.functional.interpolate(
            out_chroma, (lr.shape[-2], lr.shape[-1]), mode="bilinear", antialias=True
        )
        out_chroma_blur = self.blur(out_chroma_ds)
        lr_chroma_blur = self.blur(lr_chroma)

        return self.l1(out_chroma_blur, lr_chroma_blur)

from torch import nn
import importlib
import torch
import numpy as np
import random

from enum import Enum
from torchmetrics.classification import MulticlassF1Score
import torchvision

from matplotlib.colors import LinearSegmentedColormap


def create_image_grid(*images, nrow=4):
    size = images[0].size()[0]
    images = torch.cat(images, dim=0)
    grid = torchvision.utils.make_grid(images, nrow=min(size, nrow))
    return grid


def log_image_output(logger, lr, output, hr, step, phase="val"):
    lr_upscaled = torch.nn.functional.interpolate(
        lr[:4][:, :3, :, :], (output.shape[-2], output.shape[-1]), mode="bicubic"
    )
    grid = create_image_grid(lr_upscaled, hr[:4][:, :3, :, :], output[:4][:, :3, :, :])
    logger.experiment.add_image(f"{phase}images", grid, step)

    if hr.shape[1] == 4:
        grid = create_image_grid(hr[:4, 3:4, :, :], output[:4, 3:4, :, :])
        logger.experiment.add_image(f"{phase}/images_ir", grid, step)

        # Compute NDVI
        hr_red = hr[:4, 0:1, :, :]
        hr_nir = hr[:4, 3:4, :, :]

        output_red = output[:4, 0:1, :, :]
        output_nir = output[:4, 3:4, :, :]

        hr_ndvi = colorize(ndvi(hr_red, hr_nir))
        output_ndvi = colorize(ndvi(output_red, output_nir))

        grid = create_image_grid(hr_ndvi, output_ndvi)
        logger.experiment.add_image(f"{phase}/images_ndvi", grid, step)

        # Add CIR Images
        hr_cir = hr[:4, (3, 0, 1)]
        output_cir = output[:4, (3, 0, 1)]
        grid = create_image_grid(hr_cir, output_cir)
        logger.experiment.add_image(f"{phase}/images_cir", grid, step)


def ndvi(red, nir, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def colorize(x, vmin=0.0, vmax=1.0, cmap=LinearSegmentedColormap.from_list("white_green", ["white", "green"])):
    """x: (B, 1, H, W) tensor -> (B, 3, H, W) RGB tensor."""
    x = x.detach().float().cpu().squeeze(1)
    x = ((x - vmin) / (vmax - vmin)).clamp(0, 1)
    rgb = cmap(x.numpy())[..., :3]
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float()


class GaussianBlur(torch.nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel_2d = g[:, None] @ g[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).clone()
        self.register_buffer("weight", kernel)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


def rgb_to_ycbcr(x):
    """
    x: (B, 3, H, W), values in [0, 1]
    Returns YCbCr in [0, 1] (Y) and [-0.5, 0.5]-ish range for Cb/Cr (ITU-R BT.601 full-range).
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b

    return torch.cat([y, cb, cr], dim=1)


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_fun(fullname):
    path, name = fullname.rsplit(".", 1)
    return getattr(importlib.import_module(path), name)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        self.avg_item = self.avg.tolist()
        fmtstr = "{avg_item" + self.fmt + "}"
        try:
            return fmtstr.format(**self.__dict__)
        except TypeError:
            # print a list of elements
            fmtstr = "{" + self.fmt + "}"
            return " ".join([fmtstr for _ in range(len(self.avg_item))]).format(*self.avg_item)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError("Invalid summary type {}".format(self.summary_type))

        return fmtstr.format(**self.__dict__)


class F1AverageMeter(AverageMeter):
    def __init__(self, cfg, average, **kwargs):
        self.cfg = cfg
        self._cfg_average = average
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._to_update = True
        self.fun = MulticlassF1Score(len(self.cfg.classes), average=self._cfg_average).to(self.cfg.device)

    @property
    def avg(self):
        if self._to_update:
            self._avg = self.fun.compute()
            self._to_update = False
        return self._avg

    @avg.setter
    def avg(self, value):
        self._avg = value
        self._to_update = False

    def update(self, val, n=1):
        pred, gt = val
        self.fun.update(pred, gt)
        self._to_update = True


def set_required_grad(model, value):
    for parameters in model.parameters():
        parameters.requires_grad = value

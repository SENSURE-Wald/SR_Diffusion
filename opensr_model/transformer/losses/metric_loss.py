import piq
import torch
from opensr_model.transformer.utils import load_fun
import opensr_model.transformer.losses.swt_loss as wt_loss


def norm_0_to_1(fun):
    def wrapper(cfg):
        dset = cfg.dataset
        use_minmax = cfg.dataset.get("stats").get("use_minmax", False)
        denorm = load_fun(dset.get("denorm"))(cfg, hr_name=cfg.dataset.hr_name, lr_name=cfg.dataset.lr_name)
        evaluable = load_fun(dset.get("printable"))(
            cfg, hr_name=cfg.dataset.hr_name, lr_name=cfg.dataset.lr_name, filter_outliers=False, use_minmax=use_minmax
        )

        rfun = fun(cfg)

        def f(sr, hr):
            hr, sr, _ = evaluable(*denorm(hr, sr, None))
            sr = sr.clamp(min=0, max=1)
            hr = hr.clamp(min=0, max=1)
            return rfun(sr, hr)

        return f

    return wrapper


@norm_0_to_1
def ssim_loss(cfg):
    criterion = piq.SSIMLoss()

    def f(sr, hr):
        return criterion(sr, hr)

    return f


@norm_0_to_1
def swt_loss(cfg):
    criterion = wt_loss.SWTLoss().to("cuda")

    def f(sr, hr):
        return criterion(sr, hr)

    return f


def _cc_single_torch(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    """
    Compute the Cross-Correlation (CC) metric between two input tensors representing images.

    CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

    Args:
        raw_tensor (torch.Tensor): The image tensor to be compared.
        dst_tensor (torch.Tensor): The reference image tensor.

    Returns:
        CC (torch.Tensor): The Cross-Correlation (CC) metric score.

    """
    N_spectral = raw_tensor.shape[1]

    # Reshaping fused and reference data
    raw_tensor_reshaped = raw_tensor.reshape(N_spectral, -1)
    dst_tensor_reshaped = dst_tensor.reshape(N_spectral, -1)

    # Calculating mean value
    mean_raw = torch.mean(raw_tensor_reshaped, 1).unsqueeze(1)
    mean_dst = torch.mean(dst_tensor_reshaped, 1).unsqueeze(1)

    CC = torch.sum((raw_tensor_reshaped - mean_raw) * (dst_tensor_reshaped - mean_dst), 1) / torch.sqrt(
        torch.sum((raw_tensor_reshaped - mean_raw) ** 2, 1) * torch.sum((dst_tensor_reshaped - mean_dst) ** 2, 1)
    )

    CC = torch.mean(CC)

    return CC


def cc_loss(sr, hr):
    cc_value = _cc_single_torch(sr, hr)
    return 1 - ((cc_value + 1) * 0.5)

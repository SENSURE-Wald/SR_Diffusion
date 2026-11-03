from typing import Optional

import torch
import torch.nn.functional as F
from opensr_model.utils import hq_histogram_matching, gaussian_blur_channelwise


class Harmonizer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, _: torch.Tensor, hr: torch.Tensor):
        return hr


class GammaHarmonizer(Harmonizer):
    """
    Based on:
      https://github.com/ESAOpenSR/opensr-degradation/blob/main/opensr_degradation/naipd/harmonization.py
    And [Aybar et al., 2024]
    """

    def __init__(self, gamma: torch.Tensor, sigma: torch.Tensor):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, lr: torch.Tensor, hr: torch.Tensor):
        hr_hist = hq_histogram_matching(lr, hr)
        log_hr = hr_hist.log()
        hr_gamma = torch.exp(log_hr / self.gamma.view(1, -1, 1, 1))
        hr_blurred = gaussian_blur_channelwise(hr_gamma, self.sigma)
        return hr_blurred


class ColouringHarmonizer(Harmonizer):
    """
    Given an image I_c and I_s the goal is to find matrix A such that:
              Cov(AI_c) = Cov(I_s)
              µ(AI_c)   = µ(I_s)

    Whitening transform produces I' from I_c with Cov(I') = Unit.
              I' = W I_c
              W  = Λ_c^{−1/2} Φ_c^T I_c
              with Cov(I_c) = Φ_c Λ_c^{1/2} Λ_c^{1/2} Φ_c^T

    Colouring transform transforms unit random variable I' to random variable with desired Cov.
            Cov(C I') = Cov(I_s)
            µ(C I')   = Cov(I_s)
    with
                   C  = Φ_s Λ_s^{1/2} I'

    Now we also need to adjust the mean values. For this we need to first move the I_c by the mean µ_c

                  I'' = C(W I_c - µ_c) I_c + µ_s.

    Further reading: https://www.projectrhea.org/rhea/index.php/ECE662_Whitening_and_Coloring_Transforms_S14_MH

    Code and method based on: https://github.com/AronKovacs/g-style
    """

    def __init__(self, eps=1e-8, min_pixels=100):
        super().__init__()
        self.eps = eps
        self.min_pixels = min_pixels

    @staticmethod
    def _masked_stats(img, mask):
        """img: [N, C], mask: [N] bool (True = don't use pixel)"""
        x = img[~mask]
        mu = x.mean(0, keepdim=True)
        xc = x - mu
        cov = xc.T @ xc / float(x.size(0))
        return mu, cov

    @staticmethod
    def apply_color_match(img, alpha, beta):
        c, h, w = img.shape
        img = img.permute(1, 2, 0).reshape(-1, c)  # [bs, c, h, w] -> [bs, h * w, c]
        img = img @ alpha + beta
        img = img.reshape(h, w, c).permute(2, 0, 1)  # [bs, h * w, c] -> [bs, c, h, w]
        return torch.clamp(img, 0.0, 1.0)

    def find_color_match(self, input_image, gt_image, mask=None):
        """
        input_img: [C, H, W]
        gt_image:  [C, H, W]
        mask:      [H, W]     True = unclear / invalid pixel
        """
        c = gt_image.shape[0]
        gt = gt_image.permute(1, 2, 0).reshape(-1, c)
        inp = input_image.permute(1, 2, 0).reshape(-1, c).to(gt.device)

        if mask is None:
            mask = torch.zeros(gt.shape[0], dtype=torch.bool, device=gt.device)

        else:
            mask = mask.reshape(-1).to(device=gt.device, dtype=torch.bool)

        if (~mask).sum() < self.min_pixels:
            return torch.eye(c, device=gt.device), torch.zeros(1, c, device=gt.device)

        # Mean values
        mu_c, cov_c = self._masked_stats(gt, mask)
        mu_s, cov_s = self._masked_stats(inp, mask)

        # N, 3, 3
        # Cov(I_c) = Φ_c Λ_c^{1/2} Λ_c^{1/2} Φ_c^T
        u_c, sig_c, _ = torch.svd(cov_c)
        # Cov(I_s) = Φ_s Λ_s^{1/2} Λ_s^{1/2} Φ_s^T
        u_s, sig_s, _ = torch.svd(cov_s)

        # Φ_c^T
        u_c_i = u_c.T
        # Φ_s^T
        u_s_i = u_s.T

        # Inverse of second component of SVD
        # Λ_c^{−1/2}
        scl_c = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        # Λ_s^{1/2}
        scl_s = torch.diag_embed(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        alpha = (u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i).T
        beta = mu_s - mu_c @ alpha

        return alpha, beta

    def forward(self, lr: torch.Tensor, hr: torch.Tensor, invalid_mask: torch.Tensor = None):
        # Downsample the DOP to the Sentinel grid for estimating the transform only
        if hr.shape[-2:] != lr.shape[-2:]:
            hr_small = F.interpolate(hr[None], size=lr.shape[-2:], mode="area")[0]
        else:
            hr_small = hr

        alpha, beta = self.find_color_match(lr, hr_small, invalid_mask)
        return self.apply_color_match(hr, alpha, beta)


def get_harmonizer(harmonizer_config: Optional[dict]) -> Harmonizer:
    name = harmonizer_config.name if harmonizer_config is not None else ""
    match name:
        case "gamma":
            return GammaHarmonizer(harmonizer_config)
        case "colouring":
            return ColouringHarmonizer(harmonizer_config)
        case _:
            return Harmonizer()

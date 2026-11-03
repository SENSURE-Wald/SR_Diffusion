#
# Source code: https://github.com/mv-lab/swin2sr
#
# -----------------------------------------------------------------------------------
# Swin2SR: Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration, https://arxiv.org/abs/2209.11345
# Written by Conde and Choi et al.
# Adjustments made by Haider Ali: RGB Super-Resolution for Multispectral Remote Sensing Images, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11242707
# -----------------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import lightning as L
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import piq
from pytorch_lightning.utilities import rank_zero_only
import torchvision

from opensr_model.transformer.losses.metric_loss import cc_loss
from opensr_model.transformer.losses.ndvi_loss import NDVILoss
from opensr_model.transformer.losses.swt_loss import SWTLoss

from opensr_model.transformer.utils import window_reverse, Mlp, window_partition, ndvi, colorize
from opensr_model.transformer.HFM import one_module
from opensr_model.transformer.losses.similarity_loss import ChromaSimilarityLoss, DownsampleSimilarityLoss


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, res_scale=1):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(default_conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = BasicBlock(default_conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(
            N, -1, -1, -1
        )  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum("btf,bfhi->bhti", x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(
            hash_codes + offsets,
            (
                N,
                -1,
            ),
        )  # [N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        # number of hash buckets/hash bits
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = indices % L  # now range from (0->H*W)
        x_embed_sorted = batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(
            x_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C)
        )  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum(
            "bhkie,bhkje->bhkij", x_att_buckets, x_match
        )  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum("bukij,bukje->bukie", score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(
            bucket_score,
            (
                N,
                -1,
            ),
        )  # [N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        use_lepe=False,
        use_cpb_bias=True,
        use_rpe_bias=False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.use_cpb_bias = use_cpb_bias

        if self.use_cpb_bias:
            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
            )

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = (
                torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
            )  # 1, 2*Wh-1, 2*Ww-1, 2
            if pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
            else:
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
            )

            self.register_buffer("relative_coords_table", relative_coords_table)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.use_rpe_bias = use_rpe_bias
        if self.use_rpe_bias:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            rpe_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("rpe_relative_position_index", rpe_relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.use_lepe = use_lepe
        if self.use_lepe:
            self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.use_lepe:
            lepe = self.lepe_pos(v)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01)).to(self.logit_scale.device)
        ).exp()
        attn = attn * logit_scale

        if self.use_cpb_bias:
            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)

        if self.use_rpe_bias:
            relative_position_bias = self.relative_position_bias_table[self.rpe_relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = attn @ v

        if self.use_lepe:
            x = x + lepe

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def lepe_pos(self, v):
        B, NH, HW, NW = v.shape
        C = NH * NW
        H = W = int(math.sqrt(HW))
        v = v.transpose(-2, -1).contiguous().view(B, C, H, W)
        lepe = self.get_v(v)
        lepe = lepe.reshape(-1, self.num_heads, NW, HW)
        lepe = lepe.permute(0, 1, 3, 2).contiguous()
        return lepe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        use_rpe_bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            use_lepe=use_lepe,
            use_cpb_bias=use_cpb_bias,
            use_rpe_bias=use_rpe_bias,
        )

        self.conv_block = CAB(num_feat=dim, compress_ratio=3, squeeze_factor=30)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if MoE_config is None:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            print(MoE_config)
            self.mlp = MoE(input_size=dim, output_size=dim, hidden_size=mlp_hidden_dim, **MoE_config)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        # Conv_x
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x)) + conv_x

        # FFN

        loss_moe = None
        res = self.mlp(x)
        if not torch.is_tensor(res):
            res, loss_moe = res

        x = x + self.drop_path(self.norm2(res))

        return x, loss_moe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        pretrained_window_size=0,
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        use_rpe_bias=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    use_lepe=use_lepe,
                    use_cpb_bias=use_cpb_bias,
                    MoE_config=None,
                    use_rpe_bias=use_rpe_bias,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        loss_moe_all = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)

            if not torch.is_tensor(x):
                x, loss_moe = x
                loss_moe_all += loss_moe or 0

        if self.downsample is not None:
            x = self.downsample(x)
        return x, loss_moe_all

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1],
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        #  MoE_config=None,
        use_rpe_bias=False,
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_lepe=use_lepe,
            use_cpb_bias=use_cpb_bias,
            MoE_config=None,
            use_rpe_bias=use_rpe_bias,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = (
                nn.Sequential(
                    nn.Conv2d(dim, dim // 4, 3, 1, 1),
                    nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(dim // 4, dim, 3, 1, 1),
                ),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None
        )

    def forward(self, x, x_size):
        loss_moe = None
        res = self.residual_group(x, x_size)

        if not torch.is_tensor(res):
            res, loss_moe = res

        res = self.patch_embed(self.conv(self.patch_unembed(res, x_size)))
        return res + x, loss_moe

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n and 3.")
        super(Upsample, self).__init__(*m)


class Upsample_hf(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n and 3.")
        super(Upsample_hf, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        """
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        """

        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        """

        residual = self.block(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, in_channels=13, num_layers=2, kernel_size=3, channel_size=96):
        """
        Args:
            config : dict, configuration file
        """

        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.channel_size,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.PReLU(),
        )

        res_layers = [ResidualBlock(self.channel_size, self.kernel_size) for _ in range(self.num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_size,
                out_channels=self.channel_size,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )
        )

    def forward(self, x):
        """
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        """

        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class Attention(nn.Module):

    def __init__(self, embed_dim, fft_norm="ortho"):
        # bn_layer not used
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real)))))
        imag = ffted.imag + self.conv_layer3(self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag)))))
        ffted = torch.complex(real, imag)

        ifft_shape_slice = x.shape[-2:]

        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return x * output


#####################################
# Addition on shallow feature extractor for extracting shallow features written by CHATGPT-o1 preview
#####################################
class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpectralAttention, self).__init__()
        # Spectral attention mechanism
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Pool over spatial dimensions
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        y = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        y = self.fc(y)  # (batch_size, channels, 1, 1)
        return x * y  # Element-wise multiplication


class MultispectralFeatureExtractor(nn.Module):
    def __init__(self, in_channels=13, out_channels=96, use_spectral_attention=True):
        super(MultispectralFeatureExtractor, self).__init__()
        self.use_spectral_attention = use_spectral_attention

        if self.use_spectral_attention:
            self.spectral_attention = SpectralAttention(in_channels)

        # 3D Convolutional Layer
        self.conv3d = nn.Conv3d(
            in_channels=1,  # After unsqueezing, treat spectral bands as depth
            out_channels=out_channels,
            kernel_size=(in_channels, 3, 3),  # Kernel spans all spectral bands
            stride=(1, 1, 1),
            padding=(0, 1, 1),  # No padding in spectral dimension
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        if self.use_spectral_attention:
            x = self.spectral_attention(x)
        x = x.unsqueeze(1)  # (batch_size, 1, in_channels, height, width)
        x = self.conv3d(x)  # (batch_size, out_channels, 1, height, width)
        x = self.activation(x)
        x = x.squeeze(2)  # Remove the spectral dimension (singleton)
        return x  # Output: (batch_size, out_channels, height, width)


class Swin2SR(nn.Module):
    r"""Swin2SR
        A PyTorch impl of : `Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration`.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=13,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.0,
        upsampler="",
        resi_connection="1conv",
        use_lepe=False,
        use_cpb_bias=True,
        MoE_config=None,
        use_rpe_bias=False,
        output_chans=3,
        **kwargs,
    ):
        super(Swin2SR, self).__init__()
        print("==== SWIN 2SR With High Frequency Block")
        num_in_ch = in_chans
        num_out_ch = output_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.nlsa1 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=3, reduction=3, res_scale=1)
        self.nlsa2 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=3, reduction=3, res_scale=1)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        # High Frequency Module
        # https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Lu_Transformer_for_Single_Image_Super-Resolution_CVPRW_2022_paper.pdf
        self.layers_hfm = nn.ModuleList()

        for i_layer in range(2):
            layer = one_module(embed_dim)
            self.layers_hfm.append(layer)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_lepe=use_lepe,
                use_cpb_bias=use_cpb_bias,
                MoE_config=None,
                use_rpe_bias=use_rpe_bias,
            )
            self.layers.append(layer)

        if self.upsampler == "pixelshuffle_hf":
            self.layers_hf = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = RSTB(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # no impact on SR results
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                    use_lepe=use_lepe,
                    use_cpb_bias=use_cpb_bias,
                    MoE_config=None,
                    use_rpe_bias=use_rpe_bias,
                )
                self.layers_hf.append(layer)

        self.norm = norm_layer(self.num_features)

        self.nlsa3 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=3, reduction=3, res_scale=1)
        self.nlsa4 = NonLocalSparseAttention(channels=embed_dim, chunk_size=36, n_hashes=3, reduction=3, res_scale=1)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffle_aux":
            self.conv_bicubic = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_aux = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == "pixelshuffle_hf":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.upsample_hf = Upsample_hf(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_first_hf = nn.Sequential(nn.Conv2d(num_feat, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_after_body_hf = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_before_upsample_hf = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_last_hf = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1])
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        loss_moe_all = 0
        for layer in self.layers:
            x = layer(x, x_size)

            if not torch.is_tensor(x):
                x, loss_moe = x
                loss_moe_all += loss_moe or 0

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, loss_moe_all

    def forward_features_hf(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        loss_moe_all = 0
        for layer in self.layers_hf:
            x = layer(x, x_size)

            if not torch.is_tensor(x):
                x, loss_moe = x
                loss_moe_all += loss_moe or 0

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x, loss_moe_all

    def forward_backbone(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
        else:
            raise Exception("not implemented yet")

        x = x / self.img_range + self.mean
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        loss_moe = 0
        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)

            nlsa_st = self.nlsa2(self.nlsa1(x))
            for layer in self.layers_hfm:
                x = layer(nlsa_st)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
            nlsa_fn = self.nlsa4(self.nlsa3(x))
            x = self.conv_before_upsample(nlsa_fn)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffle_aux":
            bicubic = F.interpolate(x, size=(H * self.upscale, W * self.upscale), mode="bicubic", align_corners=False)
            bicubic = self.conv_bicubic(bicubic)
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            aux = self.conv_aux(x)  # b, 3, LR_H, LR_W
            x = self.conv_after_aux(aux)
            x = (
                self.upsample(x)[:, :, : H * self.upscale, : W * self.upscale]
                + bicubic[:, :, : H * self.upscale, : W * self.upscale]
            )
            x = self.conv_last(x)
            aux = aux / self.img_range + self.mean
        elif self.upsampler == "pixelshuffle_hf":
            # for classical SR with HF
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
            x_before = self.conv_before_upsample(x)
            x_out = self.conv_last(self.upsample(x_before))

            x_hf = self.conv_first_hf(x_before)

            res_hf = self.forward_features_hf(x_hf)
            if not torch.is_tensor(res_hf):
                res_hf, loss_moe_hf = res_hf
                loss_moe += loss_moe_hf

            x_hf = self.conv_after_body_hf(res_hf) + x_hf
            x_hf = self.conv_before_upsample_hf(x_hf)
            x_hf = self.conv_last_hf(self.upsample_hf(x_hf))
            x = x_out + x_hf
            x_hf = x_hf / self.img_range + self.mean

        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)

            res = self.forward_features(x)
            if not torch.is_tensor(res):
                res, loss_moe = res

            x = self.conv_after_body(res) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)

            res = self.forward_features(x_first)
            if not torch.is_tensor(res):
                res, loss_moe = res

            res = self.conv_after_body(res) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        if self.upsampler == "pixelshuffle_aux":
            return x[:, :, : H * self.upscale, : W * self.upscale], aux, loss_moe

        elif self.upsampler == "pixelshuffle_hf":
            x_out = x_out / self.img_range + self.mean
            return (
                x_out[:, :, : H * self.upscale, : W * self.upscale],
                x[:, :, : H * self.upscale, : W * self.upscale],
                x_hf[:, :, : H * self.upscale, : W * self.upscale],
                loss_moe,
            )

        else:
            return x[:, :, : H * self.upscale, : W * self.upscale], loss_moe

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class SWIN2SRLightning(L.LightningModule):
    def __init__(self, config: dict, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = Swin2SR(**config)
        self.learning_rate = learning_rate
        self.wlt_loss = SWTLoss()
        self.metric_loss = piq.SSIMLoss()
        similarity_loss = config.get("similarity_loss", False)
        self.ndvi_loss = None
        ndvi_loss = config.get("ndvi_loss", False)

        if similarity_loss == "chroma":
            self.similarity_loss = ChromaSimilarityLoss().to(self.device)
        elif similarity_loss == "downsample":
            self.similarity_loss = DownsampleSimilarityLoss().to(self.device)
        else:
            self.similarity_loss = None

        if ndvi_loss:
            self.ndvi_loss = NDVILoss()

    def on_fit_start(self):
        log_dir = self.trainer.log_dir
        if self.trainer.is_global_zero:
            print(f"Logging to: {log_dir}")

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def training_step(self, batch, _):
        lr, hr = batch
        output, _ = self.model(lr)
        output = output.clamp(min=0, max=1).contiguous()

        wlt_loss = self.wlt_loss(output, hr)
        metric_loss = self.metric_loss(output, hr)
        zncc_loss = cc_loss(output, hr)

        total_loss = wlt_loss + metric_loss + zncc_loss

        if self.similarity_loss is not None:
            sim_loss = self.similarity_loss(output[:, :3, :, :], lr[:, :3, :, :])
            self.log(
                f"train/similarity_loss",
                sim_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            total_loss += 10 * sim_loss

        if self.ndvi_loss is not None:
            ndvi_loss = self.ndvi_loss(output, hr)
            self.log(
                f"train/ndvi_loss",
                ndvi_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            total_loss += ndvi_loss

        self.log(
            f"train/wlt_loss", wlt_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"train/ssim_loss",
            metric_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"train/zncc_loss",
            zncc_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"train/total_loss",
            (total_loss).item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        output, _ = self.model(lr)
        output = output.clamp(min=0, max=1).contiguous()

        wlt_loss = self.wlt_loss(output, hr)
        metric_loss = self.metric_loss(output, hr)
        zncc_loss = cc_loss(output, hr)

        self.log(
            f"val/wlt_loss", wlt_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val/ssim_loss",
            metric_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val/zncc_loss", zncc_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        total_loss = wlt_loss + metric_loss + zncc_loss
        if self.similarity_loss is not None:
            sim_loss = self.similarity_loss(output[:, :3, :, :], lr[:, :3, :, :])
            self.log(
                f"val/similarity_loss",
                sim_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            total_loss += sim_loss

        if self.ndvi_loss is not None:
            ndvi_loss = self.ndvi_loss(output, hr)
            self.log(
                f"val/ndvi_loss",
                ndvi_loss.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            total_loss += ndvi_loss

        self.log(
            f"val/total_loss",
            (total_loss).item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_idx == 0:
            lr_upscaled = torch.nn.functional.interpolate(
                lr[:4][:, :3, :, :], (output.shape[-2], output.shape[-1]), mode="bicubic"
            )
            images = torch.cat([lr_upscaled, hr[:4][:, :3, :, :], output[:4][:, :3, :, :]])
            grid = torchvision.utils.make_grid(images, nrow=min(hr.size()[0], 4))
            self.logger.experiment.add_image("val/images", grid, self.current_epoch)

            if hr.shape[1] == 4:
                images = torch.cat([hr[:4, 3:4, :, :], output[:4, 3:4, :, :]])
                grid = torchvision.utils.make_grid(images, nrow=min(hr.size()[0], 4))
                self.logger.experiment.add_image("val/images_ir", grid, self.current_epoch)

                # Compute NDVI
                hr_red = hr[:4, 0:1, :, :]
                hr_nir = hr[:4, 3:4, :, :]

                output_red = output[:4, 0:1, :, :]
                output_nir = output[:4, 3:4, :, :]

                hr_ndvi = colorize(ndvi(hr_red, hr_nir))
                output_ndvi = colorize(ndvi(output_red, output_nir))

                images = torch.cat([hr_ndvi, output_ndvi])
                grid = torchvision.utils.make_grid(images, nrow=min(hr.size()[0], 4))
                self.logger.experiment.add_image("val/images_ndvi", grid, self.current_epoch)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        return optimizer


if __name__ == "__main__":
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = Swin2SR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
    )
    # print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x[0].shape)

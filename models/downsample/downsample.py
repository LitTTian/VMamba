import torch
import torch.nn as nn
import torch.nn.functional as F

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        Linear = nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class DownsampleV1(nn.Module):
    def __init__(self, 
                 dim=96,
                 out_dim=192,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.downsample = nn.Sequential(
            Permute(0, 3, 1, 2), # to (B, C, H, W)
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1), # to (B, H, W, C)
            norm_layer(out_dim),
        )
    def forward(self, x):
        return self.downsample(x)
    
class DownsampleV3(nn.Module):
    def __init__(self, 
                 dim=96,
                 out_dim=192,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.downsample = nn.Sequential(
            Permute(0, 3, 1, 2), # to (B, C, H, W)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1), # to (B, H, W, C)
            norm_layer(out_dim),
        )
    def forward(self, x):
        return self.downsample(x)
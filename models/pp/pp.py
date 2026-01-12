import torch
import torch.nn as nn

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class PatchPartitionV1(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.patch_partition = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            norm_layer(embed_dim),
        )

    def forward(self, x):
        # x (B, C, H, W) --Conv-> (B, embed_dim, H/4, W/4)
        x = self.patch_partition(x)
        return x

class PatchPartitionV2(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, norm_layer=nn.LayerNorm,):
        super().__init__()
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        self.patch_partition = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            Permute(0, 2, 3, 1), # to (B, H/2, W/2, embed_dim // 2)
            norm_layer(embed_dim // 2),
            Permute(0, 3, 1, 2), # to (B, embed_dim // 2, H/2, W/2)
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Permute(0, 2, 3, 1), # to (B, H/4, W/4, embed_dim)
            norm_layer(embed_dim),
        )

    def forward(self, x):
        # x (B, C, H, W) --Conv-> (B, embed_dim/2, H/2, W/2) --Conv-> (B, embed_dim, H/4, W/4)
        x = self.patch_partition(x)
        return x
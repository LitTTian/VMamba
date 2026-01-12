import torch
import torch.nn as nn
from .downsample import PatchMerging2D, DownsampleV1, DownsampleV3
from .ss2d import SS2D
from .pp import PatchPartitionV1, PatchPartitionV2
from timm.models.layers import DropPath

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class PatchPartition(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, norm_layer=nn.LayerNorm, 
                 channel_first=False, pp_type='v2'):
        super().__init__()
        pp = {
            'v1': PatchPartitionV1,
            'v2': PatchPartitionV2
        }[pp_type]
        self.patch_partition = pp(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer,
        )
        self.channel_first = channel_first
    
    def forward(self, x):
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2)
        x = self.patch_partition(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, 
                 in_channels=96, 
                 out_channels=192, 
                 norm_layer=nn.LayerNorm, 
                 channel_first=False,
                 downsample_type='v1'):
        super().__init__()
        down = {
            'v1': PatchMerging2D,
            'v2': DownsampleV1,
            'v3': DownsampleV3,
        }[downsample_type]
        self.downsample = down(
            dim=in_channels,
            out_dim=out_channels,
            norm_layer=norm_layer,
        )
        self.channel_first = channel_first
    def forward(self, x):
        if self.channel_first:
            x = x.permute(0, 2, 3, 1)  # to (B, H, W, C)
        x = self.downsample(x)
        if self.channel_first:  # 还原成本来的形状
            x = x.permute(0, 3, 1, 2)  # to (B, C, H, W)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VSSBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int = 0, 
                 drop_path: float = 0, 
                 norm_layer: nn.Module = nn.LayerNorm, 
                 ssm_d_state: int = 16,
                 ssm_ratio=2.0,
                 ssm_act_layer=nn.SiLU,
                 channel_first=False,
                 mlp_ratio=4.0,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate: float = 0.0,
                 **kwargs):
        super().__init__()
        self.channel_first = channel_first
        self.norm = norm_layer(hidden_dim)
        self.drop_path = DropPath(drop_path)
        # self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.ss2d = SS2D(
            d_model=hidden_dim,  # 96
            d_state=ssm_d_state,  # 16
            ssm_ratio=ssm_ratio,  # 2.0
            dt_rank="auto",
            act_layer=ssm_act_layer,  # ssm_act_layer
        )
        self.norm2 = norm_layer(hidden_dim)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate)
        
    def forward(self, x):  # x: (64, 96, 56, 56)
        x = x + self.drop_path(self.ss2d(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x
    
class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(1, 2))

class VMambaB(nn.Module):
    def __init__(self, 
                 hidden_dim=128,
                 out_features=1000,
                 norm_layer: nn.Module = nn.LayerNorm, ):
        super().__init__()
        self.stem = PatchPartition(in_chans=3, embed_dim=hidden_dim, channel_first=True)  # 第一次图像输入(B, C, H, W)
        self.stage1 = nn.Sequential(
            *[VSSBlock(hidden_dim=hidden_dim) for _ in range(2)],
            DownsampleV3(dim=hidden_dim, out_dim=hidden_dim*2)  # Permute + Conv2d + Permute + LayerNorm
        )
        
        self.stage2 = nn.Sequential(
            *[VSSBlock(hidden_dim=hidden_dim*2) for _ in range(2)],
            DownsampleV3(dim=hidden_dim*2, out_dim=hidden_dim*4)
        )

        self.stage3 = nn.Sequential(
            *[VSSBlock(hidden_dim=hidden_dim*4) for _ in range(15)],
            DownsampleV3(dim=hidden_dim*4, out_dim=hidden_dim*8)
        )

        self.stage4 = nn.Sequential(
            *[VSSBlock(hidden_dim=hidden_dim*8) for _ in range(2)],
        )

        # 分类头
        self.classifier = nn.Sequential(
            # Permute(0, 3, 1, 2),
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            GlobalAvgPool(),
            nn.Linear(in_features=hidden_dim*8, out_features=out_features),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Input Images (B, 3, H, W) -> Patch Partition (B, C1, H/4, W/4)
        # Stage 1: VSS Block xL1 (B, C1, H/4, W/4)
        # Stage 2: Downsampling + VSS Block xL2 (B, C2, H/8, W/8)
        # Stage 3: Downsampling + VSS Block xL3 (B, C3, H/16, W/16)
        # Stage 4: Downsampling + VSS Block xL4 (B, C4, H/32, W/32)
        # Classification Head
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x
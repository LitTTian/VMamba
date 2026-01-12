import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import selective_scan_cuda
# ===
import time


def cross_scan_fwd(x: torch.Tensor): # x: (B, C, H, W) -> y: (B, K, D, H*W) (B, 4, C, H*W)
    B, C, H, W = x.shape
    # NT: SS2D默认模式，对应4条互补扫描路径
    y = x.new_empty((B, 4, C, H * W))
    y[:, 0, :, :] = x.flatten(2, 3)  # (左上) -> ↓ -> 
    y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # (左上) ↓ -> ↓ 
    y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])  # 2: <- ↑ <- (右下) 3: ↑ <- ↑ (右下)
    return y

def cross_merge_fwd(y: torch.Tensor):  # y: (B, K, D, H, W) -> x: (B, D, H, W)
    B, K, D, H, W = y.shape
    y = y.view(B, K, D, -1)
    y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)   
    return y

class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        y = cross_scan_fwd(x)
        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        ys = ys.view(B, -1, C, H, W)
        y = cross_merge_fwd(ys)
        y = y.view(B, -1, H, W)
        return y, None, None, None, None

class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        y = cross_merge_fwd(ys)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        B, C, H, W = ctx.shape
        x = x.view(B, C, H, W)
        x = cross_scan_fwd(x)
        x = x.view(B, 4, C, H, W)
        return x, None, None, None, None

# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    CSF = CrossScanF
    with torch.cuda.device(x.device):
        return CSF.apply(x)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    CMF = CrossMergeF
    with torch.cuda.device(y.device):
        return CMF.apply(y)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class SoftmaxSpatial(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return F.softmax(x, dim=self.dim)
    
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SelectiveScanCuda(torch.autograd.Function):  # HL: 加速版本
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None

def selective_scan(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus: bool = True,
    *args,
    **kwargs
):  # HL: 非加速版本
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out

def selective_scan_fn(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
):
    fn = selective_scan if False else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus)

class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        Linear = nn.Linear
        self.out_norm = nn.LayerNorm(self.d_inner)


        # in proj =======================================
        d_proj = self.d_inner
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        # self.in_proj = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )

    def forward_core(
        self,
        x: torch.Tensor=None, 
        **kwargs,
    ):
        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W
        
        # MODULE: cross scan
        xs = cross_scan_fn(x)
        
        # MODULE: projections
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        # # MODULE: selective scan
        ys: torch.Tensor = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus=True
        ).view(B, K, -1, H, W)
        
        # MODULE: cross merge
        y: torch.Tensor = cross_merge_fn(ys)

        y = y.view(B, self.d_inner, H, W).permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):  # x: (64, 96, 56, 56)
        x = self.in_proj(x)  # Linear -> (64, 192, 56, 56)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # DWConv -> (64, 192, 56, 56)
        x = self.act(x)  # SiLU
        y = self.forward_core(x)  # SS2D
        y = self.out_act(y)  # LN
        out = self.dropout(self.out_proj(y))  # Linear + Dropout -> (64, 96, 56, 56)
        return out
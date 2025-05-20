# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from models.mamba.mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj


# def randbool(size, p=0.5):
#     return torch.rand(*size) < p

def randbool(size, p=0.5):
    return torch.cuda.FloatTensor(*size).uniform_() < p
    
    
def generate_spiral_matrix(height, width):
    matrix = torch.zeros(height, width, dtype=torch.int64)
    top, bottom = 0, height - 1
    left, right = 0, width - 1
    num = height * width - 1
    while top <= bottom and left <= right:
        for i in range(right, left - 1, -1):
            matrix[bottom][i] = num
            num -= 1
        bottom -= 1
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = num
            num -= 1
        left += 1
        for i in range(left, right + 1):
            matrix[top][i] = num
            num -= 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num -= 1
        right -= 1
    return torch.abs(matrix)  # FIXME: may yield num < 0


def build_mapping(matrix1, matrix2):
    mapping = {}
    for i in range(matrix1.size(0)):
        for j in range(matrix1.size(1)):
            mapping[matrix1[i][j].item()] = matrix2[i][j].item()
    return mapping


def map_matrix(matrix, mapping):
    mapped_matrix = torch.zeros_like(matrix)
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            mapped_matrix[i][j] = mapping[matrix[i][j].item()]
    return mapped_matrix


class Trans:
    def __init__(self, l, h, w):
        self.l = l
        self.h = h
        self.w = w
        self.num_patch = self.l * self.h * self.w
        self.L = None
        self.H = None
        self.W = None
        self.frame_size = None
        self.hg = None
        self.wg = None
        self.group_size = None

        # self.flag = False
        self.split_x = None
        self.split_g = None
        self.split_l = None
        self.last_l = None
        self.last_num_patch = None

    def set(self, L, H, W):
        self.L = L
        self.H = H
        self.W = W
        assert self.H % self.h == 0 and self.W % self.w == 0
        self.frame_size = self.H * self.W
        self.hg = self.H // self.h
        self.wg = self.W // self.w
        self.group_size = self.hg * self.wg

        # self.flag = False if self.L % self.l == 0 else True
        self.split_x = (self.L // self.l) * self.l * H * W
        self.split_g = self.split_x // self.num_patch
        self.split_l = self.split_g // self.group_size * self.l
        self.last_l = self.L % self.l
        self.last_num_patch = self.last_l * self.h * self.w

    """
    eg.
    L, H, W = 3, 4, 4
    l, h, w = 2, 2, 2
            b0 b2           b1 b3
    g0:    [00 01]02 03    [16 17]18 19     32 33 34 35
            b4 b5           b6 b7            
           [04 05]06 07    [20 21]22 23     36 37 38 39
              
            08 09 10 11     24 25 26 27     40 41 42 43
            
            12 13 14 15     28 29 30 31     44 45 46 47
             
                  b0 b2           b1 b3
    g1:     00 01[02 03]    16 17[18 19]    32 33 34 35
                  b4 b5           b6 b7
            04 05[06 07]    20 21[22 23]    36 37 38 39
        
            08 09 10 11     24 25 26 27     40 41 42 43
        
            12 13 14 15     28 29 30 31     44 45 46 47
            
            
    g2:     00 01 02 03     16 17 18 19     32 33 34 35
             
            04 05 06 07     20 21 22 23     36 37 38 39
            b0 b2           b1 b3
           [08 09]10 11    [24 25]26 27     40 41 42 43
            b4 b5           b6 b7            
           [12 13]14 15    [28 29]30 31     44 45 46 47
            
    ...
    
    g7:     00 01 02 03     16 17 18 19     32 33 34 35
             
            04 05 06 07     20 21 22 23     36 37 38 39
                                                  b0 b1
            08 09 10 11     24 25 26 27     40 41[42 43]
                                                  b2 b3
            12 13 14 15     28 29 30 31     44 45[46 47]
    """
    
    def f_(self, x):
        if (x < self.split_x):
            gi = x // self.num_patch
            li = gi // self.group_size * self.l
            hi = gi % self.group_size // self.hg * self.h
            wi = gi % self.group_size % self.wg * self.w
            bi = x % self.num_patch
            li = li + bi % self.l
            hi = hi + bi // (self.w * self.l)
            wi = wi + (bi // self.l) % self.w
        else:
            gi = self.split_g + (x - self.split_x) // self.last_num_patch
            li = self.split_l
            hi = gi % self.group_size // self.hg * self.h
            wi = gi % self.group_size % self.wg * self.w
            bi = (x - self.split_x) % self.last_num_patch
            li = li + bi % self.last_l
            hi = hi + bi // (self.w * self.last_l)
            wi = wi + (bi // self.last_l) % self.w
        # print(gi, bi)
        return li * (self.frame_size) + hi * self.W + wi

    def g_(self, x):
        li = x // self.frame_size
        x = x % self.frame_size
        hi = x // self.W
        wi = x % self.W
        
        gi = (li // self.l) * self.group_size + (hi // self.h) * self.wg + (wi // self.w)
        if (li < self.split_l):
            bi = ((hi % self.h) * self.w + (wi % self.w)) * self.l + li % self.l
            return gi * self.num_patch + bi 
        else:
            bi = ((hi % self.h) * self.w + (wi % self.w)) * self.last_l + li % self.last_l
            return self.split_x + (gi - self.split_g) * self.last_num_patch + bi
        
    def _test(self, x):
        assert self.g_(self.f_(x)) == x


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        md_type=None,
        drop_prob=None,
        l=None,
        h=None,
        w=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        
        # b
        if md_type == 'b':  # bi-direction
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        
            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
        # c
        elif md_type == 'c':  # LHW,LWH,HWL order
            # c-1
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            # c-2
            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
            A_c = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_c_log = torch.log(A_c)  # Keep A_c_log in fp32
            self.A_c_log = nn.Parameter(A_c_log)
            self.A_c_log._no_weight_decay = True 

            self.conv1d_c = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_c = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_c._no_weight_decay = True
        
        elif md_type == 'd':  # center-to-boarder
            # >>>  FIXME: fixed h = 65/62, 33/31, 17/16, 9/8
            sizes = [(65, 62), (33, 31), (17, 16), (9, 8), (3, 3)]
            self.rev_trans = {s[0]: generate_spiral_matrix(s[0], s[1]) for s in sizes}
            ori_order = {s[0]: torch.arange(0, s[0] * s[1]).view(s[0], s[1]) for s in sizes}
            mapping = {s[0]: build_mapping(self.rev_trans[s[0]], ori_order[s[0]]) for s in sizes}
            self.trans = {s[0]: map_matrix(ori_order[s[0]], mapping[s[0]]) for s in sizes}
            # <<< 
            
            A_d = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_d_log = torch.log(A_d)  # Keep A_b_log in fp32
            self.A_d_log = nn.Parameter(A_d_log)
            self.A_d_log._no_weight_decay = True 

            self.conv1d_d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        
            self.x_proj_d = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_d = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_d = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_d._no_weight_decay = True
        
        elif md_type == 'e':  # abstract - concrete
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
        elif md_type == 'f':  # abstract - concrete
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
        
        
            A_c = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_c_log = torch.log(A_c) 
            self.A_c_log = nn.Parameter(A_c_log)
            self.A_c_log._no_weight_decay = True 

            self.conv1d_c = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_c = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_c._no_weight_decay = True
            
            self.trans = Trans(l, h, w)
            
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_model)
        
        
        self.md_type = md_type
        self.drop_prob = drop_prob

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D) / (B, L, C, H, W)
        Returns: same shape as hidden_states
        """
        if self.md_type == 'e':
            B, L, C, H, W = hidden_states.shape
            hidden_states = hidden_states.permute(0, 1, 3, 4, 2).reshape(B, L * H * W, C)
            
        elif self.md_type == 'f':
            B, L, C, H, W = hidden_states.shape
            hidden_states = hidden_states.permute(0, 1, 3, 4, 2).reshape(B, L * H * W, C)
            self.trans.set(L, H, W)
            t = torch.arange(0, L * H * W)
            mt = torch.arange(0, L * H * W).apply_(lambda x: self.trans.f_(x))
            mrt = torch.arange(0, L * H * W).apply_(lambda x: self.trans.g_(x))
            
        hidden_states = self.norm(hidden_states)
            
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")  # B, C, LHW
        
        B, Ci, _ = xz.shape

        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            # origin
            if self.md_type is None:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                
            elif self.md_type == 'b':
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                
            elif self.md_type == 'c':
                # in_a = xz.view(B, Ci, L, H * W)
                # start = torch.argmax(torch.mean(in_a, dim=1, keepdim=True), dim=3, keepdim=True)  # B, 1, L, 1
                
                # in_a = in_a.view(B * Ci * L, H * W)
                # shifts = start.repeat(1, Ci, 1, 1)  # B, C, L, 1
                # shifts = shifts.view(B * Ci * L)
                # indices_a = (torch.arange(in_a.shape[1], device=in_a.device)[None, :] + shifts[:, None]) % in_a.shape[1]
                
                # in_a = torch.gather(in_a, 1, indices_a)
                # in_a = in_a.view(B, Ci, L * H * W)
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                # _, Co, _ = out.shape
                # out = out.view(B * Co * L, H * W)
        
                # out_shifts = start.repeat(1, Co, 1, 1)  # B, C, L, 1
                # out_shifts = out_shifts.view(B * Co * L)
                # indices_a = (torch.arange(out.shape[1], device=out.device)[None, :] - out_shifts[:, None]) % out.shape[1]

                # out = torch.gather(out, 1, indices_a)
                # out = out.view(B, Co, L * H * W)
            elif self.md_type == 'd':
                in_a = xz.view(xz.shape[0], xz.shape[1], L, H * W)
                in_a = in_a[:, :, :, self.trans[H]].view(in_a.shape[0], in_a.shape[1], L * H * W)
                
                out = mamba_inner_fn_no_out_proj(
                    in_a, 
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj_d.bias.float(),
                    delta_softplus=True,
                )
                
                out = out.view(out.shape[0], out.shape[1], L, H * W)
                out = out[:, :, :, self.rev_trans[H]].view(out.shape[0], out.shape[1], L * H * W)
    
            elif self.md_type == 'e':
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
            elif self.md_type == 'f':
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

            else:
                raise NotImplementedError
                
            # other
            if self.md_type == 'b':
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]), 
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            
            elif self.md_type == 'c':
            
                # in_b = xz.view(B, Ci, L, H, W).permute(0, 1, 2, 4, 3).reshape(B * Ci * L, W * H)  # L W H
                # shifts = (shifts // W) + (shifts % W) * H
                # indices_b = (torch.arange(in_b.shape[1], device=in_b.device)[None, :] + shifts[:, None]) % in_b.shape[1]
                    
                # in_b = torch.gather(in_b, 1, indices_b)
                # in_b = in_b.view(B, Ci, L * W * H)
                
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.view(xz.shape[0], xz.shape[1], L, H, W).permute(0, 1, 2, 4, 3).reshape(xz.shape[0], xz.shape[1], L * H * W),  
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                
                # out_b = out_b.view(B * Co * L, W * H)
                # indices_b = (torch.arange(out_b.shape[1], device=out_b.device)[None, :] - out_shifts[:, None]) % out_b.shape[1]
    
                # out_b = torch.gather(out_b, 1, indices_b)
                # out_b = out_b.view(B, Co, L * W * H)
                
                A_c = -torch.exp(self.A_c_log.float())
                out_c = mamba_inner_fn_no_out_proj(
                    xz.view(xz.shape[0], xz.shape[1], L, H, W).permute(0, 1, 3, 4, 2).reshape(xz.shape[0], xz.shape[1], L * H * W),  # H W L
                    self.conv1d_c.weight,
                    self.conv1d_c.bias,
                    self.x_proj_c.weight,
                    self.dt_proj_c.weight,
                    A_c,
                    None,
                    None,
                    self.D_c.float(),
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                )
                
                # out = F.linear(rearrange(out + \
                #                          out_c.view(out_c.shape[0], out_c.shape[1], H, W, L).permute(0, 1, 4, 2, 3).reshape(out_c.shape[0], out_c.shape[1], L * H * W),
                #                          "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                out = F.linear(rearrange(out + \
                                         out_b.view(out_b.shape[0], out_b.shape[1], L, W, H).permute(0, 1, 2, 4, 3).reshape(out_b.shape[0], out_b.shape[1], L * H * W) + \
                                         out_c.view(out_c.shape[0], out_c.shape[1], H, W, L).permute(0, 1, 4, 2, 3).reshape(out_c.shape[0], out_c.shape[1], L * H * W),
                                         "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            
            elif self.md_type == 'd':
                A_d = -torch.exp(self.A_d_log.float())
                out_d = mamba_inner_fn_no_out_proj(
                    xz.view(xz.shape[0], xz.shape[1], L, H, W).permute(0, 1, 3, 4, 2).reshape(xz.shape[0], xz.shape[1], L * H * W),  # H W L
                    self.conv1d_d.weight,
                    self.conv1d_d.bias,
                    self.x_proj_d.weight,
                    self.dt_proj_d.weight,
                    A_d,
                    None,
                    None,
                    self.D_d.float(),
                    delta_bias=self.dt_proj_d.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + \
                                         out_d.view(out_d.shape[0], out_d.shape[1], H, W, L).permute(0, 1, 4, 2, 3).reshape(out_d.shape[0], out_d.shape[1], L * H * W),
                                         "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                
            elif self.md_type == 'e':
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.view(xz.shape[0], xz.shape[1], L, H, W).permute(0, 1, 2, 4, 3).reshape(xz.shape[0], xz.shape[1], L * H * W),  
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + \
                                         out_b.view(out_b.shape[0], out_b.shape[1], L, W, H).permute(0, 1, 2, 4, 3).reshape(out_b.shape[0], out_b.shape[1], L * H * W),
                                         "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                out = out.view(B, L, H, W, C).permute(0, 1, 4, 2, 3).view(B, L, C, H, W)
                
            elif self.md_type == 'f':
                A_b = -torch.exp(self.A_b_log.float())
                out_b = mamba_inner_fn_no_out_proj(
                    xz.view(xz.shape[0], xz.shape[1], L, H, W).permute(0, 1, 2, 4, 3).reshape(xz.shape[0], xz.shape[1], L * H * W),  
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                
                A_c = -torch.exp(self.A_c_log.float())
                out_c = mamba_inner_fn_no_out_proj(
                    xz[:, :, mt],  
                    self.conv1d_c.weight,
                    self.conv1d_c.bias,
                    self.x_proj_c.weight,
                    self.dt_proj_c.weight,
                    A_c,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + \
                                         out_b.view(out_b.shape[0], out_b.shape[1], L, W, H).permute(0, 1, 2, 4, 3).reshape(out_b.shape[0], out_b.shape[1], L * H * W) + \
                                         out_c[:, :, mrt],
                                         "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                out = out.view(B, L, H, W, C).permute(0, 1, 4, 2, 3).view(B, L, C, H, W)

                
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

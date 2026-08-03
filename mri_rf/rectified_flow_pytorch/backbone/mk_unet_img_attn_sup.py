import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import os
from functools import partial

import timm
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

__all__ = ['MKUNet']

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes == None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activation = act_layer(activation, inplace=True)

        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
                        
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out= self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class GroupedAttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int, kernel_size=1, groups=1, activation='relu'):
        super(GroupedAttentionGate,self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size,stride=1,padding=kernel_size//2,groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size,stride=1,padding=kernel_size//2,groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1+x1)
        psi = self.psi(psi)

        return x*psi

class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs

class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )        
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation, dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c*1
        else:
            self.combined_channels = self.ex_c*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False), # 
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False) 
        
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x+out
        else:
            return out

def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6'):
    """
    create a series of multi-kernel inverted residual blocks.
    """
    convs = []
    xx = MultiKernelInvertedResidualBlock(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation)
    convs.append(xx)
    if n>1:
        for i in range(1,n):
            xx = MultiKernelInvertedResidualBlock(out_c, out_c, 1, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv

# channels = [4,8,16,24,32] for MK_UNet-T
# channels = [8,16,32,48,80] for MK_UNet-S
# channels = [16,32,64,96,160] for MK_UNet
# channels = [32,64,128,192,320] for MK_UNet-M
# channels = [64,128,256,384,512] for MK_UNet-L

#################################################################################################



        
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------
# Time Embedding
# --------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=device) * -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        emb = t * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------
# Scale-Shift
# --------------------------
def apply_scale_shift(x, t_emb):
    B, C, H, W = x.shape
    t = t_emb.unsqueeze(-1).unsqueeze(-1)      # (B,2C,1,1)
    scale, shift = t.chunk(2, dim=1)
    return x * (scale + 1) + shift


# =====================================================================
#                       MKRF_UNet (Stage-wise Time Modulation)
# =====================================================================
class MKRF_UNet(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        channels=[64, 128, 256, 384, 512],
        depths=[1,1,1,1,1],
        kernel_sizes=[1,3,5],
        expansion_factor=2,
        gag_kernel=3,
        time_embed_dim=256,
        deep_supervision=True,
        regis=False,
        model = None
    ):
        super().__init__()

        C0, C1, C2, C3, C4 = channels

        # -----------------------
        # Time embedding
        # -----------------------
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)

        # Stage-wise MLPs
        self.time_mlp = nn.ModuleDict({
            "enc5": nn.Sequential(nn.Linear(time_embed_dim, C4*2), nn.GELU(), nn.Linear(C4*2, C4*2)),
            "dec4": nn.Sequential(nn.Linear(time_embed_dim, C3*2), nn.GELU(), nn.Linear(C3*2, C3*2)),
            "dec3": nn.Sequential(nn.Linear(time_embed_dim, C2*2), nn.GELU(), nn.Linear(C2*2, C2*2)),
            "dec2": nn.Sequential(nn.Linear(time_embed_dim, C1*2), nn.GELU(), nn.Linear(C1*2, C1*2)),
            "dec1": nn.Sequential(nn.Linear(time_embed_dim, C0*2), nn.GELU(), nn.Linear(C0*2, C0*2)),
            "final":nn.Sequential(nn.Linear(time_embed_dim, C0*2), nn.GELU(), nn.Linear(C0*2, C0*2)),
        })
        self.deep_supervision = deep_supervision

        MIND_CH = 9   # MIND(non_local_region_size=3) → 3*3=9 channels

        cond_num = 0
        self.cond_num = cond_num
        self.model_type = model
        if model == 2:
            self.cond_num = 1

        # -----------------------
        # Encoder
        # -----------------------
        # image feature encoder: x (in_channels) + seg (1 ch)  — CT gaussian / cond path
        self.encoder1_img = mk_irb_bottleneck(
            in_channels + 1, C0, depths[0], 1,
            expansion_factor, True, True, kernel_sizes
        )

        # flow / deformation encoder — unconditional path
        self.encoder1_flow = mk_irb_bottleneck(
            in_channels + self.cond_num, C0, depths[0], 1,
            expansion_factor, True, True, kernel_sizes
        )

        # model=3: MIND feature condition encoder — x (in_channels) + MIND (9 ch)
        if model == 3:
            self.encoder1_cond = mk_irb_bottleneck(
                in_channels + MIND_CH, C0, depths[0], 1,
                expansion_factor, True, True, kernel_sizes
            )

        self.encoder2 = mk_irb_bottleneck(C0, C1, depths[1], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(C1, C2, depths[2], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(C2, C3, depths[3], 1, expansion_factor, True, True, kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(C3, C4, depths[4], 1, expansion_factor, True, True, kernel_sizes)

        # -----------------------
        # Attention Gates
        # -----------------------
        self.AG1 = GroupedAttentionGate(C3, C3, C3//2, gag_kernel, C3//2)
        self.AG2 = GroupedAttentionGate(C2, C2, C2//2, gag_kernel, C2//2)
        self.AG3 = GroupedAttentionGate(C1, C1, C1//2, gag_kernel, C1//2)
        self.AG4 = GroupedAttentionGate(C0, C0, C0//2, gag_kernel, C0//2)

        # -----------------------
        # Decoder
        # -----------------------
        self.decoder1 = mk_irb_bottleneck(C4, C3, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(C3, C2, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(C2, C1, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(C1, C0, 1, 1, expansion_factor, True, True, kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(C0, C0, 1, 1, expansion_factor, True, True, kernel_sizes)

        self.CA1 = ChannelAttention(512)  # decoder1 output
        self.SA1 = SpatialAttention()

        self.CA2 = ChannelAttention(384)   # decoder2 output
        self.SA2 = SpatialAttention()

        self.CA3 = ChannelAttention(256)   # decoder3 output
        self.SA3 = SpatialAttention()

        self.CA4 = ChannelAttention(128)   # decoder4 output
        self.SA4 = SpatialAttention()

        self.CA5 = ChannelAttention(64)   # decoder5 output (final)
        self.SA5 = SpatialAttention()

        self.ds1 = nn.Conv2d(C2, out_channels, 1)
        self.ds2 = nn.Conv2d(C1, out_channels, 1)
        self.ds3 = nn.Conv2d(C0, out_channels, 1)

        if regis is True:
            self.final = nn.Conv2d(C0, 2, 1)
        else:
            self.final = nn.Conv2d(C0, out_channels, 1)
        # -----------------------

        self.in_channels = in_channels
        

    # =====================================================================
    #                              Forward
    # =====================================================================
    def forward(self, x, times=None, cond=None, seg=None):

        
        if times.dim() == 1:
            times = times.unsqueeze(1)


        t = self.time_pos_emb(times)     # (B, 256)

            
        # -------------------
        # Encoder
        # t_k: Skip connection 용도
        # -------------------
        if self.model_type == 3 and seg is not None:
            # model=3: MIND feature condition — x + MIND(9ch) → encoder1_cond
            out = self.encoder1_cond(torch.cat([x, seg], dim=1))
        elif seg is not None:
            # CT gaussian: x + seg(NECT 1ch) → encoder1_img
            out = self.encoder1_img(torch.cat([x, seg], dim=1))
        elif cond is not None:
            out = self.encoder1_img(torch.cat([x, cond], dim=1))
        else:
            out = self.encoder1_flow(x)
            
        out = F.max_pool2d(out, 2); t1 = out
        out = F.max_pool2d(self.encoder2(out), 2); t2 = out
        out = F.max_pool2d(self.encoder3(out), 2); t3 = out
        out = F.max_pool2d(self.encoder4(out), 2); t4 = out

        out = F.max_pool2d(self.encoder5(out), 2)
        # time embedding scale shift
        out = apply_scale_shift(out, self.time_mlp["enc5"](t))  # C=160

        # -------------------
        # Decoder
        # -------------------
        preds = []   # deep supervision outputs

        # =========================
        # Stage 4
        # =========================
        out = self.CA1(out) * out
        out = self.SA1(out) * out
        out = F.relu(F.interpolate(
            self.decoder1(out),
            size=t4.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        out = apply_scale_shift(out, self.time_mlp["dec4"](t))
        out = out + self.AG1(out, t4)

        # =========================
        # Stage 3  (DS-1)
        # =========================
        out = self.CA2(out) * out
        out = self.SA2(out) * out
        out = F.relu(F.interpolate(
            self.decoder2(out),
            size=t3.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        out = apply_scale_shift(out, self.time_mlp["dec3"](t))

        # 🔹 deep supervision head (coarse)
        if self.training and self.deep_supervision:
            p1 = self.ds1(out)
            p1 = F.interpolate(p1, size=x.shape[-2:], mode='bilinear', align_corners=False)
            preds.append(p1)

        out = out + self.AG2(out, t3)

        # =========================
        # Stage 2  (DS-2)
        # =========================
        out = self.CA3(out) * out
        out = self.SA3(out) * out
        out = F.relu(F.interpolate(
            self.decoder3(out),
            size=t2.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        out = apply_scale_shift(out, self.time_mlp["dec2"](t))

        # 🔹 deep supervision head (mid)
        if self.training and self.deep_supervision:
            p2 = self.ds2(out)
            p2 = F.interpolate(p2, size=x.shape[-2:], mode='bilinear', align_corners=False)
            preds.append(p2)

        out = out + self.AG3(out, t2)

        # =========================
        # Stage 1  (DS-3)
        # =========================
        out = self.CA4(out) * out
        out = self.SA4(out) * out
        out = F.relu(F.interpolate(
            self.decoder4(out),
            size=t1.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        out = apply_scale_shift(out, self.time_mlp["dec1"](t))

        # 🔹 deep supervision head (fine)
        if self.training and self.deep_supervision:
            p3 = self.ds3(out)
            p3 = F.interpolate(p3, size=x.shape[-2:], mode='bilinear', align_corners=False)
            preds.append(p3)

        out = out + self.AG4(out, t1)

        # =========================
        # Final
        # =========================
        out = self.CA5(out) * out
        out = self.SA5(out) * out
        out = F.relu(F.interpolate(
            self.decoder5(out),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        ))
        out = apply_scale_shift(out, self.time_mlp["final"](t))

        flow_out = self.final(out)

        # =========================
        # Return
        # =========================
        if self.training and self.deep_supervision:
            preds.append(flow_out)
            return preds[::-1]   # [final, ds3, ds2, ds1]

        return flow_out
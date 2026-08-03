from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

# 3D 전용 설정
norm_layer = partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
up_sample_mode = 'trilinear' # 3D용 보간
resnet_n_blocks = 1

def get_init_function(activation, init_function, **kwargs):
    a = 0.2 if activation == 'leaky_relu' else 0.0
    if init_function == 'kaiming':
        return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity='leaky_relu' if activation == 'leaky_relu' else 'relu')
    elif init_function == 'zeros':
        return partial(torch.nn.init.constant_, val=0.0)
    return partial(torch.nn.init.normal_, mean=0.0, std=0.02)

def get_activation(activation, **kwargs):
    if activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=False)
    elif activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    return None

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='leaky_relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(Conv, self).__init__()
        # nn.Conv2d -> nn.Conv3d
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)

        init_ = get_init_function(activation, init_func)
        init_(self.conv3d.weight)
        if self.conv3d.bias is not None:
            nn.init.constant_(self.conv3d.bias, 0.0)

    def forward(self, x):
        x = self.conv3d(x)
        if self.norm is not None: x = self.norm(x)
        if self.activation is not None: x = self.activation(x)
        if self.resnet_block is not None: x = self.resnet_block(x)
        return x

class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='leaky_relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, skip=True, pool=True, pool_size=2, **kwargs):
        super(DownBlock, self).__init__()
        self.conv_0 = Conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                           activation=activation, init_func=init_func, use_norm=use_norm,
                           use_resnet=use_resnet, **kwargs)
        self.skip = skip
        self.pool = nn.MaxPool3d(kernel_size=(1, pool_size, pool_size)) if pool else None

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.pool is not None:
            x = self.pool(x)
        return (x, skip) if self.skip else x

class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        model = [ResnetBlock(dim, norm_layer, use_bias=True) for _ in range(n_blocks)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                init_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        # 3D 전용 패딩 및 컨볼루션
        self.conv_block = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)
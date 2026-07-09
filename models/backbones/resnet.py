import torch.nn as nn
import torch.nn.functional as F
from models.arch_util import initialize_weights
from models.backbones.doconv_pytorch import *

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

# 原文的ResidualBlock_noBN
class ResidualBlock_noBN(nn.Module): #大小不变,尺寸不变
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out


#------------------------------添加的卷积块和带有傅里叶变换的残差块-------------------------------------

class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        # if bias and norm:
        #     bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
            # layers.append(
            #     nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
            #               bias=bias))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):

        return self.main(x)

class ResnetBlock_fft(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_fft, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.main_fft = nn.Sequential(
            BasicConv_do(dim * 2, dim * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(dim * 2, dim * 2, kernel_size=1, stride=1, relu=False)
        )  # fft网络
        self.norm = 'backward'

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)  #经历过傅里叶变换
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)   #1*1Conv  Relu  1*1Conv
        y_real, y_imag = torch.chunk(y, 2, dim=dim)  #在给定维度(轴)上将输入张量进行分块儿,2是分块的个数
        y = torch.complex(y_real, y_imag)  #组成一个复数,y_real实部  y_imag虚部
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)  #通过反傅里叶变换
        out = self.conv_block(x) + x + y
        return out

class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel=64):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(   #3*3Conv Relu 3*3Conv
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True,bias=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False,bias=True)
        )   #传统的残差网络
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )  #fft网络
        # self.dim = out_channel
        self.norm = 'backward'
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)  #经历过傅里叶变换
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)   #1*1Conv  Relu  1*1Conv
        y_real, y_imag = torch.chunk(y, 2, dim=dim)  #在给定维度(轴)上将输入张量进行分块儿,2是分块的个数
        y = torch.complex(y_real, y_imag)  #组成一个复数,y_real实部  y_imag虚部
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)  #通过反傅里叶变换
        return self.main(x) + x + y   #self.main(x)是残差 Y是fft x是原图
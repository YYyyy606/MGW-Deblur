""" Parts of the U-Net model """

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.doconv_pytorch import DOConv2d

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(      #512,512, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct an Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) --previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)  # 下采样,尺寸变小
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        waveres_down = Wavelet_ResBlock(inner_nc, inner_nc)
        waveres_up = Wavelet_ResBlock(inner_nc * 2, inner_nc * 2)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        #原来的
        # if outermost:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     up = [uprelu, upconv, nn.Tanh()]
        #     down = [downconv]  # down = [downconv]
        #     self.down = nn.Sequential(*down)
        #     self.submodule = submodule
        #     self.up = nn.Sequential(*up)
        # elif innermost:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     down = [downrelu, downconv]  # down = [downrelu, downconv]
        #     up = [uprelu, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
        #     self.down = nn.Sequential(*down)
        #     self.up = nn.Sequential(*up)
        # else:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)  # 上采样,尺寸变大
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     down = [downrelu, downconv, downnorm]  # down = [downrelu, downconv, downnorm]
        #     up = [uprelu, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
        #     if use_dropout:
        #         up += [nn.Dropout(0.5)]
        #
        #     self.down = nn.Sequential(*down)
        #     self.submodule = submodule
        #     self.up = nn.Sequential(*up)

        #kernel_size=4; 仅下采样小波
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            up = [uprelu, upconv, nn.Tanh()]
            down = [downconv, waveres_down]  # down = [downconv]
            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1,
                                        bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, waveres_down]  # down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1,
                                        bias=use_bias)  # 上采样,尺寸变大
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, waveres_down, downnorm]  # down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)

        #kernel_size=3；上下采样插入小波残差块
        # if outermost:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1)
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     up = [uprelu, waveres_up, upconv, nn.Tanh()]
        #     down = [downconv, waveres_down]  # down = [downconv]
        #     self.down = nn.Sequential(*down)
        #     self.submodule = submodule
        #     self.up = nn.Sequential(*up)
        # elif innermost:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     down = [downrelu, downconv, waveres_down]  # down = [downrelu, downconv]
        #     up = [uprelu, waveres_up, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
        #     self.down = nn.Sequential(*down)
        #     self.up = nn.Sequential(*up)
        # else:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)  # 上采样,尺寸变大
        #     # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #     # upconv = DoubleConv(inner_nc * 2, outer_nc)
        #     down = [downrelu, downconv, waveres_down, downnorm]  # down = [downrelu, downconv, downnorm]
        #     up = [uprelu,waveres_up, upconv, upnorm]  # up = [uprelu, upconv, upnorm]
        #     if use_dropout:
        #         up += [nn.Dropout(0.5)]
        #
        #     self.down = nn.Sequential(*down)
        #     self.submodule = submodule
        #     self.up = nn.Sequential(*up)


    def forward(self, x, noise):

        if self.outermost:
            return self.up(self.submodule(self.down(x), noise))
        elif self.innermost:  # add skip connections
            if noise is None:
                noise = torch.randn((1, 512, 8, 8)).cuda() * 0.0007
            return torch.cat((self.up(torch.cat((self.down(x), noise), dim=1)), x), dim=1)
        else:
            return torch.cat((self.up(self.submodule(self.down(x), noise)), x), dim=1)



'''
(adapter): KernelAdapter(
    (model): UnetSkipConnectionBlock(
      (down): Sequential(
        (0): Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (submodule): UnetSkipConnectionBlock(
        (down): Sequential(
          (0): LeakyReLU(negative_slope=0.2, inplace=True)
          (1): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (2): Identity()
        )
        (submodule): UnetSkipConnectionBlock(
          (down): Sequential(
            (0): LeakyReLU(negative_slope=0.2, inplace=True)
            (1): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (2): Identity()
          )
          (submodule): UnetSkipConnectionBlock(
            (down): Sequential(
              (0): LeakyReLU(negative_slope=0.2, inplace=True)
              (1): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
              (2): Identity()
            )
            (submodule): UnetSkipConnectionBlock(
              (down): Sequential(
                (0): LeakyReLU(negative_slope=0.2, inplace=True)
                (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
              )
              (up): Sequential(
                (0): ReLU(inplace=True)
                (1): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                (2): Identity()
              )
            )
            (up): Sequential(
              (0): ReLU(inplace=True)
              (1): ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
              (2): Identity()
            )
          )
          (up): Sequential(
            (0): ReLU(inplace=True)
            (1): ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (2): Identity()
          )
        )
        (up): Sequential(
          (0): ReLU(inplace=True)
          (1): ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (2): Identity()
        )
      )
      (up): Sequential(
        (0): ReLU(inplace=True)
        (1): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): Tanh()
      )
    )
  )

'''




#----------------------------小波变换残差块-------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2  # 填充1/2卷积核大小
        layers = list()
        if transpose:  # 如果转置的话，就执行下面的
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:  # 根据默认设置，执行下面句子
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:  # 如果进行归一化，就执行下面的
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)
        # main的容器中是如下两个操作
        # nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        # nn.ReLU(inplace=True)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):  # 残差网络
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(  # 两个卷积层
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 最后不能有relu，残差网络
        )

    def forward(self, x):
        return self.main(x) + x  # +x，残差块，不会出现梯度消失的现象


'''
网络中加入小波变换
'''


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2  # 4,3,128,256
    x02 = x[:, :, 1::2, :] / 2  # 4,3,128,256
    x1 = x01[:, :, :, 0::2]  # 4,3,128,128
    x2 = x02[:, :, :, 0::2]  # 4,3,128,128
    x3 = x01[:, :, :, 1::2]  # 4,3,128,128
    x4 = x02[:, :, :, 1::2]  # 4,3,128,128
    x_LL = x1 + x2 + x3 + x4  # 4,3,128,128
    x_HL = -x1 - x2 + x3 + x4  # 4,3,128,128
    x_LH = -x1 + x2 - x3 + x4  # 4,3,128,128
    x_HH = x1 - x2 - x3 + x4  # 4,3,128,128

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


'''
小波变换残差块，将小波变换和小波逆变换加入到残差块中，进行高频特征的提取
'''


class Wavelet_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Wavelet_ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True, relu_method=nn.LeakyReLU),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.DWT = DWT()
        self.IWT = IWT()

        self.Conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, relu_method=nn.LeakyReLU),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )


    def forward(self, x):  # 4,32,256,256
        res2 = self.DWT(x)  # 16,32,128,128
        res2 = self.Conv(res2)  # 16,32,128,128
        res2 = self.IWT(res2)  # 4,32,256,256
        return self.main(x) + res2 + x




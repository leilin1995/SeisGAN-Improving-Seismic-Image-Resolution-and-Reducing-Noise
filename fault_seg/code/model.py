"""
__author__ = 'linlei'
__project__:model
__time__:2021/9/28 10:33
__email__:"919711601@qq.com"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLu) * 2"""
    def __init__(self,in_ch,out_ch):
        """
        (conv -> BN -> ReLu) * 2
        Args:
            in_ch: the channels of input feature maps
            out_ch: the channels of output feature maps
        """
        super().__init__()
        # use or not use bias
        self.conv = nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size = 3,padding = 1,bias = False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(out_ch,out_ch,kernel_size = 3,padding = 1,bias = False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace = True)
                                  )

    def forward(self,x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        """
        Input layer block
        Args:
            in_ch: input channels of feature map
            out_ch: output channels of feature map
        """
        super().__init__()
        self.conv = DoubleConv(in_ch,out_ch)

    def forward(self,x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """maxpool -> double conv -> attention"""
    def __init__(self,in_ch,out_ch):
        """
        maxpool -> double conv
        Args:
            in_ch: input channels of feature map
            out_ch: output channels of feature map
        """
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(kernel_size = 2,stride = 2),
                                    DoubleConv(in_ch,out_ch))

    def forward(self,x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    """upsample or convtranspose -> double conv
            args:
                upsample:上采样 or 反卷积"""

    def __init__(self,in_ch,out_ch,upsample = True):
        """
        upsample or convtranspose -> double conv
        Args:
            in_ch: input channels of feature map
            out_ch: output channels of feature map
            upsample: Upsample or ConvTranspose2d
        """
        super().__init__()
        if upsample:
            self.up = nn.Upsample(scale_factor = 2,mode = "bilinear",align_corners = True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch,out_ch)

    def forward(self,x1,x2):
        """x1 for upsample,x2 come from backbone
                    we need  upsample x1 and then cat with x2
                """
        x1 = self.up(x1)
        # if size not equal , pading image
        if x1.size()[2:] != x2.size()[2:]:
            diffy = x2.size()[2] - x1.size()[2]
            diffx = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffx // 2, diffx - diffx // 2,
                            diffy // 2, diffy - diffy // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        """
        Output layer block
        Args:
            in_ch: input channels of feature map
            out_ch: output channels of feature map
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size = 1)

    def forward(self,x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self,in_ch = 1,out_ch = 1,init_weights = True):
        """
        U-net
        Args:
            in_ch: input channels of feature map
            out_ch: output channels of feature map
            init_weights: init weight or not
        """
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch

        self.inc = InConv(in_ch = self.in_channels,out_ch = 64) # [in_ch,128,128] -> [64,128,128]
        self.down1 = Down(in_ch = 64,out_ch = 128)  # [64,128,128] -> [128,64,64]
        self.down2 = Down(in_ch = 128,out_ch = 256) # [128,64,64] -> [256,32,32]
        self.down3 = Down(in_ch = 256,out_ch = 512) # [256,32,32] -> [512,16,16]
        self.down4 = Down(in_ch = 512,out_ch = 512) # [512,16,16] -> [512,8,8]

        self.up1 = Up(in_ch = 1024,out_ch = 256)    # [1024,16,16] -> [256,16,16]
        self.up2 = Up(in_ch = 512,out_ch = 128) # [512,32,32] -> [128,32,32]
        self.up3 = Up(in_ch = 256,out_ch = 64)  # [256,64,64] -> [64,64,64]
        self.up4 = Up(in_ch = 128,out_ch = 32)  # [128,128,128] -> [64,128,128]

        self.out = OutConv(in_ch = 32,out_ch = out_ch)  # [64,128,128] -> [out_ch,128,128]
        self.sig = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)

        x = self.out(x)
        x = self.sig(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = "fan_out",nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    a = torch.rand(10,8,64,64)
    net = Unet(in_ch = 8,out_ch = 8)
    out = net(a)
    print(out.size())
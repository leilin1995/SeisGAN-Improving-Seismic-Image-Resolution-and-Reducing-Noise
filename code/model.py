import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self,in_channels = 1,out_channels = 1,scale_factor = 2):
        """
        Network structure of generator in GAN
        Args:
            in_channels: Number of channels for input images
            out_channels: Number of channels for output images
            scale_factor: Magnification of the image
        """
        super().__init__()
        upsample_block_num = int(math.log(scale_factor , 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = 64,kernel_size = 9,padding = 4,stride = 1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1,stride = 1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(in_channels = 64,up_scale = 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(in_channels = 64,out_channels = out_channels,kernel_size = 9,padding = 4))
        self.block8 = nn.Sequential(*block8)

    def forward(self,x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self,in_channels = 1):
        """
        Patch GAN
        Args:
            in_channels: Number of channels for input images
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = 64,kernel_size = 3,padding = 1),
            nn.LeakyReLU(0.2),
        )

        channels = [64,64,128,128,256,256,512,512]
        strides = [2,1,2,1,2,1,2]
        block2 = [Conv_D(in_channels=channels[i],out_channels=channels[i + 1],kernel_size=3,padding = 1,stride = strides[i]) for i in range(len(strides))]
        self.block2 = nn.Sequential(*block2)

        self.block3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512,1024,kernel_size = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024,1,kernel_size = 1)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        batch_size = x.size()[0]
        return torch.sigmoid(x.view(batch_size))

class Conv_D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride = 1):
        """
        conv D block
        Args:
            in_channels: Number of channels for input images
            out_channels: Number of channels for output images
            kernel_size: Size of the convolution kernel
            padding: The size of the feature map complementary zero
            stride: Step size of the convolution
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,padding = padding,stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = channels,out_channels = channels,kernel_size = 3,padding = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,up_scale):
        """
        Upsample by PixelShuffle
        Args:
            in_channels: Number of channels for input images
            up_scale: Magnification of the image
        Examples:
            [B,C,H,W] --> [B,C/4,2H,2W] up_scale=2
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,out_channels = in_channels * up_scale ** 2,kernel_size = 3,padding = 1)
        self.pixel_shullfe = nn.PixelShuffle(upscale_factor = up_scale)
        self.prelu = nn.PReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.pixel_shullfe(x)
        x = self.prelu(x)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

class ConvBlock(nn.Module):
    """
    ( 3x3 conv -> batch norm -> elu ) x 2
    """
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = conv3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        return x

class InBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InBlock, self).__init__()
        self.layer = ConvBlock(in_channel, out_channel)
    
    def forward(self, x):
        x = self.layer(x)
        return x

class OutBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutBlock, self).__init__()
        self.layer = ConvBlock(in_channel, out_channel)
    
    def forward(self, x):
        x = self.layer(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.layer = ConvBlock(in_channel, out_channel)
    def forward(self, x):
        x = self.down(x)
        x = self.layer(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channel, forward_channel, out_channel, bilinear):
        super(UpBlock, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)

        self.layer = ConvBlock(in_channel + forward_channel, out_channel)

    def forward(self, x, prev):
        x = self.up(x) # in_channel -> in_channel
        diffX = x.size()[2] - prev.size()[2]
        diffY = x.size()[3] - prev.size()[3]
        prev = F.pad(prev, (diffX // 2, int(diffX / 2),
                           diffY // 2, int(diffY / 2)))
        x = torch.cat([x, prev], dim=1)
        x = self.layer(x) # in_channel + forward_channel => out_channel
        return x

class UNet(nn.Module):
    def __init__(self, channels, classes, bilinear=True):
        super(UNet, self).__init__()
        self.in_conv = InBlock(channels, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 128)
        self.up1 = UpBlock(128, 128, 64, bilinear)
        self.up2 = UpBlock(64, 64, 32, bilinear)
        self.up3 = UpBlock(32, 32, 16, bilinear)
        self.up4 = UpBlock(16, 16, 16, bilinear)
        self.out_conv = OutBlock(16, classes)

    def forward(self, x):
        x1 = self.in_conv(x) # (2, 224, 224) => (16, 112, 122)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x), x5
        return x
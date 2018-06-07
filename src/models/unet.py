import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_3(in_channel, out_channel):
    # return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
    return SepConv_3(in_channel, out_channel)

class SepConv_3(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.add_module('conv_1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False))
        self.add_module('conv_3', nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel, bias=False))

    def forward(self, x):
        return super().forward(x)

def conv_1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1)

class ConvBlock(nn.Module):
    """
    ( 3x3 conv -> batch norm -> lrelu ) x 2
    """
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = conv_3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.lrelu2 = nn.LeakyReLU(inplace=True)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.lrelu2(out)
        return out

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
    def __init__(self, channels, classes, bilinear=True, unit=4, dropout=False, sigmoid=True):
        super(UNet, self).__init__()
        self.UNIT = unit
        self.sigmoid = sigmoid
        self.in_conv = InBlock(channels, self.UNIT)
        self.down1 = DownBlock(self.UNIT, self.UNIT*2)
        self.down2 = DownBlock(self.UNIT*2, self.UNIT*4)
        self.down3 = DownBlock(self.UNIT*4, self.UNIT*8)
        self.down4 = DownBlock(self.UNIT*8, self.UNIT*8)
        self.dropout1 = nn.Dropout()
        self.up1 = UpBlock(self.UNIT*8, self.UNIT*8, self.UNIT*4, bilinear)
        self.up2 = UpBlock(self.UNIT*4, self.UNIT*4, self.UNIT*2, bilinear)
        self.up3 = UpBlock(self.UNIT*2, self.UNIT*2, self.UNIT, bilinear)
        self.up4 = UpBlock(self.UNIT, self.UNIT, self.UNIT, bilinear)
        self.out_conv = OutBlock(self.UNIT, classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        if self.sigmoid:
            x = F.sigmoid(x)

        return x


class WNet(nn.Module):
    def __init__(self, channels, classes, bilinear=True, unit=4, dropout=False, sigmoid=True):
        super(WNet, self).__init__()
        self.UNIT = unit
        self.sigmoid = sigmoid
        self.in_conv = InBlock(channels, self.UNIT)
        self.down1 = DownBlock(self.UNIT, self.UNIT*2)
        self.down2 = DownBlock(self.UNIT*2, self.UNIT*4)
        self.down3 = DownBlock(self.UNIT*4, self.UNIT*8)
        self.down4 = DownBlock(self.UNIT*8, self.UNIT*8)
        self.up1 = UpBlock(self.UNIT*8, self.UNIT*8, self.UNIT*4, bilinear)
        self.up2 = UpBlock(self.UNIT*4, self.UNIT*4, self.UNIT*2, bilinear)
        self.up3 = UpBlock(self.UNIT*2, self.UNIT*2, self.UNIT, bilinear)
        self.up4 = UpBlock(self.UNIT, self.UNIT, self.UNIT, bilinear)
        self.out_conv = OutBlock(self.UNIT, classes)
        self.dropout1 = nn.Dropout()

        self.in_conv2 = InBlock(channels + classes, self.UNIT)
        self.down1_1 = DownBlock(self.UNIT, self.UNIT*2)
        self.down2_1 = DownBlock(self.UNIT*2, self.UNIT*4)
        self.down3_1 = DownBlock(self.UNIT*4, self.UNIT*8)
        self.down4_1 = DownBlock(self.UNIT*8, self.UNIT*8)
        self.up1_1 = UpBlock(self.UNIT*8, self.UNIT*8, self.UNIT*4, bilinear)
        self.up2_1 = UpBlock(self.UNIT*4, self.UNIT*4, self.UNIT*2, bilinear)
        self.up3_1 = UpBlock(self.UNIT*2, self.UNIT*2, self.UNIT, bilinear)
        self.up4_1 = UpBlock(self.UNIT, self.UNIT, self.UNIT, bilinear)
        self.out_conv2 = OutBlock(self.UNIT, classes)

    def forward(self, input):
        x1 = self.in_conv(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)

        x = torch.cat([x, input], dim=1)
        x1 = self.in_conv2(x)
        x2 = self.down1_1(x1)
        x3 = self.down2_1(x2)
        x4 = self.down3_1(x3)
        x5 = self.down4_1(x4)
        x = self.up1_1(x5, x4)
        x = self.up2_1(x, x3)
        x = self.up3_1(x, x2)
        x = self.up4_1(x, x1)
        x = self.out_conv2(x)
        if self.sigmoid:
            x = F.sigmoid(x)

        return x

class RecurNet(nn.Module):
    def __init__(self, channels, classes, unit=4, loop=4):
        super(RecurNet, self).__init__()
        self.base = UNet(channels+1, classes, False, unit, sigmoid=False)
        self.loop = loop
        self.outblock = conv_1(2 * self.loop, 2)

    def forward(self, x):
        out = []
        pano = x.narrow(1, 1, 1)
        box = (x.narrow(1, 0, 1)).clone()
        x = torch.cat([box, x], dim=1)

        for i in range(self.loop):
            x = self.base(x)
            out.append(x)
            x = torch.cat([x, pano], dim=1)
           
        out = torch.cat(out, dim=1)
        out = self.outblock(out)
        out = F.sigmoid(out)
        return out

class RecurNet2(nn.Module):
    def __init__(self, channels, classes, unit=4, loop=4):
        super(RecurNet2, self).__init__()
        self.base = UNet(channels+1, classes, False, unit, sigmoid=False)
        self.loop = loop
        self.outblock = conv_1(2 * self.loop, 2)

    def forward(self, x):
        out = []
        pano = x.narrow(1, 1, 1)
        box = (x.narrow(1, 0, 1)).clone()
        x = torch.cat([box, x], dim=1)

        for i in range(self.loop):
            x = self.base(x)
            out.append(x)
            x = torch.stack(out, dim=0)
            x = torch.mean(x, dim=0)
            x = torch.cat([x, pano], dim=1)
           
        out = torch.cat(out, dim=1)
        out = self.outblock(out)
        out = F.sigmoid(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

def conv_1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1)

class ConvBlock(nn.Module):
    """
    ( 3x3 conv -> batch norm -> elu ) x 2
    """
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = conv_3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = conv_3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.elu2 = nn.ELU(inplace=True)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.elu2(out)
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

class BranchBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BranchBlock, self).__init__()
        self.conv1 = conv_1(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.elu1 = nn.ELU()
        self.conv2 = conv_1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.elu2 = nn.ELU()
        self.dropout = nn.Dropout()
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.elu2(out)
        out = self.dropout(out)
        return torch.mean(out.view(out.size()[0], -1), dim=1)

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
    def __init__(self, channels, classes, bilinear=True, unit=4, dropout=False):
        super(UNet, self).__init__()
        self.UNIT = unit
        self.dropout = dropout
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
        self.dropout2 = nn.Dropout()
        self.branch = BranchBlock(self.UNIT*8, 1)

    def forward(self, x):
        x1 = self.in_conv(x) # (2, 224, 224) => (16, 112, 122)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.branch(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)

        if self.dropout:
            x = self.dropout2(x)
        return x, y
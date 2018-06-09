import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv_3(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.add_module('conv_1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False))
        self.add_module('conv_3', nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel, bias=False))

    def forward(self, x):
        return super().forward(x)

def conv_3(in_channel, out_channel):
    return SepConv_3(in_channel, out_channel)

def conv_1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1)

# modified from https://github.com/moskomule/senet.pytorch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        shrink = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, shrink),
            nn.LeakyReLU(inplace=True),
            nn.Linear(shrink, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    """
    ( 3x3 conv -> batch norm -> lrelu ) x 2
    """
    def __init__(self, in_channel, out_channel, se=False):
        super(ConvBlock, self).__init__()
        self.conv1 = conv_3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = conv_3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.lrelu2 = nn.LeakyReLU(inplace=True)
        self.se = se

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.se = SELayer(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se(out)
        out += self.shortcut(x)
        out = self.lrelu2(out)
        return out

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
    def __init__(self, in_channel, forward_channel, out_channel):
        super(UpBlock, self).__init__()
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

class Encoder(nn.Module):
    def __init__(self, channels, unit=4):
        super(Encoder, self).__init__()

        self.U = unit
        self.in_conv = ConvBlock(channels, self.U)
        self.d1 = DownBlock(self.U, self.U*2)
        self.d2 = DownBlock(self.U*2, self.U*4)
        self.d3 = DownBlock(self.U*4, self.U*8)
        self.d4 = DownBlock(self.U*8, self.U*8)
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        return x5, x4, x3, x2, x1

class Decoder(nn.Module):
    def __init__(self, classes, unit=4):
        super(Decoder, self).__init__()
        self.U = unit
        self.u1 = UpBlock(self.U*8, self.U*8, self.U*4)
        self.u2 = UpBlock(self.U*4, self.U*4, self.U*2)
        self.u3 = UpBlock(self.U*2, self.U*2, self.U)
        self.u4 = UpBlock(self.U, self.U, self.U)
        self.out_conv = ConvBlock(self.U, classes)
    def forward(self, x):
        x5, x4, x3, x2, x1 = x
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.out_conv(x)
        return x

class SplitNet(nn.Module):
    def __init__(self, channels, classes, unit=4, dropout=False):
        super(SplitNet, self).__init__()
        self.share = Encoder(channels, unit)
        self.dropout = dropout
        self.drop = nn.Dropout2d(p=0.5)
        self.major = Decoder(1, unit)
        self.minor = Decoder(1, unit)
    def forward(self, x):
        x = self.share(x)
        if self.dropout:
            x[0] = self.drop(x[0])
        o_major = self.major(x)
        o_minor = self.minor(x)
        out = torch.cat([o_major, o_minor], dim=1)
        return out

class CrossNet(nn.Module):
    def __init__(self, channels, classes, unit=4, dropout=False):
        super(CrossNet, self).__init__()
        self.share1 = Encoder(channels, unit)
        self.dropout = dropout
        self.drop1 = nn.Dropout2d(p=0.5)
        self.major1 = Decoder(1, unit)
        self.minor1 = Decoder(1, unit)
        self.share2 = Encoder(channels, unit)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.major2 = Decoder(1, unit)
        self.minor2 = Decoder(1, unit)
    def forward(self, x):
        x = self.share1(x)
        if self.dropout:
            x[0] = self.drop1(x[0])
        o_major = self.major1(x)
        o_minor = self.minor1(x)
        out = torch.cat([o_major, o_minor], dim=1)

        x = self.share2(out)
        if self.dropout:
            x[0] = self.drop2(x[0])
        o_major = self.major2(x)
        o_minor = self.minor2(x)
        out = torch.cat([o_major, o_minor], dim=1)
        return out

class MiniDecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(MiniDecoder, self).__init__()
        self.major = UpBlock(in_c, in_c, out_c)
        self.minor = UpBlock(in_c, in_c, out_c)
        self.shrink = conv_1(out_c *2, out_c)
    def forward(self, x, skip):
        o_1 = self.major(x, skip)
        o_2 = self.minor(x, skip)
        x = torch.cat([o_1, o_2], dim=1)
        x = self.shrink(x)
        return x

class CrossDecoder(nn.Module):
    def __init__(self, classes, unit=4):
        super(CrossDecoder, self).__init__()
        self.U = unit
        self.u1 = MiniDecoder(self.U*8, self.U*4)
        self.u2 = MiniDecoder(self.U*4, self.U*2)
        self.u3 = MiniDecoder(self.U*2, self.U)
        self.u4 = MiniDecoder(self.U, self.U)
        self.out1 = ConvBlock(self.U, classes // 2)
        self.out2 = ConvBlock(self.U, classes // 2)
    def forward(self, x):
        x5, x4, x3, x2, x1 = x
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        o1 = self.out1(x)
        o2 = self.out2(x)
        x = torch.cat([o1, o2], dim=1)
        return x

class DNANet(nn.Module):
    def __init__(self, channels, classes, unit=4, dropout=False):
        super(DNANet, self).__init__()
        self.share = Encoder(channels, unit)
        self.dropout = dropout
        self.drop = nn.Dropout2d(p=0.5)
        self.cross = CrossDecoder(classes, unit)

    def forward(self, x):
        x = self.share(x)
        if self.dropout:
            x[0] = self.drop1(x[0])
        x = self.cross(x)
        return x

        
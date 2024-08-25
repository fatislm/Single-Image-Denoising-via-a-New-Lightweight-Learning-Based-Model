import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.residual_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.residual_adjust(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x += residual
        return x

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=24):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.twocon1 = nn.Conv2d(n_chan, chan_embed, kernel_size=3, padding=1)
        self.resnet_block1 = ResnetBlock(n_chan, chan_embed)

        self.twocon2 = nn.Conv2d(chan_embed, chan_embed, kernel_size=3, padding=1)
        self.resnet_block2 = ResnetBlock(chan_embed, chan_embed)

        self.twocon3 = nn.Conv2d(chan_embed, n_chan, kernel_size=1, padding=0)
        self.resnet_block3 = ResnetBlock(chan_embed, n_chan, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = x
        x = self.act(self.twocon1(x))
        x2 = self.resnet_block1(x1)
        x = x2 + x

        x3 = x
        x = self.act(self.twocon2(x))
        x4 = self.resnet_block2(x3)
        x = x4 + x

        x5 = x
        x = self.act(self.twocon3(x))
        x6 = self.resnet_block3(x5)
        x = x6 + x
        return x

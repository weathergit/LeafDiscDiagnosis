# encoding:utf-8
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MaskBlock(nn.Module):
    def __init__(self, embed_dim):
        super(MaskBlock, self).__init__()
        self.act = nn.ReLU(True)
        self.conv_head = default_conv(embed_dim, embed_dim, 3)

        self.conv_self = default_conv(embed_dim, embed_dim, 1)

        self.conv1 = default_conv(embed_dim, embed_dim, 3)
        self.conv1_1 = default_conv(embed_dim, embed_dim, 1)
        self.conv1_2 = default_conv(embed_dim, embed_dim, 1)
        self.conv_tail = default_conv(embed_dim, embed_dim, 3)

    def forward(self, x):
        x = self.conv_head(x)
        x = self.conv_self(x)
        x = x.mul(x)
        x = self.act(self.conv1(x))
        x = self.conv1_1(x).mul(self.conv1_2(x))

        return self.conv_tail(x)


class SnowMaskBlock(nn.Module):
    def __init__(self, embed_dim):
        super(SnowMaskBlock, self).__init__()
        self.smblock = MaskBlock(embed_dim)
        self.conv3 = default_conv(embed_dim, embed_dim, 3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        shortcut = x
        x = self.smblock(x)
        x = self.norm(x.flatten(2).transpose(-1, -2))
        x = self.conv3(x.transpose(-1, -2).view(B, -1, H, W))
        return x + shortcut


if __name__ == '__main__':
    x = torch.rand((64, 64, 224, 224))
    print((x.shape))
    embed_dim = x.shape[1]
    model = SnowMaskBlock(embed_dim=embed_dim)
    y = model(x)
    print(y.shape)
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
import warnings


class SENet(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SENet, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = self.fc(out.view(out.size(0), -1))
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out*x


class UNetSE(nn.Module):
    def __init__(self):
        super(UNetSE, self).__init__()
        self.__name__ = 'UNetSE52'
        self.encoder = get_encoder(
            name='resnet34', in_channels=3,
            depth=5, weights='imagenet')

        self.decoder = UnetDecoder(
            encoder_channels=[3, 64, 64, 128, 256, 512],
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=2,
            activation='sigmoid',
            kernel_size=3,
        )
        self.se_list = nn.ModuleList([SENet(in_channel=chan) for chan in [64, 64, 128, 256, 512]])

    def forward(self, x):
        x_list = self.encoder(x)
        x_list[2] = self.se_list[1](x_list[2])
        x_list[-1] = self.se_list[-1](x_list[-1])
        # for i, (xx, senet) in enumerate(zip(x_list[1:], self.se_list)):
        #     x_list[i+1] = senet(xx)
        x4 = self.decoder(*x_list)
        x5 = self.segmentation_head(x4)
        return x5


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 224, 224))
    net = UNetSE()
    y = net(x)
    print(y.shape)

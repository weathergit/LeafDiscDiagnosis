import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
import math
import warnings
import torch.nn.functional as F


class SPPF(nn.Module):
    # results : 96.42% -1
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class UMSNet(nn.Module):
    def __init__(self):
        super(UMSNet, self).__init__()
        self.__name__ = 'UMSNet'
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
        # concat all depth
        self.sppf = nn.ModuleList([SPPF(c1=chan, c2=chan) for chan in [64, 64, 128, 256, 512]])
        # concat last depth
        # self.sppf = SPPF(c1=512, c2=512)

    def forward(self, x):
        x_list = self.encoder(x)
        # x_list[-1] = self.sppf(x_list[-1])
        for i, (xx, sppf) in enumerate(zip(x_list[1:], self.sppf)):
            x_list[i+1] = sppf(xx)
        x4 = self.decoder(*x_list)
        x5 = self.segmentation_head(x4)
        return x5


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 224, 224))
    net = UMSNet()
    y = net(x)
    print(y.shape)

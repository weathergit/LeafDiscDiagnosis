import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
import math


class DCTUNet(nn.Module):
    def __init__(self):
        super(DCTUNet, self).__init__()
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

        self.fc = nn.Linear(in_features=512, out_features=2)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.alpha_range = 2.0
        self.beta_range = 0.1

    def forward(self, x):
        # DCT regression
        x_list = self.encoder(x)
        x1 = x_list[-1]

        x_pool = self.pool(x1)
        x2 = x_pool.view(x_pool.shape[:-2])

        x2 = self.fc(x2)
        alpha = torch.sigmoid(x2[:, :1, ...]) * self.alpha_range
        alpha = alpha.view(-1, 1, 1, 1)

        beta = torch.atan(x2[:, 1:, ...]) * self.beta_range / math.pi
        beta = beta.view(-1, 1, 1, 1)

        new_list = []
        for z in x_list:
            zy = z * alpha + beta
            new_list.append(zy)

        x4 = self.decoder(*new_list)

        x5 = self.segmentation_head(x4)
        return x5


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 224, 224))
    # encoder = get_encoder(name='resnet34')
    # y = encoder(x)
    # print(encoder.out_channels)
    # for i in range(len(y)):
    #     print(y[i].shape)
    net = DCTUNet()
    y = net(x)
    print(y.shape)

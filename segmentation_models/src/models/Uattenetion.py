import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
# import math
# from attention_module import cbam_block
from attention_module import SENet


class UANet(nn.Module):
    def __init__(self):
        super(UANet, self).__init__()
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
        new_list = []
        new_list.append(x_list[0])
        for y in x_list[1:]:
            channel = y.shape[1]
            attention = SENet(in_channel=channel).to('cuda')
            yy = attention(y)
            new_list.append(yy)
        x4 = self.decoder(*new_list)
        x5 = self.segmentation_head(x4)
        return x5


if __name__ == '__main__':
    x = torch.randn(size=(8, 3, 224, 224))
    # encoder = get_encoder(name='resnet34')
    # y = encoder(x)
    # print(encoder.out_channels)
    # for i in range(len(y)):
    #     print(y[i].shape)
    net = UANet()
    print(net)
    y = net(x)
    print(y.shape)

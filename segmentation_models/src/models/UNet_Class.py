import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import ClassificationHead


class UCNet(nn.Module):
    def __init__(self):
        super(UCNet, self).__init__()
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

        self.class_head = ClassificationHead(
            in_channels=16,
            classes=6,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(*x)
        x1 = self.segmentation_head(x)
        # x2 = self.class_head(x)
        return x1


if __name__ == '__main__':
    x = torch.randn(size=(8, 3, 224, 224))
    net = UCNet()
    y1, y2 = net(x)
    print(y1.shape)
    print(y2.shape)

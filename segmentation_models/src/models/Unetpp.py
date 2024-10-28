import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.base import SegmentationHead

from SnowMask import SnowMaskBlock


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class UPPNet(nn.Module):
    def __init__(self):
        super(UPPNet, self).__init__()
        self.encoder = get_encoder(
            name='resnet34', in_channels=3,
            depth=5, weights='imagenet')

        self.decoder = UnetPlusPlusDecoder(
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

    def forward(self, x):
        x_list = self.encoder(x)
        new_list = []
        new_list.append(x_list[0])
        for tensor in x_list[1:]:
            embed = tensor.shape[1]
            sm = SnowMaskBlock(embed_dim=embed).to(device)
            yy = sm(tensor)
            new_list.append(yy)
        x4 = self.decoder(*new_list)
        x5 = self.segmentation_head(x4)
        return x5


if  __name__ == '__main__':
    x = torch.rand((64, 3, 224, 224))
    print(x.shape)
    model = UPPNet()
    y = model(x)
    print(y.shape)
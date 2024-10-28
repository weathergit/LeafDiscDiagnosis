import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.multi_scale = SPP(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.multi_scale(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(SegmentationModel, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for i in range(depth):
            if i == 0:
                self.encoder_blocks.append(EncoderBlock(in_channels, 64))
            else:
                self.encoder_blocks.append(EncoderBlock(64 * 2**(i-1), 64 * 2**i))
        
        for i in range(depth-1, 0, -1):
            self.decoder_blocks.append(DecoderBlock(64 * 2**i, 64 * 2**(i-1)))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skips.append(x)
            x = self.pool(x)
        
        x = skips.pop()
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, skips.pop())
        
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    # 初始化权重采用 Xavier 分布
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)

    # 使用示例
    num_classes = 2
    model = SegmentationModel(in_channels=3, out_channels=num_classes, depth=3)
    model.apply(weights_init)
    x = torch.randn(size=(1,3,224,224))
    y = model(x)
    print(y.shape)


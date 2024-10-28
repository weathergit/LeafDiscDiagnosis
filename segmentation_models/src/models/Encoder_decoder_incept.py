import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3_reduce = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        self.conv5x5_reduce = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv5x5 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)

    def forward(self, x):
        out1x1 = F.relu(self.conv1x1(x))
        out3x3 = F.relu(self.conv3x3(self.conv3x3_reduce(x)))
        out5x5 = F.relu(self.conv5x5(self.conv5x5_reduce(x)))
        out_pool = F.relu(self.pool_proj(self.pool(x)))
        return torch.cat([out1x1, out3x3, out5x5, out_pool], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.inception = InceptionModule(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.inception(x)
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




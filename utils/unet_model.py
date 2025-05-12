import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def crop_and_concat(enc_feat, dec_feat):
    """
    手动中心裁剪 enc_feat 以匹配 dec_feat 尺寸，然后拼接
    """
    _, _, h1, w1 = enc_feat.shape
    _, _, h2, w2 = dec_feat.shape

    # 计算起始裁剪位置
    delta_h = (h1 - h2) // 2
    delta_w = (w1 - w2) // 2

    enc_feat = enc_feat[:, :, delta_h:delta_h + h2, delta_w:delta_w + w2]
    return torch.cat([dec_feat, enc_feat], dim=1)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(crop_and_concat(e4, d4))

        d3 = self.up3(d4)
        d3 = self.dec3(crop_and_concat(e3, d3))

        d2 = self.up2(d3)
        d2 = self.dec2(crop_and_concat(e2, d2))

        d1 = self.up1(d2)
        d1 = self.dec1(crop_and_concat(e1, d1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out  # [B, 1, H, W] logits

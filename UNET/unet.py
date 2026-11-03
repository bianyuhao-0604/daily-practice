"""
unet.py — 标准 U-Net（Ronneberger 2015 风格）
默认 base=32 时参数量约 7.2M（双线性）/ 7.8M（转置卷积），256×256 输入下 8GB 显存无压力；
bilinear=True 用双线性上采样+1×1 卷积减通道，False 用转置卷积，两条路径均可跑通。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """上采样 + skip 拼接 + DoubleConv。

    两条上采样路径的输出通道必须一致（都是 in_ch//2），这样与 skip 的 in_ch//2
    拼接后恰好是 DoubleConv 的输入通道数 in_ch：
      - 转置卷积：ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
      - 双线性：  Upsample 不改通道，必须再补一个 1×1 卷积把 in_ch 压到 in_ch//2
    """
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐尺寸（输入边长非 2 的整数次幂倍数时）
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, base=32, bilinear=True):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = Down(base,     base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, bilinear)
        self.up2 = Up(base * 8,  base * 4, bilinear)
        self.up3 = Up(base * 4,  base * 2, bilinear)
        self.up4 = Up(base * 2,  base,     bilinear)
        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)   # logits，不加 sigmoid


if __name__ == "__main__":
    # 自检：两条上采样路径、常规/奇数输入尺寸、参数量
    for bl in (True, False):
        m = UNet(bilinear=bl)
        n = sum(p.numel() for p in m.parameters())
        y = m(torch.randn(2, 1, 256, 256))
        assert y.shape == (2, 1, 256, 256), y.shape
        y2 = m(torch.randn(1, 1, 97, 97))          # 奇数边长，走 Up 内 pad 对齐
        assert y2.shape == (1, 1, 97, 97), y2.shape
        print(f"bilinear={bl}: forward OK, params = {n/1e6:.2f}M")

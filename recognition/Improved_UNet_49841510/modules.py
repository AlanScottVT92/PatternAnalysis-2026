import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)

class Up(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        self.inc = DoubleConv(in_channels, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up4 = Up(32, 16, 16)

        self.outc = OutConv(16, num_classes)

    def forward(self, x):
        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits



if __name__ == "__main__":
    block = DoubleConv(1, 16)

    x = torch.randn(4, 1, 256, 128)
    y = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    double_conv = DoubleConv(1, 16)
    down = Down(16, 32)

    y1 = double_conv(x)
    y2 = down(y1)

    print("Input:", x.shape)
    print("After DoubleConv:", y1.shape)
    print("After Down:", y2.shape)

    up = Up(32, 16, 16)

    y3 = up(y2, y1)

    print("After Up:", y3.shape)

    out_conv = OutConv(16, 4)

    y4 = out_conv(y3)

    print("After OutConv:", y4.shape)

    prediction = torch.argmax(y4, dim=1)

    print("Prediction shape:", prediction.shape)



    model = UNet(in_channels=1, num_classes=4)

    x = torch.randn(4, 1, 256, 128)

    with torch.no_grad():
        output = model(x)

    print("U-Net Input:", x.shape)
    print("U-Net Output:", output.shape)
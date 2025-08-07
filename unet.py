from torch import nn
import torch




class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv1(x)

class Double_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv_3_k(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            Conv_3_k(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Double_Conv(channels_in, channels_out)
        )
    def forward(self, x):
        return self.encoder(x)

class Up_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.decoder = Double_Conv(channels_in, channels_out)
    def forward(self, x1, x2):
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)

# UNET implementation
class UNET(nn.Module):
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels)
        self.down_conv1 = Down_Conv(channels, 2 * channels)
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels)
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels)
        self.middle_conv = Down_Conv(8 * channels, 16 * channels)
        self.up_conv1 = Up_Conv(16 * channels, 8 * channels)
        self.up_conv2 = Up_Conv(8 * channels, 4 * channels)
        self.up_conv3 = Up_Conv(4 * channels, 2 * channels)
        self.up_conv4 = Up_Conv(2 * channels, channels)
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.middle_conv(x4)
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        return self.sigmoid(self.last_conv(u4))
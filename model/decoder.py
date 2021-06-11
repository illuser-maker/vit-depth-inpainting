import torch
import torch.nn as nn
import torchvision

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, output_padding=0, batch_norm=True, relu=True):
        super().__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding,
                                        output_padding, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dec5 = DeconvBlock(in_channels, 256, kernel=2, stride=2, padding=1, output_padding=1)
        self.dec4 = DeconvBlock(256, 128, kernel=2, stride=2, padding=1, output_padding=0)
        self.dec3 = DeconvBlock(128, 64, kernel=2, stride=2, padding=1, output_padding=0)
        self.dec2 = DeconvBlock(64, 32, kernel=2, stride=2, padding=1, output_padding=0)
        self.dec1 = DeconvBlock(32, 16, kernel=2, stride=1, padding=1, output_padding=0)
        self.dec0 = DeconvBlock(16, 1, kernel=2, stride=1, padding=1, output_padding=0)

    def forward(self, input):
        x = self.dec5(input)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.dec0(x)
        return x

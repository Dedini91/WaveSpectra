"""Model definitions"""
import torch.nn.functional as F
import torch.nn as nn
import torch

from utils.utils import init_layers


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode=self.mode)
        return x


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()

        self.resize14_36 = Interpolate(size=(14, 36), mode='bilinear')
        self.resize28_72 = Interpolate(size=(28, 72), mode='bilinear')
        self.resize56_144 = Interpolate(size=(56, 144), mode='bilinear')
        self.resize112_288 = Interpolate(size=(112, 288), mode='bilinear')

        self.pre1_1 = nn.Sequential(
            self.resize112_288,  # 64x32x32 -> 64x64x64
            nn.Conv2d(1, 4, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pre1_2 = nn.Sequential(
            nn.Conv2d(4, 8, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.pre1_3 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.input_layer1 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),   # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc1 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc1 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc1 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec_int1 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize14_36,  # 64x32x32 -> 64x64x64
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, (1, 1), (1, 1)),     # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize28_72,    # 64x32x32 -> 64x64x64
        )
        self.clean1 = nn.Sequential(
            self.resize56_144,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )
        self.conv_out1 = nn.Conv2d(32, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64

        self.pre1_1.apply(init_layers)
        self.pre1_2.apply(init_layers)
        self.pre1_3.apply(init_layers)
        self.input_layer1.apply(init_layers)
        self.sml_enc1.apply(init_layers)
        self.med_enc1.apply(init_layers)
        self.lrg_enc1.apply(init_layers)
        self.dec_int1.apply(init_layers)
        self.dec1.apply(init_layers)
        nn.init.kaiming_uniform_(self.conv_out1.weight)

        self.pre2_1 = nn.Sequential(
            self.resize112_288,  # 64x32x32 -> 64x64x64
            nn.Conv2d(4, 4, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pre2_2 = nn.Sequential(
            nn.Conv2d(4, 8, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.pre2_3 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.input_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc2 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc2 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec_int2 = nn.Sequential(
            nn.Conv2d(224, 128, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize14_36,  # 64x32x32 -> 64x64x64
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize28_72,  # 64x32x32 -> 64x64x64
        )
        self.clean2 = nn.Sequential(
            self.resize56_144,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )
        self.conv_out2 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64

        self.pre2_1.apply(init_layers)
        self.pre2_2.apply(init_layers)
        self.pre2_3.apply(init_layers)
        self.input_layer2.apply(init_layers)
        self.sml_enc2.apply(init_layers)
        self.med_enc2.apply(init_layers)
        self.lrg_enc2.apply(init_layers)
        self.dec_int2.apply(init_layers)
        self.dec2.apply(init_layers)
        nn.init.kaiming_uniform_(self.conv_out2.weight)

        self.relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv_out = nn.Conv2d(4, 1, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out.apply(init_layers)

    def forward(self, input):
        w = self.pre1_1(input)
        w = self.pre1_2(w)
        w = self.pre1_3(w)
        w = self.input_layer1(w)
        x = self.sml_enc1(w)
        y = self.med_enc1(w)
        z = self.lrg_enc1(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec_int1(w)
        w = self.dec1(w)
        w = self.relu(self.conv_out1(w))
        w = self.clean1(w)

        w = self.pre2_1(w)
        w = self.pre2_2(w)
        w = self.pre2_3(w)
        w = self.input_layer2(w)
        x = self.sml_enc2(w)
        y = self.med_enc2(w)
        z = self.lrg_enc2(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec_int2(w)
        w = self.dec2(w)
        w = self.relu(self.conv_out2(w))
        w = self.clean2(w)

        output = self.conv_out(w)

        return output

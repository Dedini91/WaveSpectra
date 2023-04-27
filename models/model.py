"""Model definitions"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import sigmoid, init_layers, normalise_to_source


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
        self.resize64 = Interpolate(size=(64, 64), mode='bilinear')
        self.resize128 = Interpolate(size=(128, 128), mode='bilinear')

        self.input_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),   # 1x64x64 -> 32x32x32
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
        self.dec1 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),     # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,    # 64x32x32 -> 64x64x64
        )

        self.input_layer1.apply(init_layers)
        self.sml_enc1.apply(init_layers)
        self.med_enc1.apply(init_layers)
        self.lrg_enc1.apply(init_layers)
        self.dec1.apply(init_layers)

        self.input_layer2 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
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
        self.dec2 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer2.apply(init_layers)
        self.sml_enc2.apply(init_layers)
        self.med_enc2.apply(init_layers)
        self.lrg_enc2.apply(init_layers)
        self.dec2.apply(init_layers)

        self.input_layer3 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc3 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc3 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc3 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer3.apply(init_layers)
        self.sml_enc3.apply(init_layers)
        self.med_enc3.apply(init_layers)
        self.lrg_enc3.apply(init_layers)
        self.dec3.apply(init_layers)

        self.input_layer4 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc4 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc4 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc4 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer4.apply(init_layers)
        self.sml_enc4.apply(init_layers)
        self.med_enc4.apply(init_layers)
        self.lrg_enc4.apply(init_layers)
        self.dec4.apply(init_layers)

        self.input_layer5 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc5 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc5 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc5 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer5.apply(init_layers)
        self.sml_enc5.apply(init_layers)
        self.med_enc5.apply(init_layers)
        self.lrg_enc5.apply(init_layers)
        self.dec5.apply(init_layers)

        self.input_layer6 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc6 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc6 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc6 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec6 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer6.apply(init_layers)
        self.sml_enc6.apply(init_layers)
        self.med_enc6.apply(init_layers)
        self.lrg_enc6.apply(init_layers)
        self.dec6.apply(init_layers)

        self.input_layer7 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc7 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc7 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc7 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec7 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer7.apply(init_layers)
        self.sml_enc7.apply(init_layers)
        self.med_enc7.apply(init_layers)
        self.lrg_enc7.apply(init_layers)
        self.dec7.apply(init_layers)

        self.input_layer8 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc8 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc8 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc8 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec8 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer8.apply(init_layers)
        self.sml_enc8.apply(init_layers)
        self.med_enc8.apply(init_layers)
        self.lrg_enc8.apply(init_layers)
        self.dec8.apply(init_layers)

        self.input_layer9 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 1x64x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.sml_enc9 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3), (2, 2), 1, padding_mode='reflect'),  # 32x32x32 -> 128x32x32
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.med_enc9 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), (2, 2), 2, padding_mode='reflect'),  # 32x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.lrg_enc9 = nn.Sequential(
            nn.Conv2d(32, 32, (7, 7), (2, 2), 3, padding_mode='reflect'),  # 32x32x32 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dec9 = nn.Sequential(
            nn.Conv2d(224, 64, (1, 1), (1, 1)),  # 224x32x32 -> 64x32x32
            nn.LeakyReLU(negative_slope=0.2),
            self.resize64,  # 64x32x32 -> 64x64x64
        )

        self.input_layer9.apply(init_layers)
        self.sml_enc9.apply(init_layers)
        self.med_enc9.apply(init_layers)
        self.lrg_enc9.apply(init_layers)
        self.dec9.apply(init_layers)

        self.conv_out1 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out2 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out3 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out4 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out5 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out6 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out7 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out8 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.conv_out9 = nn.Conv2d(64, 4, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        nn.init.kaiming_uniform_(self.conv_out1.weight)
        nn.init.kaiming_uniform_(self.conv_out2.weight)
        nn.init.kaiming_uniform_(self.conv_out3.weight)
        nn.init.kaiming_uniform_(self.conv_out4.weight)
        nn.init.kaiming_uniform_(self.conv_out5.weight)
        nn.init.kaiming_uniform_(self.conv_out6.weight)
        nn.init.kaiming_uniform_(self.conv_out7.weight)
        nn.init.kaiming_uniform_(self.conv_out8.weight)
        nn.init.kaiming_uniform_(self.conv_out9.weight)

        self.clean1 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean2 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean3 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean4 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean5 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean6 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean7 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean8 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean9 = nn.Sequential(
            self.resize128,
            nn.AvgPool2d((2, 2), (2, 2), padding=0)
        )

        self.clean1.apply(init_layers)
        self.clean2.apply(init_layers)
        self.clean3.apply(init_layers)
        self.clean4.apply(init_layers)
        self.clean5.apply(init_layers)
        self.clean6.apply(init_layers)
        self.clean7.apply(init_layers)
        self.clean8.apply(init_layers)
        self.clean9.apply(init_layers)

        self.conv_out10 = nn.Conv2d(4, 1, (1, 1), (1, 1))  # 64x64x64 -> 1x64x64
        nn.init.xavier_uniform_(self.conv_out5.weight)

    def forward(self, input):
        w = self.input_layer1(input)
        x = self.sml_enc1(w)
        y = self.med_enc1(w)
        z = self.lrg_enc1(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec1(w)
        w = self.relu(self.conv_out1(w))
        w = self.clean1(w)

        w = self.input_layer2(w)
        x = self.sml_enc2(w)
        y = self.med_enc2(w)
        z = self.lrg_enc2(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec2(w)
        w = self.relu(self.conv_out2(w))
        w = self.clean2(w)

        w = self.input_layer3(w)
        x = self.sml_enc3(w)
        y = self.med_enc3(w)
        z = self.lrg_enc3(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec3(w)
        w = self.relu(self.conv_out3(w))
        w = self.clean3(w)

        w = self.input_layer4(w)
        x = self.sml_enc4(w)
        y = self.med_enc4(w)
        z = self.lrg_enc4(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec4(w)
        w = self.relu(self.conv_out4(w))
        w = self.clean4(w)

        w = self.input_layer5(w)
        x = self.sml_enc5(w)
        y = self.med_enc5(w)
        z = self.lrg_enc5(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec5(w)
        w = self.relu(self.conv_out5(w))
        w = self.clean5(w)

        w = self.input_layer6(w)
        x = self.sml_enc6(w)
        y = self.med_enc6(w)
        z = self.lrg_enc6(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec6(w)
        w = self.relu(self.conv_out6(w))
        w = self.clean6(w)

        w = self.input_layer7(w)
        x = self.sml_enc7(w)
        y = self.med_enc7(w)
        z = self.lrg_enc7(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec7(w)
        w = self.relu(self.conv_out7(w))
        w = self.clean7(w)

        w = self.input_layer8(w)
        x = self.sml_enc8(w)
        y = self.med_enc8(w)
        z = self.lrg_enc8(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec8(w)
        w = self.relu(self.conv_out8(w))
        w = self.clean8(w)

        w = self.input_layer9(w)
        x = self.sml_enc9(w)
        y = self.med_enc9(w)
        z = self.lrg_enc9(w)
        w = torch.cat((x, y, z), 1)
        w = self.dec9(w)
        w = self.relu(self.conv_out9(w))
        w = self.clean9(w)

        output = normalise_to_source(input, sigmoid(self.conv_out10(w)))

        return output


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, padding_mode='reflect')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(128, 192, 3, 1, 1, padding_mode='reflect')
        self.conv_extra1 = nn.Conv2d(192, 256, 3, 1, 1, padding_mode='reflect')
        self.conv_extra2 = nn.Conv2d(256, 512, 3, 1, 1, padding_mode='reflect')
        self.conv_extra3 = nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect')
        self.conv_extra7 = nn.Conv2d(256, 128, 3, 1, 1, padding_mode='reflect')
        self.conv_extra8 = nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1, padding_mode='reflect')

        # Kaiming/Xavier initialization for relu/sigmoid
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)
        nn.init.kaiming_uniform_(self.conv_extra1.weight)
        nn.init.kaiming_uniform_(self.conv_extra2.weight)
        nn.init.kaiming_uniform_(self.conv_extra3.weight)
        nn.init.kaiming_uniform_(self.conv_extra7.weight)
        nn.init.kaiming_uniform_(self.conv_extra8.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv_extra1(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = self.lrelu(self.conv_extra2(x))
        x = self.lrelu(self.conv_extra3(x))
        x = self.lrelu(self.conv_extra7(x))
        x = self.lrelu(self.conv_extra8(x))

        x = sigmoid(self.conv4(x))

        return x
    
    
class SmlTest(nn.Module):
    def __init__(self):
        super(CAEsml, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv2d(1, 3, 3, 2, 1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(3, 2, 1, 1, 0, padding_mode='reflect')
        self.conv4 = nn.Conv2d(2, 1, 3, 1, 1, padding_mode='reflect')

        self.conv5 = nn.Conv2d(1, 3, 3, 2, 1, padding_mode='reflect')
        self.conv7 = nn.Conv2d(3, 2, 1, 1, 0, padding_mode='reflect')
        self.conv8 = nn.Conv2d(2, 1, 3, 1, 1, padding_mode='reflect')
        
        self.conv1b = nn.Conv2d(1, 3, 3, 2, 1, padding_mode='reflect')
        self.conv3b = nn.Conv2d(3, 2, 1, 1, 0, padding_mode='reflect')
        self.conv4b = nn.Conv2d(2, 1, 3, 1, 1, padding_mode='reflect')

        self.conv5b = nn.Conv2d(1, 3, 3, 2, 1, padding_mode='reflect')
        self.conv7b = nn.Conv2d(3, 2, 1, 1, 0, padding_mode='reflect')
        self.conv8b = nn.Conv2d(2, 1, 3, 1, 1, padding_mode='reflect')

        # Kaiming/Xavier initialization for relu/sigmoid
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        
        nn.init.kaiming_uniform_(self.conv5.weight)
        nn.init.kaiming_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        
        nn.init.kaiming_uniform_(self.conv1b.weight)
        nn.init.kaiming_uniform_(self.conv3b.weight)
        nn.init.xavier_uniform_(self.conv4b.weight)
        
        nn.init.kaiming_uniform_(self.conv5b.weight)
        nn.init.kaiming_uniform_(self.conv7b.weight)
        nn.init.xavier_uniform_(self.conv8b.weight)

    def forward(self, input):
        src_min = torch.min(input)
        src_max = torch.max(input)

        x = self.lrelu(self.conv1(input))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = self.lrelu(self.conv3(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * (src_max - src_min) + src_min
        x = sigmoid(self.conv4(x))


        x = self.lrelu(self.conv5(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = self.lrelu(self.conv7(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * (src_max - src_min) + src_min
        x = sigmoid(self.conv8(x))


        x = self.lrelu(self.conv1b(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = self.lrelu(self.conv3b(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * (src_max - src_min) + src_min
        x = sigmoid(self.conv4b(x))
        
        x = self.lrelu(self.conv5b(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = self.lrelu(self.conv7b(x))
        x = F.interpolate(x, (64, 64), mode='nearest-exact')
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * (src_max - src_min) + src_min
        x = sigmoid(self.conv8b(x))

        return x

"""Model definitions"""
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import sigmoid


# layers calculated with https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, padding_mode='reflect')
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(128, 192, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv_extra1 = nn.Conv2d(192, 256, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv_extra2 = nn.Conv2d(256, 512, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv_extra3 = nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv_extra7 = nn.Conv2d(256, 128, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv_extra8 = nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1, padding_mode='reflect')  # or 5, 1, 2 to preserve dims

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
        # x = self.lrelu(self.conv1(F.interpolate(x, (256, 256), mode='nearest-exact')))
        x = self.lrelu(self.conv1(x))   # 64->32
        x = self.lrelu(self.conv2(x))   # 32->16
        x = self.lrelu(self.conv3(x))   # 16
        x = self.lrelu(self.conv_extra1(x))     # 16
        x = F.interpolate(x, (64, 64), mode='nearest-exact')    # 128
        x = self.lrelu(self.conv_extra2(x))     # 128
        x = self.lrelu(self.conv_extra3(x))     # 64
        x = self.lrelu(self.conv_extra7(x))
        x = self.lrelu(self.conv_extra8(x))

        x = sigmoid(self.conv4(x))

        return x

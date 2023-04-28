

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

from torch import nn


class IdResidualConvBlockBNIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, nonlin=nn.LeakyReLU):
        super(IdResidualConvBlockBNIdentity, self).__init__()
        self.nonlin = nonlin()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        return 0.1 * h + x


class IdResidualConvBlockBNResize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, nonlin=nn.LeakyReLU):
        super(IdResidualConvBlockBNResize, self).__init__()
        self.nonlin = nonlin()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=2, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv_residual = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=2, padding=0, bias=False)
        self.bn_residual = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        residual = self.conv_residual(x)
        residual = self.bn_residual(residual)

        return 0.1 * h + residual


class IdResidualConvTBlockBNResize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding=0, nonlin=nn.LeakyReLU):
        super(IdResidualConvTBlockBNResize, self).__init__()
        self.nonlin = nonlin()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        self.conv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, stride=2, padding=self.padding, output_padding=self.output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv_residual = nn.ConvTranspose2d(self.in_channels, self.out_channels, 1, stride=2, padding=0, output_padding=self.output_padding, bias=False)
        self.bn_residual = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        residual = self.conv_residual(x)
        residual = self.bn_residual(residual)

        return 0.1 * h + residual


class IdResidualConvTBlockBNIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding=0, nonlin=nn.LeakyReLU):
        super(IdResidualConvTBlockBNIdentity, self).__init__()
        self.nonlin = nonlin()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        self.conv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=self.output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        return 0.1 * h + x
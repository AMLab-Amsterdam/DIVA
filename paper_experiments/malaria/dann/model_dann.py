import torch
import torch.nn.functional as F

from paper_experiments.malaria.dann.functions_dann import ReverseLayerF
from paper_experiments.resnet_blocks_batchnorm import *


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.rn1 = IdResidualConvBlockBNResize(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvBlockBNIdentity(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvBlockBNResize(32, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)

        self.class_classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 1024, bias=False),
                                              nn.BatchNorm1d(1024),
                                              nn.LeakyReLU(),
                                              nn.Linear(1024, 2))
        self.domain_classifier = nn.Sequential(nn.Linear(64 * 4 * 4, 1024, bias=False),
                                               nn.BatchNorm1d(1024),
                                               nn.LeakyReLU(),
                                               nn.Linear(1024, 9))

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.class_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)
        self.class_classifier[1].weight.data.fill_(1)
        self.class_classifier[1].bias.data.zero_()
        self.domain_classifier[1].weight.data.fill_(1)
        self.domain_classifier[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.class_classifier[3].weight)
        self.class_classifier[3].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.domain_classifier[3].weight)
        self.domain_classifier[3].bias.data.zero_()

    def forward(self, input_data, alpha):
        h = self.conv1(input_data)
        h = self.bn1(h)
        # activation function is inside of IdResidualConvBlockBN

        h = self.rn1(h)
        h = self.rn2(h)
        h = self.rn3(h)
        h = self.rn4(h)
        h = self.rn5(h)
        h = self.rn6(h)
        h = self.rn7(h)
        h = F.leaky_relu(h)
        h = h.view(-1, 64 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(h, alpha)
        class_output = self.class_classifier(h)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
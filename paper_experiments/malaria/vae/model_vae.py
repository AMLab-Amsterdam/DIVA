import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from paper_experiments.malaria.diva.pixel_cnn_utils import log_mix_dep_Logistic_256
from paper_experiments.resnet_blocks_batchnorm import *


# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(z_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        self.rn1 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvTBlockBNResize(64, 64, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvTBlockBNResize(64, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvTBlockBNIdentity(32, 32, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvTBlockBNResize(32, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvTBlockBNIdentity(32, 32, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn8 = IdResidualConvTBlockBNResize(32, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.conv1 = nn.Conv2d(32, 100, 3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, 1, padding=0)

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.zero_()

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 64, 4, 4)
        h = self.rn1(h)
        h = self.rn2(h)
        h = self.rn3(h)
        h = self.rn4(h)
        h = self.rn5(h)
        h = self.rn6(h)
        h = self.rn7(h)
        h = self.rn8(h)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        loc_img = self.conv2(h)

        return loc_img


# Encoders
class qz(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qz, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.rn1 = IdResidualConvBlockBNResize(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvBlockBNIdentity(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvBlockBNResize(32, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)

        self.fc11 = nn.Sequential(nn.Linear(64 * 4 * 4, z_dim))
        self.fc12 = nn.Sequential(nn.Linear(64 * 4 * 4, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.conv1(x)
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
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(z_dim, d_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = self.activation(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(z_dim, y_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = self.activation(zy)
        loc_y = self.fc1(h)

        return loc_y


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_dim = args.z_dim
        self.d_dim = args.d_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        self.qz = qz(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.z_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.z_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta = args.beta

        self.cuda()

    def forward(self, d, x, y):
        # Encode
        z_q_loc, z_q_scale = self.qz(x)

        # Reparameterization trick
        qz = dist.Normal(z_q_loc, z_q_scale)
        z_q = qz.rsample()

        # Decode
        x_recon = self.px(z_q)

        # Prior
        z_p_loc, z_p_scale = torch.zeros(z_q.size()[0], self.z_dim).cuda(),\
                                   torch.ones(z_q.size()[0], self.z_dim).cuda()

        # Reparameterization trick
        pz = dist.Normal(z_p_loc, z_p_scale)

        # Auxiliary losses
        d_hat = self.qd(z_q)
        y_hat = self.qy(z_q)

        return x_recon, d_hat, y_hat, qz, pz, z_q

    def loss_function(self, d, x, y):
        x_recon, d_hat, y_hat, qz, pz, z_q = self.forward(d, x, y)

        CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)

        KL_z = torch.sum(pz.log_prob(z_q) - qz.log_prob(z_q))

        _, d_target = d.max(dim=1)
        CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

        _, y_target = y.max(dim=1)
        CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

        return CE_x - self.beta * KL_z + self.aux_loss_multiplier_d * CE_d + self.aux_loss_multiplier_y * CE_y, CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            z_q_loc, _ = self.qz(x)
            z = z_q_loc
            alpha_d = F.softmax(self.qd(z), dim=1)
            d = alpha_d.argmax(dim=1, keepdim=True)

            alpha_y = F.softmax(self.qy(z), dim=1)
            y = alpha_y.argmax(dim=1, keepdim=True)

        return d, y, alpha_d, alpha_y

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from paper_experiments.malaria.diva.pixel_cnn_utils import log_mix_dep_Logistic_256
from paper_experiments.malaria.resnet_blocks_batchnorm import *


# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        self.rn1 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn2 = nn.Upsample(8)
        self.rn3 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn4 = nn.Upsample(16)
        self.rn5 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn6 = nn.Upsample(32)
        self.rn7 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn8 = nn.Upsample(64)
        self.conv1 = nn.Conv2d(64, 100, 3, padding=1)
        self.conv2 = nn.Conv2d(100, 100, 1, padding=0)

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.zero_()

    def forward(self, zd, zx, zy):
        zdzxzy = torch.cat((zd, zx, zy), dim=1)
        h = self.fc1(zdzxzy)
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


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.LeakyReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        return zd_loc, zd_scale


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.LeakyReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.rn1 = nn.MaxPool2d(2, 2)
        self.rn2 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = nn.MaxPool2d(2, 2)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = nn.MaxPool2d(2, 2)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = nn.MaxPool2d(2, 2)

        self.fc11 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim), nn.Softplus())

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


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.rn1 = nn.MaxPool2d(2, 2)
        self.rn2 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = nn.MaxPool2d(2, 2)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = nn.MaxPool2d(2, 2)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = nn.MaxPool2d(2, 2)

        self.fc11 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim), nn.Softplus())

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
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.rn1 = nn.MaxPool2d(2, 2)
        self.rn2 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = nn.MaxPool2d(2, 2)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = nn.MaxPool2d(2, 2)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = nn.MaxPool2d(2, 2)

        self.fc11 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(64 * 4 * 4, zd_dim), nn.Softplus())

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
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7

        return zy_loc, zy_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = self.activation(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = self.activation(zy)
        loc_y = self.fc1(h)

        return loc_y


class DIVAResnetBNLinearUpsample(nn.Module):
    def __init__(self, args):
        super(DIVAResnetBNLinearUpsample, self).__init__()
        self.zd_dim = args.zd_dim
        self.zx_dim = args.zx_dim
        self.zy_dim = args.zy_dim
        self.d_dim = args.d_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.cuda()

    def forward(self, d, x, y):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        qzx = dist.Normal(zx_q_loc, zx_q_scale)
        zx_q = qzx.rsample()
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        # Prior
        zd_p_loc, zd_p_scale = self.pzd(d)
        zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(),\
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # Do standard forward pass for everything not involving y
            zd_q_loc, zd_q_scale = self.qzd(x)
            zx_q_loc, zx_q_scale = self.qzx(x)
            zy_q_loc, zy_q_scale = self.qzy(x)

            qzd = dist.Normal(zd_q_loc, zd_q_scale)
            zd_q = qzd.rsample()
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            zy_q = qzy.rsample()

            zd_p_loc, zd_p_scale = self.pzd(d)
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                                       torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()

            pzd = dist.Normal(zd_p_loc, zd_p_scale)
            pzx = dist.Normal(zx_p_loc, zx_p_scale)

            d_hat = self.qd(zd_q)
            x_recon = self.px(zd_q, zx_q, zy_q)

            CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(2)
            y_onehot = y_onehot.repeat(1, x.size()[0])
            y_onehot = y_onehot.view(2*x.size()[0], 2).cuda()

            zy_q = zy_q.repeat(2, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(2, 1), zy_q_scale.repeat(2, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)

            # Reparameterization trick
            pzy = dist.Normal(zy_p_loc, zy_p_scale)

            # Auxiliary losses
            y_hat = self.qy(zy_q)

            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1)
            qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)

            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)

            prior_y = dist.Categorical(torch.Tensor([0.12, 0.88]).cuda())
            prior_y_minus_qy = prior_y.log_prob(y_onehot[:, 1]) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * marginal_zy_p_minus_zy_q \
                   - marginal_prior_y_minus_qy \
                   + self.aux_loss_multiplier_d * CE_d

        else: # supervised
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

            CE_x = -log_mix_dep_Logistic_256(x, x_recon, average=False, n_comps=10)

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y\
                , CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_q_loc, _ = self.qzd(x)
            zd = zd_q_loc
            alpha_d = F.softmax(self.qd(zd), dim=1)
            d = alpha_d.argmax(dim=1, keepdim=True)

            zy_q_loc, _ = self.qzy.forward(x)
            zy = zy_q_loc
            alpha_y = F.softmax(self.qy(zy), dim=1)
            y = alpha_y.argmax(dim=1, keepdim=True)

        return d, y, alpha_d, alpha_y
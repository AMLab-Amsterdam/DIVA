import torch
import torch.nn.functional as F


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_mix_dep_Logistic_256(x, params, average=False, n_comps=10):

    bin_size = 1. / 255.
    logits = params[:, 0:n_comps, :, :]
    means_r = params[:, n_comps:2 * n_comps, :, :]
    means_g = params[:, 2 * n_comps:3 * n_comps, :, :] + torch.tanh(params[:, 3 * n_comps:4 * n_comps]) * x[:, 0:1,
                                                                                                          :, :]
    means_b = params[:, 4 * n_comps:5 * n_comps, :, :] + torch.tanh(params[:, 5 * n_comps:6 * n_comps]) * x[:, 0:1,
                                                                                                          :, :] + \
              torch.tanh(params[:, 6 * n_comps:7 * n_comps, :, :]) * x[:, 1:2, :, :]

    log_scale_r = torch.clamp(params[:, 7 * n_comps:8 * n_comps, :, :], min=-7.)
    log_scale_g = torch.clamp(params[:, 8 * n_comps:9 * n_comps, :, :], min=-7.)
    log_scale_b = torch.clamp(params[:, 9 * n_comps:10 * n_comps, :, :], min=-7.)

    # final size is [B, N_comps, H, W, C]
    means = torch.cat([means_r[:, :, :, :, None], means_g[:, :, :, :, None], means_b[:, :, :, :, None]], 4)
    logvars = torch.cat(
        [log_scale_r[:, :, :, :, None], log_scale_g[:, :, :, :, None], log_scale_b[:, :, :, :, None]], 4)
    # final size is [B, C, H, W, N_comps]
    means = means.transpose(4, 1)
    logvars = logvars.transpose(4, 1)
    x = x[:, :, :, :, None]

    # calculate log probs per component
    # inv_scale = torch.exp(- logvar)[:, :, :, :, None]
    inv_scale = torch.exp(- logvars)
    centered_x = x - means
    inp_cdf_plus = inv_scale * (centered_x + .5 * bin_size)
    inp_cdf_minus = inv_scale * (centered_x - .5 * bin_size)
    cdf_plus = torch.sigmoid(inp_cdf_plus)
    cdf_minus = torch.sigmoid(inp_cdf_minus)

    # bin for 0 pixel is from -infinity to x + 0.5 * bin_size
    log_cdf_zero = F.logsigmoid(inp_cdf_plus)  # cdf_plus

    # bin for 255 pixel is from x - 0.5 * bin_size till infinity
    log_cdf_one = F.logsigmoid(- inp_cdf_minus)  # 1. - cdf_minus

    # calculate final log-likelihood for an image
    mask_zero = (x.data == 0.).float()
    mask_one = (x.data == 1.).float()

    log_logist_256 = mask_zero * log_cdf_zero + (1 - mask_zero) * mask_one * log_cdf_one + \
                     (1 - mask_zero) * (1 - mask_one) * torch.log(cdf_plus - cdf_minus + 1e-7)

    # [B, H, W, n_comps]
    log_logist_256 = torch.sum(log_logist_256, 1) + F.log_softmax(logits.permute(0, 2, 3, 1), 3)

    # log_sum_exp for n_comps
    log_logist_256 = log_sum_exp(log_logist_256)

    # flatten to [B, H * W]
    log_logist_256 = log_logist_256.view(log_logist_256.size(0), -1)

    # if reduce:
    if average:
        return torch.mean(log_logist_256, 1)
    else:
        return torch.sum(log_logist_256)
        # else:
    #     return log_logist_256


def sample(x_gen):
    n_comps = 10
    logits = x_gen[:, 0:n_comps, :, :]
    sel = torch.argmax(logits,  # -
                       # torch.log(- torch.log(self.float_tensor(logits.size()).uniform_(1e-5, 1-1e-5))),
                       dim=1, keepdim=True)
    one_hot = torch.zeros(logits.size())
    if torch.cuda.is_available():
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, sel, 1.0)

    mean_x_r = torch.sum(x_gen[:, n_comps:2 * n_comps, :, :] * one_hot, 1, keepdim=True)
    # u_r = self.float_tensor(mean_x_r.size()).uniform_(1e-5, 1 - 1e-5)
    x_r = F.hardtanh(mean_x_r,  # + torch.exp(log_scale_r) * (torch.log(u_r) - torch.log(1. - u_r)),
                     min_val=0., max_val=1.)

    mean_x_g = torch.sum(x_gen[:, 2 * n_comps:3 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 3 * n_comps:4 * n_comps] * one_hot, 1, keepdim=True)) * x_r
    # u_g = self.float_tensor(mean_x_g.size()).uniform_(1e-5, 1 - 1e-5)
    x_g = F.hardtanh(mean_x_g,  # + torch.exp(log_scale_g) * (torch.log(u_g) - torch.log(1. - u_g)),
                     min_val=0., max_val=1.)

    mean_x_b = torch.sum(x_gen[:, 4 * n_comps:5 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 5 * n_comps:6 * n_comps] * one_hot, 1, keepdim=True)) * x_r + \
               torch.tanh(
                   torch.sum(x_gen[:, 6 * n_comps:7 * n_comps, :, :] * one_hot, 1, keepdim=True)) * x_g
    # u_b = self.float_tensor(mean_x_b.size()).uniform_(1e-5, 1 - 1e-5)
    x_b = F.hardtanh(mean_x_b,  # + torch.exp(log_scale_b) * (torch.log(u_b) - torch.log(1. - u_b)),
                     min_val=0., max_val=1.)

    sample = torch.cat([x_r, x_g, x_b], 1)
    return sample

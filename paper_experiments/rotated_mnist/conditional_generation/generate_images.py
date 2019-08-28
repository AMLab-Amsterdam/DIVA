import numpy as np

import torch
from torch.nn import functional as F
import torch.distributions as dist
import torch.utils.data as data_utils

from paper_experiments.rotated_mnist.dataset.mnist_loader import MnistRotated

from torchvision.utils import save_image

import random


def save_reconstructions(x):

    recon_batch = x.view(-1, 1, 28, 28, 256)
    sample = torch.zeros(8, 1, 28, 28).cuda()

    for i in range(28):
        for j in range(28):

            # out[:, :, i, j]
            probs = F.softmax(recon_batch[:, :, i, j], dim=2).data

            # Sample single pixel (each channel independently)
            for k in range(1):
                val, ind = torch.max(probs[:, k], dim=1)
                sample[:, k, i, j] = ind.squeeze().float() / 255.

    return sample


# Load model
model = torch.load('test_domain_75_le_net_first_64_seed_0.model')
model.eval()
args = torch.load('test_domain_75_le_net_first_64_seed_0.config')

args.batch_size = 8

# Set seed
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# Load supervised training
train_loader = data_utils.DataLoader(
    MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './../dataset/',
                 train=True),
    batch_size=args.batch_size,
    shuffle=True)

# Get 8 images
for batch_idx, (x, y, d) in enumerate(train_loader):
    with torch.no_grad():
        x, y, d = x.cuda(), y.cuda(), d.cuda()
        break

save_image(x.cpu(), 'input_images.png')

### Simple reconstructions
x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = model.forward(d, x, y)
x_recon = model.px(qzd.mean, qzx.mean, qzy.mean)

x_recon = save_reconstructions(x_recon)
comparison = torch.cat([x.view(8, 1, 28, 28),
                        x_recon])
save_image(comparison.cpu(),
           'simple_reconstructions.png', nrow=8)

### keep style and domain, change y
recon_list = []

for i in range(10):
    label_tensor = torch.zeros_like(y).cuda()
    label_tensor[:, i] = 1

    # forward
    zy_p_loc, zy_p_scale = model.pzy(label_tensor)

    # Reconstruct
    x_recon = model.px(qzd.mean, qzx.mean, zy_p_loc)
    x_recon = save_reconstructions(x_recon)

    recon_list.append(x_recon)

recon_list = torch.cat(recon_list)
n = 8
comparison = torch.cat([x.view(8, 1, 28, 28),
                        recon_list])
save_image(comparison.cpu(),
           'reconstruction_label.png', nrow=n)


### keep style and label, change d
recon_list = []
x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = model.forward(d, x, y)

for i in range(5):
    label_tensor = torch.zeros_like(d).cuda()
    label_tensor[:, i] = 1

    # forward
    zd_p_loc, zd_p_scale = model.pzd(label_tensor)

    # Reconstruct
    x_recon = model.px(zd_p_loc, qzx.mean, qzy.mean)
    x_recon = save_reconstructions(x_recon)

    recon_list.append(x_recon)

recon_list = torch.cat(recon_list)
n = 8
comparison = torch.cat([x.view(8, 1, 28, 28),
                        recon_list])
save_image(comparison.cpu(),
           'reconstruction_domain.png', nrow=n)

### all sample
domain_tensor = torch.zeros(8, 5).cuda()
class_tensor = torch.zeros(8, 10).cuda()

for i in range(8):
    domain_random = random.randint(0, 4)
    domain_tensor[i, domain_random] = 1

    class_random = random.randint(0, 9)
    class_tensor[i, class_random] = 1

# forward
zd_p_loc, zd_p_scale = model.pzd(domain_tensor)
zx_p_loc = torch.zeros(8, 64).cuda()
zx_p_scale = torch.ones(8, 64).cuda()
zy_p_loc, zy_p_scale = model.pzy(class_tensor)

pzd = dist.Normal(zd_p_loc, zd_p_scale)
zd_p = pzd.sample()
pzx = dist.Normal(zx_p_loc, zx_p_scale)
zx_p = pzx.sample()
pzy = dist.Normal(zy_p_loc, zy_p_scale)
zy_p = pzy.sample()

# Reconstruct
x_recon = model.px(zd_p, zx_p, zy_p)
x_recon = save_reconstructions(x_recon)

n = 8

save_image(x_recon.cpu(),
           'reconstruction_all_samples.png', nrow=n)
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision.utils import save_image

from paper_experiments.malaria.data_loader_topk import MalariaData, get_patient_ids
from paper_experiments.malaria.diva.pixel_cnn_utils import sample


def save_reconstructions(x_recon):
    # Save reconstuction
    with torch.no_grad():
        sample_t = sample(x_recon)

        return sample_t


device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': False}

# Load model
model = torch.load('../trained_diva_models_supervised/DIVA.model')

model.eval()

# add stuff back
model.qd.activation = nn.LeakyReLU()
model.qy.activation = nn.LeakyReLU()

batch_size = 14

# seeds
torch.manual_seed(1)
np.random.seed(1)

patient_ids = get_patient_ids('../dataset/', 400)
print(len(patient_ids))

train_patient_ids = patient_ids[:]
test_patient_ids = patient_ids[0]
train_patient_ids.remove(test_patient_ids)

train_dataset = MalariaData('../dataset/', domain_list=train_patient_ids)
train_size = int(0.80 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

# Get 8 images
for batch_idx, (x, y, d) in enumerate(train_loader):
    with torch.no_grad():
        x, y, d = x.cuda(), y.cuda(), d.cuda()
        break

save_image(x.cpu(), 'input_images.png')

### Simple reconstructions
x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = model.forward(d, x, y)

x_recon = model.px(qzd.mean, qzx.mean, qzy.mean)
x_recon_all = save_reconstructions(x_recon)
comparison = torch.cat([x.view(14, 3, 64, 64),
                        x_recon_all])
save_image(comparison.cpu(),
           'reconstructions.png', nrow=14)

# zd only
x_recon = model.px(qzd.mean, torch.zeros_like(qzx.mean), torch.zeros_like(qzy.mean))
x_recon_zd = save_reconstructions(x_recon)
comparison = torch.cat([x.view(14, 3, 64, 64),
                        x_recon_zd])
save_image(comparison.cpu(),
           'reconstructions_zd_only.png', nrow=14)

# zx only
x_recon = model.px(torch.zeros_like(qzd.mean), qzx.mean, torch.zeros_like(qzy.mean))
x_recon_zx = save_reconstructions(x_recon)
comparison = torch.cat([x.view(14, 3, 64, 64),
                        x_recon_zx])
save_image(comparison.cpu(),
           'reconstructions_zx.png', nrow=14)

# zy only
x_recon = model.px(torch.zeros_like(qzd.mean), torch.zeros_like(qzx.mean), qzy.mean)
x_recon_zy = save_reconstructions(x_recon)
comparison = torch.cat([x.view(14, 3, 64, 64),
                        x_recon_zy])
save_image(comparison.cpu(),
           'reconstructions_zy.png', nrow=14)

comparison = torch.cat([x.view(14, 3, 64, 64),
                        x_recon_all, x_recon_zd, x_recon_zx, x_recon_zy])
save_image(comparison.cpu(),
           'reconstructions_all_in_one_plot.png', nrow=14)

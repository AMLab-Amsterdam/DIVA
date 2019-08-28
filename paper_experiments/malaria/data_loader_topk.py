import glob

import numpy.random

import numpy as np

from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional


def get_patient_ids(path, threshold):
    files = [f for f in glob.glob(path + "**/*.png", recursive=True)]
    files = [f.split('/')[-1].split('.')[0].split('_')[0].split('thin')[0].split('Thin')[0] for f in files]

    unique_file_names = list(set(files))

    patients_with_threshold_cells = []

    for file_name in unique_file_names:
        if files.count(file_name) >= threshold:

            patients_with_threshold_cells.append(file_name)

    patients_with_threshold_cells.sort()
    return patients_with_threshold_cells


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = numpy.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = torchvision.transforms.functional.rotate(img, random_rotation*90)
        return img


class MalariaData(data_utils.Dataset):
    def __init__(self, path, domain_list=[], transform=False):
        self.path = path
        self.domain_list = domain_list
        self.transform = transform

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((64, 64), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()

        self.train_data, self.train_labels, self.train_domain = self.get_data()

    def get_cells_from_imgs(self, label_folder, domain):
        all_cells = [f for f in glob.glob(self.path + label_folder + "*.png", recursive=True)]

        cells_belonging_to_domain = []

        for cell in all_cells:
            if domain in cell:
                cells_belonging_to_domain.append(cell)

        cell_tensor_list = []
        for cell in cells_belonging_to_domain:
            with open(cell, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            cell_tensor_list.append(self.to_tensor(self.resize(img)))

        # Concatenate
        return torch.stack(cell_tensor_list)

    def get_data(self):
        cells_per_domain_list = []
        labels_per_domain_list = []
        domain_per_domain_list = []

        for i, domain in enumerate(self.domain_list):
            cells_unifected = self.get_cells_from_imgs('Uninfected/', domain)
            label_unifected = torch.zeros(cells_unifected.size()[0]) + 0

            cells_parasitized = self.get_cells_from_imgs('Parasitized/', domain)
            label_parasitized = torch.zeros(cells_parasitized.size()[0]) + 1

            cells_per_domain_list.append(torch.cat((cells_unifected, cells_parasitized), 0))
            labels_per_domain_list.append(torch.cat((label_unifected, label_parasitized), 0))
            domain_labels = torch.zeros(label_unifected.size()[0] + label_parasitized.size()[0]) + i
            domain_per_domain_list.append(domain_labels)

        # One last cat
        train_imgs = torch.cat(cells_per_domain_list)
        train_labels = torch.cat(labels_per_domain_list).long()
        train_domains = torch.cat(domain_per_domain_list).long()

        # Convert to onehot
        y = torch.eye(2)
        train_labels = y[train_labels]

        d = torch.eye(9)
        train_domains = d[train_domains]

        return train_imgs, train_labels, train_domains

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domain[index]

        if self.transform:
            x = self.to_tensor(self.rrotate(self.vflip(self.hflip(self.to_pil(x)))))

        return x, y, d


if __name__ == "__main__":
    from torchvision.utils import save_image

    kwargs = {'num_workers': 8, 'pin_memory': False}

    seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(args.seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    patient_ids = get_patient_ids('./dataset/', 400)

    train_patient_ids = patient_ids[:]
    test_patient_ids = patient_ids[seed]
    train_patient_ids.remove(test_patient_ids)

    print(test_patient_ids, train_patient_ids)

    train_dataset = MalariaData('./dataset/', domain_list=train_patient_ids, transform=True)
    train_size = int(0.80 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = data_utils.DataLoader(train_dataset, batch_size=100, shuffle=True, **kwargs)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=100, shuffle=True, **kwargs)

    # d_check = 0

    for j in range(2):

        y_array = np.zeros(2)
        d_array = np.zeros(9)

        for i, (x, y, d) in enumerate(train_loader):
            # y = y[d[:, d_check] == 1]
            # x = x[d[:, d_check] == 1]
            # d = d[d[:, d_check] == 1]

            y_array += y.sum(dim=0).cpu().numpy()
            d_array += d.sum(dim=0).cpu().numpy()

            if i == 0:
                save_image(x[:25].cpu(),
                           'sanity_check_dataloader_train_top10_second_run_' + str(j) + '.png', nrow=5)

        print(y_array, d_array)
        print('\n')

    for j in range(2):

        y_array = np.zeros(2)
        d_array = np.zeros(9)

        for i, (x, y, d) in enumerate(val_loader):
            # y = y[d[:, d_check] == 1]
            # x = x[d[:, d_check] == 1]
            # d = d[d[:, d_check] == 1]

            y_array += y.sum(dim=0).cpu().numpy()
            d_array += d.sum(dim=0).cpu().numpy()

            if i == 0:
                save_image(x[:25].cpu(),
                           'sanity_check_dataloader_val_top10_second_run_' + str(j) + '.png', nrow=5)

        print(y_array, d_array)
        print('\n')
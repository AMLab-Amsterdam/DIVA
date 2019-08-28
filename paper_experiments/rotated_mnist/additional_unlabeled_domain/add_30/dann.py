import sys
sys.path.insert(0, "./../../../")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
from torch.utils.data.dataset import random_split

from torchvision import datasets, transforms

from paper_experiments.rotated_mnist.dataset.mnist_loader_add_unsup import MnistRotated4Domains

import torch.nn as nn
from paper_experiments.warwick.dann.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 5)
        )

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)

        torch.nn.init.xavier_uniform_(self.class_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.class_classifier[4].weight)
        self.class_classifier[4].bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[4].weight)
        self.domain_classifier[4].bias.data.zero_()

    def forward(self, input_data, alpha):
        feature = self.encoder(input_data)
        feature = feature.view(-1, 1024)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = F.log_softmax(self.class_classifier(feature), dim=1)
        domain_output = F.log_softmax(self.domain_classifier(reverse_feature), dim=1)

        return class_output, domain_output


def train(data_loaders, model, optimizer, device, periodic_interval_batches, epoch, args):
    model.train()
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])

    batches_per_epoch = sup_batches

    # initialize variables to store loss values
    epoch_losses_sup = 0
    epoch_losses_unsup = 0
    epoch_class_y_loss = 0

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    len_dataloader = len(data_loaders["sup"])

    # count the number of supervised batches seen in this epoch
    ctr_unsup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_unsupervised = (i % periodic_interval_batches == 1) and ctr_unsup < unsup_batches

        # iter
        (x_sup, y_sup, d_sup) = next(sup_iter)

        _, y_sup = y_sup.max(dim=1)
        _, d_sup = d_sup.max(dim=1)

        x_sup, y_sup, d_sup = x_sup.to(device), y_sup.to(device), d_sup.to(device)

        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        class_output_sup, domain_output_sup = model(input_data=x_sup, alpha=alpha)
        err_s_label = F.cross_entropy(class_output_sup, y_sup)
        err_s_domain = F.cross_entropy(domain_output_sup, d_sup)

        epoch_class_y_loss += err_s_label

        # if not is_unsupervised:
        #     err_s_domain *= 2

        err = err_s_domain + err_s_label

        if is_unsupervised:
            (x_unsup, _, d_unsup) = next(unsup_iter)
            ctr_unsup += 1
            _, d_unsup = d_unsup.max(dim=1)
            x_unsup, d_unsup = x_unsup.to(device), d_unsup.to(device)

            _, domain_output_unsup = model(input_data=x_unsup, alpha=alpha)
            err_unsup_domain = F.cross_entropy(domain_output_unsup, d_unsup)
            err += err_unsup_domain

        err.backward()
        optimizer.step()

    return epoch_class_y_loss


def test(test_loader, model, device):
    model.eval()

    alpha = 0

    n_correct_class = 0
    n_correct_domain = 0

    with torch.no_grad():
        for batch_idx, (data, target, domain) in enumerate(test_loader):
            _, target = target.max(dim=1)
            _, domain = domain.max(dim=1)

            input_img = data.to(device)
            class_label = target.to(device)
            domain_label = domain.to(device)

            class_output, domain_output = model(input_data=input_img, alpha=alpha)

            pred_class = class_output.data.argmax(1, keepdim=True)
            n_correct_class += pred_class.eq(class_label.data.view_as(pred_class)).cpu().sum()

            pred_domain = domain_output.data.argmax(1, keepdim=True)
            n_correct_domain += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()

        accu_class = 100. * n_correct_class.numpy() / len(test_loader.dataset)
        accu_domain = 100. * n_correct_domain.numpy() / len(test_loader.dataset)

    return accu_class, accu_domain


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lamb', type=float, default=1.0, metavar='L',
                        help='weight for domain loss')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--list_train_domains', type=list, default=['0', '15', '45', '60'],
                        help='domains used during training')
    parser.add_argument('--list_test_domain', type=str, default='75',
                        help='domain used during testing')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Empty data loader dict
    data_loaders = {}

    # Load supervised training
    train_loader_sup = data_utils.DataLoader(
        MnistRotated4Domains(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed,
                             './../../dataset/',
                             train=True),
        batch_size=args.batch_size,
        shuffle=True)

    # Load test
    test_loader_sup = data_utils.DataLoader(
        MnistRotated4Domains(args.list_train_domains, [args.list_test_domain], args.num_supervised, args.seed,
                             './../../dataset/',
                             train=False),
        batch_size=args.batch_size,
        shuffle=True)

    # Load unsupervised training
    train_loader_unsup = data_utils.DataLoader(
        MnistRotated4Domains(args.list_train_domains, ['30'], args.num_supervised, args.seed, './../../dataset/',
                             train=False),
        batch_size=args.batch_size,
        shuffle=True)

    print(args.list_train_domains, ['30'], args.list_test_domain)

    data_loaders['sup'] = train_loader_sup
    data_loaders['test'] = test_loader_sup
    data_loaders['unsup'] = train_loader_unsup

    print(len(train_loader_sup.dataset))
    print(len(test_loader_sup.dataset))
    print(len(train_loader_unsup.dataset))

    # how often would a supervised batch be encountered during inference
    periodic_interval_batches = 5

    # number of unsupervised example
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        epoch_class_y_loss = train(data_loaders, model, optimizer, device, periodic_interval_batches, epoch, args)

        train_class_acc, train_domain_acc = test(data_loaders["sup"], model, device)
        # val_class_acc, val_domain_acc = test(val_loader, model, device)
        test_class_acc, test_domain_acc = test(data_loaders["test"], model, device)

        str_print = "{} epoch: loss y {}, train acc {}, test acc {}".format(epoch, epoch_class_y_loss, train_class_acc,
                                                                            test_class_acc)
        print(str_print)


if __name__ == '__main__':
    main()
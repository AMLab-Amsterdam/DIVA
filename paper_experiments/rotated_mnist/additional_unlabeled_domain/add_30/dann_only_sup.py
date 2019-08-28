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

from paper_experiments.rotated_mnist.dataset.mnist_loader import MnistRotated

import torch.nn as nn
from paper_experiments.warwick.dann.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU()
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(256, 32, bias=False), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(32, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 32, bias=False), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(32, 5),
        )

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[3].weight)
        torch.nn.init.xavier_uniform_(self.encoder[6].weight)
        torch.nn.init.xavier_uniform_(self.encoder[9].weight)
        torch.nn.init.xavier_uniform_(self.encoder[12].weight)
        torch.nn.init.xavier_uniform_(self.encoder[15].weight)

        torch.nn.init.xavier_uniform_(self.class_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.class_classifier[4].weight)
        torch.nn.init.xavier_uniform_(self.class_classifier[8].weight)
        self.class_classifier[8].bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.domain_classifier[0].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[4].weight)
        torch.nn.init.xavier_uniform_(self.domain_classifier[8].weight)
        self.domain_classifier[8].bias.data.zero_()

    def forward(self, input_data, alpha):
        feature = self.encoder(input_data)
        feature = feature.squeeze()
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = F.log_softmax(self.class_classifier(feature), dim=1)
        domain_output = F.log_softmax(self.domain_classifier(reverse_feature), dim=1)

        return class_output, domain_output


def train(args, model, device, train_loader, test_loader, optimizer):
    cuda = True

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    # training
    len_dataloader = len(train_loader)

    early_stopping_counter = 1
    max_early_stopping = 50
    best_y_acc_val = 0
    best_y_acc_train = 0

    for epoch in range(args.epochs):
        model.train()

        class_loss = 0
        domain_loss = 0

        for batch_idx, (data, target, domain) in enumerate(train_loader):
            _, target = target.max(dim=1)
            _, domain = domain.max(dim=1)

            p = float(batch_idx + epoch * len_dataloader) / args.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            inputv_img = data.to(device)
            classv_label = target.to(device)
            domainv_label = domain.to(device)

            class_output, domain_output = model(input_data=inputv_img, alpha=alpha)
            err_s_label = loss_class(class_output, classv_label)
            err_s_domain = loss_domain(domain_output, domainv_label)
            err = err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            class_loss += err_s_label
            domain_loss += err_s_domain

        train_class_acc, train_domain_acc = test(train_loader, model, device, epoch)
        # val_class_acc, val_domain_acc = test(val_loader, model, device, epoch)
        test_class_acc, test_domain_acc = test(test_loader, model, device, epoch)

        str_print = "{} epoch: loss y {}, train acc {}, test acc {}".format(epoch, err_s_label, train_class_acc, test_class_acc)
        print(str_print)

        # if val_class_acc > best_y_acc_val:
        #     early_stopping_counter = 1
        #
        #     best_y_acc_val = val_class_acc
        #
        #     torch.save(model, model.name + '.model')
        #
        # else:
        #     early_stopping_counter += 1
        #     if early_stopping_counter == max_early_stopping:
        #         break
    torch.save(args, model.name + '.config')


def test(test_loader, model, device, epoch):
    model.eval()

    alpha = 0

    len_dataloader = len(test_loader)

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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
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

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.list_test_domain = [args.list_test_domain]

    # Choose training domains
    all_training_domains = ['0', '15', '30', '45', '75']
    all_training_domains.remove(args.list_test_domain[0])
    args.list_train_domains = all_training_domains

    print(args.list_test_domain, args.list_train_domains)

    # Load supervised training
    train_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './../../dataset/',
                     train=True),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    print(len(train_loader.dataset))

    # Load test
    test_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './../../dataset/',
                     train=False),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    train(args, model, device, train_loader, test_loader, optimizer)


if __name__ == '__main__':
    main()
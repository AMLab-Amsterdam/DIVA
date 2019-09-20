import sys
sys.path.insert(0, "../../../")

import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import sklearn.metrics

from paper_experiments.malaria.data_loader_topk import MalariaData, get_patient_ids
from paper_experiments.resnet_blocks_batchnorm import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.rn1 = IdResidualConvBlockBNResize(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvBlockBNIdentity(32, 32, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvBlockBNResize(32, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvBlockBNIdentity(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvBlockBNResize(64, 64, 3, padding=1, nonlin=nn.LeakyReLU)

        self.fc1 = nn.Linear(64 * 4 * 4, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 2)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.zero_()

        self.activation = nn.LeakyReLU()

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

        h = self.activation(self.bn2(self.fc1(h)))
        h = self.fc2(h)

        return h


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        _, target = target.max(dim=1)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= len(train_loader.dataset)


def test(args, model, device, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.argmax(dim=1, keepdim=True)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)


def final_test(args, model, device, test_loader):
    model.eval()
    pred_list = []
    target_list = []
    pred_prob_list = []

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)

            output = model(data)

            alpha = F.softmax(output, dim=1)
            _, ind = torch.topk(alpha, 1)

            pred_prob_list.append(alpha[0][1].cpu().numpy())
            pred_list.append(ind[0].cpu().numpy())
            target_list.append(target.cpu().numpy())

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_list, pred_list).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(target_list, pred_list,
                                                                                   average='binary', pos_label=1)

    fpr, tpr, _ = sklearn.metrics.roc_curve(target_list, pred_prob_list)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    return accuracy*100, precision*100, recall*100, fscore*100, roc_auc*100


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                        help='weight decay')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--max_early_stopping', type=int, default=100, metavar='S',
                        help='max number of epochs without improvement')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}

    for seed in range(5):
        args.seed = seed

        # Seed everything
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        # random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(args)

        # Get all patient ids
        patient_ids = get_patient_ids('../dataset/', 400)
        print(len(patient_ids))

        train_patient_ids = patient_ids[:]
        test_patient_ids = patient_ids[0]
        train_patient_ids.remove(test_patient_ids)

        # see what happens if we remove the other pink one
        train_patient_ids.remove('C59P20')

        train_dataset = MalariaData('../dataset/', domain_list=train_patient_ids, transform=True)
        train_size = int(0.80 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = data_utils.DataLoader(
            MalariaData('../dataset/', domain_list=[test_patient_ids]),
            batch_size=1,
            shuffle=False,
            **kwargs)

        print(train_patient_ids, test_patient_ids)

        model_name = 'resnet_without_domain_7_seed_' + str(args.seed) + '_domain_0'

        # Init model and adam
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_acc_best = 0.
        early_stopping = 0

        # Train
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)

            train_acc = test(args, model, device, train_loader)
            val_acc = test(args, model, device, val_loader)
            print(epoch, 'train: {:.2f}, val: {:.2f}'.format(train_acc, val_acc))

            # early-stopping
            if val_acc >= val_acc_best:
                early_stopping = 0
                val_acc_best = val_acc

                torch.save(model, model_name + '.model')
                print('>>--model saved--<<')
                print(model_name + '.model')

            else:
                early_stopping += 1
                if early_stopping > args.max_early_stopping:
                    break

        # Load model
        model = torch.load(model_name + '.model')

        # Test
        test_acc, test_precision, test_recall, test_f1, test_auc = final_test(args, model, device, test_loader)
        print('test {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(test_acc, test_precision, test_recall, test_f1, test_auc))

import sys
sys.path.insert(0, "../../")

import argparse
import numpy as np

import torch
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import sklearn.metrics

from paper_experiments.malaria.data_loader_topk import MalariaData, get_patient_ids
from paper_experiments.malaria.dann.model_dann import DANN

# Training settings
parser = argparse.ArgumentParser(description='TwoTaskVae')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                    help='weight decay')
parser.add_argument('--domain-classifiers', type=float, default=0.001, metavar='W',
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--max_early_stopping', type=int, default=100, metavar='S',
                    help='max number of epochs without improvement')

parser.add_argument('--outpath', type=str, default='./',
                    help='where to save')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': False} if args.cuda else {}

# seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False


def train(train_loader, train_loader_unsupervised, model, optimizer, epoch, args):
    model.train()

    len_dataloader = min(len(train_loader), len(train_loader_unsupervised))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(train_loader_unsupervised)

    train_loss = 0
    i = 0
    while i < len_dataloader:
        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader

        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label, s_domain = data_source

        model.zero_grad()

        input_img = s_img
        class_label = s_label
        domain_label = s_domain

        # Convert one hot back to ints
        _, class_label = class_label.max(dim=1)
        _, domain_label = domain_label.max(dim=1)

        input_img = input_img.cuda()
        class_label = class_label.cuda()
        domain_label = domain_label.cuda()

        class_output, domain_output = model(input_data=input_img, alpha=args.domain_classifiers*alpha)
        err_s_label = F.cross_entropy(class_output, class_label)
        err_s_domain = F.cross_entropy(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _, t_domain = data_target

        input_img = t_img
        domain_label = t_domain
        _, domain_label = domain_label.max(dim=1)

        input_img = input_img.cuda()
        domain_label = domain_label.cuda()

        _, domain_output = model(input_data=input_img, alpha=args.domain_classifiers*alpha)
        err_t_domain = F.cross_entropy(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        train_loss += err
        i += 1

    train_loss /= len(train_loader.dataset)

    return train_loss


def test(args, model, device, test_loader):
    model.eval()
    correct_y = 0
    correct_d = 0

    with torch.no_grad():
        for data, target, domain in test_loader:
            data, target, domain = data.to(device), target.to(device), domain.to(device)
            target = target.argmax(dim=1, keepdim=True)
            domain = domain.argmax(dim=1, keepdim=True)

            output_y, output_d = model(data, alpha=0)
            pred_y = output_y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred_d = output_d.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_y += pred_y.eq(target.view_as(pred_y)).sum().item()
            correct_d += pred_d.eq(domain.view_as(pred_d)).sum().item()

    return 100. * correct_d / len(test_loader.dataset), 100. * correct_y / len(test_loader.dataset)


def final_test(args, model, device, test_loader):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    correct_d = 0

    pred_list_y = []
    target_list_y = []
    pred_prob_list_y = []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in test_loader:

            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_y, _ = model(xs, alpha=0)
            alpha_y = F.softmax(pred_y, dim=1)

            _, ys = ys.max(dim=1)
            pred_y = pred_y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            pred_list_y.append(pred_y[0][0].cpu().numpy())
            target_list_y.append(ys[0].cpu().numpy())
            pred_prob_list_y.append(alpha_y[0][1].cpu().numpy())

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(target_list_y, pred_list_y).ravel()
        accuracy_y = (tp * 1.0 + tn * 1.0)  / (tp * 1.0 + fp * 1.0 + fn * 1.0 + tn * 1.0)

        precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(target_list_y, pred_list_y,
                                                                                       average='binary', pos_label=1)
        fpr, tpr, _ = sklearn.metrics.roc_curve(target_list_y, pred_prob_list_y)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        return accuracy_y * 100.,  precision * 100., recall * 100., fscore * 100., roc_auc * 100.


if __name__ == "__main__":
    for domain in range(10):
        # Seed everything
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        # random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(args)

        # Train, val, test sets
        patient_ids = get_patient_ids('../dataset/', 400)
        print(len(patient_ids))

        train_patient_ids = patient_ids[:]
        test_patient_ids = patient_ids[domain]
        train_patient_ids.remove(test_patient_ids)

        print(test_patient_ids, train_patient_ids)

        train_dataset = MalariaData('../dataset/', domain_list=train_patient_ids, transform=True)
        train_size = int(0.80 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = data_utils.DataLoader(
            MalariaData('../dataset/', domain_list=[test_patient_ids]),
            batch_size=1,
            shuffle=False)

        model_name = 'dann_top10_domain_' + str(domain) + '_seed_' + str(args.seed)

        # Data loader that copies the train loader for second path of DANN
        train_loader_unsupervised = data_utils.DataLoader(train_dataset,
                                                          batch_size=args.batch_size,
                                                          shuffle=True, **kwargs)

        # Init model and adam
        model = DANN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_acc_y_best = 0.
        early_stopping = 0

        # training loop
        print('\nStart training:', args)
        for epoch in range(1, args.epochs + 1):
            # train
            avg_epoch_losses_sup = train(train_loader, train_loader_unsupervised, model, optimizer, epoch, args)

            train_acc_d, train_acc_y = test(args, model, device, train_loader)
            val_acc_d, val_acc_y = test(args, model, device, val_loader)
            print(epoch, 'train: d {:.2f}, y {:.2f}, val: d {:.2f}, y {:.2f}'.format(train_acc_d,
                                                                                     train_acc_y,
                                                                                     val_acc_d, val_acc_y))

            # early-stopping
            if val_acc_y >= val_acc_y_best:
                early_stopping = 0
                val_acc_y_best = val_acc_y

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
        print('test {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(test_acc, test_precision, test_recall, test_f1,
                                                                     test_auc))
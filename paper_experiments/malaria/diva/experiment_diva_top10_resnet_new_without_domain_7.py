import sys
sys.path.insert(0, "../../")

import argparse

import numpy as np

import torch
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils

import sklearn.metrics

from paper_experiments.malaria.data_loader_topk import MalariaData, get_patient_ids
from paper_experiments.malaria.diva.pixel_cnn_utils import sample
from paper_experiments.malaria.diva.model_diva_resnet_new_linear import DIVAResnetBNLinear


def train(train_loader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0

    for batch_idx, (x, y, d) in enumerate(train_loader):
        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)

        # if (epoch % 10 == 0) and (batch_idx == 1):
        #     save_reconstructions(model, d, x, y)

        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        train_loss += loss
        epoch_class_y_loss += class_y_loss

    train_loss /= len(train_loader.dataset)
    epoch_class_y_loss /= len(train_loader.dataset)

    return train_loss, epoch_class_y_loss


def save_reconstructions(model, d, x, y):
    # Save reconstuction
    with torch.no_grad():
        d = d[:8]
        x = x[:8]
        y = y[:8]

        x_recon, _, _, _, _, _, _, _, _, _, _, _ = model.forward(d, x, y)
        sample_t = sample(x_recon)

        comparison = torch.cat([x, sample_t])
        save_image(comparison.cpu(),
                   'reconstruction_diva_resnet_new_top10_domain_0_' + str(epoch) + '.png', nrow=8)


def test(model, classifier_fn, test_loader):
    model.eval()
    correct_y = 0
    correct_d = 0

    with torch.no_grad():
        for (xs, ys, ds) in test_loader:
            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_d, pred_y, _, _ = classifier_fn(xs)

            _, ds = ds.max(dim=1)
            correct_d += pred_d.eq(ds.view_as(pred_d)).sum().item()

            _, ys = ys.max(dim=1)
            correct_y += pred_y.eq(ys.view_as(pred_y)).sum().item()

    return 100. * correct_d / len(test_loader.dataset), 100. * correct_y / len(test_loader.dataset)


def final_test(model, classifier_fn, test_loader):
    model.eval()

    pred_list_y = []
    target_list_y = []
    pred_prob_list_y = []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in test_loader:

            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            _, pred_y, _, alpha_y = classifier_fn(xs)
            _, ys = ys.max(dim=1)

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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--max_early_stopping', type=int, default=100, metavar='S',
                        help='max number of epochs without improvement')

    # Model
    parser.add_argument('--d-dim', type=int, default=9,
                        help='number of classes')
    parser.add_argument('--x-dim', type=int, default=64 * 64 * 3,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')

    # Aux multipliers
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=75000.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=100000.,
                        help='multiplier for d classifier')
    # Beta VAE part
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')
    parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')

    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': False} if args.cuda else {}

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

        # Train, val, test sets
        patient_ids = get_patient_ids('../dataset/', 400)
        print(len(patient_ids))

        train_patient_ids = patient_ids[:]
        test_patient_ids = patient_ids[0]
        train_patient_ids.remove(test_patient_ids)

        # see what happens if we remove the other pink one
        train_patient_ids.remove('C59P20')

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

        model_name = 'diva_top10_test_domain_0_without_7_seed_' + str(args.seed)

        # Init model and adam
        model = DIVAResnetBNLinear(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_acc_y_best = 0.
        early_stopping = 0

        # training loop
        print('\nStart training:', args)
        for epoch in range(1, args.epochs + 1):
            # Ramp beta
            model.beta_d = min([args.beta_d, args.beta_d * (epoch * 1.) / args.warmup])
            model.beta_y = min([args.beta_y, args.beta_y * (epoch * 1.) / args.warmup])
            model.beta_x = min([args.beta_x, args.beta_x * (epoch * 1.) / args.warmup])

            # Train
            avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader, model, optimizer, epoch)

            # Test on train and val set
            val_acc_d, val_acc_y = test(model, model.classifier, val_loader)

            print(epoch, 'train: overall loss: {:.10}, y_loss {:.10}, val: d {:.2f}, y {:.2f}'.format(
                avg_epoch_losses_sup,
                avg_epoch_class_y_loss,
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

        # Final test
        test_acc, test_precision, test_recall, test_f1, test_auc = final_test(model, model.classifier, test_loader)
        print('test {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(test_acc, test_precision, test_recall, test_f1,
                                                                     test_auc))
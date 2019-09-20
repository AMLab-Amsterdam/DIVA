import sys
sys.path.insert(0, "../../../")

import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data_utils

import sklearn.metrics

from paper_experiments.malaria.data_loader_topk import MalariaData, get_patient_ids
from paper_experiments.malaria.diva.model_diva_resnet_new_linear import DIVAResnetBNLinear


def train(train_loader_supervised, train_loader_unsupervised, model, optimizer, periodic_interval_batches, epoch):
    model.train()
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """

    # compute number of batches for an epoch
    sup_batches = len(train_loader_supervised)
    unsup_batches = len(train_loader_unsupervised)
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = 0
    epoch_losses_unsup = 0
    epoch_class_y_loss = 0

    # setup the iterators for training data loaders
    sup_iter = iter(train_loader_supervised)
    unsup_iter = iter(train_loader_unsupervised)

    # count the number of supervised batches seen in this epoch
    ctr_unsup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_unsupervised = (i % periodic_interval_batches == 1) and ctr_unsup < unsup_batches


        # extract the corresponding batch
        if is_unsupervised:
            (x, y, d) = next(unsup_iter)
            ctr_unsup += 1

        else:
            (x, y, d) = next(sup_iter)

        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        optimizer.zero_grad()

        if is_unsupervised:
            new_loss = model.loss_function(d, x)
            epoch_losses_unsup += new_loss

        else:
            new_loss, class_y_loss = model.loss_function(d, x, y)
            epoch_losses_sup += new_loss
            epoch_class_y_loss += class_y_loss

        # print(epoch_losses_sup, epoch_losses_unsup)
        new_loss.backward()
        optimizer.step()

    # return the values of all losses
    return epoch_losses_sup, epoch_class_y_loss


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
    parser.add_argument('--unsup', type=int, default=1, metavar='US',
                        help='number of unsupervised domains')

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
        train_patient_ids_unsupervised = 'C59P20'

        train_patient_ids.remove(test_patient_ids)
        train_patient_ids.remove(train_patient_ids_unsupervised)

        print(test_patient_ids, train_patient_ids, train_patient_ids_unsupervised)

        train_dataset = MalariaData('../dataset/', domain_list=train_patient_ids, transform=True)
        train_size = int(0.80 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        train_loader_supervised = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader_supervised = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = data_utils.DataLoader(
            MalariaData('../dataset/', domain_list=[test_patient_ids]),
            batch_size=1,
            shuffle=False,
            **kwargs)

        train_loader_unsupervised = data_utils.DataLoader(MalariaData('../dataset/',
                                                                      domain_list=[train_patient_ids_unsupervised],
                                                                      transform=True),
                                                         batch_size=args.batch_size,
                                                         shuffle=True,
                                                         **kwargs)

        # calculate alpha_y
        args.aux_loss_multiplier_y = 75000 \
                                     * (len(train_loader_unsupervised.dataset)
                                     + len(train_loader_supervised.dataset)) \
                                     / len(train_loader_supervised.dataset)

        # number of unsupervised examples
        sup_num = len(train_loader_supervised.dataset)
        unsup_num = len(train_loader_unsupervised.dataset)

        # how often would a supervised batch be encountered during inference
        periodic_interval_batches = int(len(train_loader_supervised)/len(train_loader_unsupervised) + 1)

        model_name = 'diva_top10_semisupervised_test_domain_0_unlabeled_7_seed_' + str(args.seed)

        # Init model and adam
        model = DIVAResnetBNLinear(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_acc_y_best = 0.
        early_stopping = 0

        # training loop
        print('\nStart training:', args)
        for epoch in range(1, args.epochs + 1):
            model.beta_d = min([args.beta_d, args.beta_d * (epoch * 1.) / args.warmup])
            model.beta_y = min([args.beta_y, args.beta_y * (epoch * 1.) / args.warmup])
            model.beta_x = min([args.beta_x, args.beta_x * (epoch * 1.) / args.warmup])

            # train
            avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader_supervised,
                                                                 train_loader_unsupervised,
                                                                 model, optimizer, periodic_interval_batches, epoch)

            val_acc_d, val_acc_y = test(model, model.classifier, val_loader_supervised)
            print(epoch, 'val: d {:.2f}, y {:.2f}'.format(val_acc_d, val_acc_y))

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
        test_acc, test_precision, test_recall, test_f1, test_auc = final_test(model, model.classifier, test_loader)
        print('test {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(test_acc, test_precision, test_recall, test_f1,
                                                                     test_auc))

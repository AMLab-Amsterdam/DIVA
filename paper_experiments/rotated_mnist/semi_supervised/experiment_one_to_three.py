import sys
sys.path.insert(0, "./../../../")

import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data_utils

from paper_experiments.rotated_mnist.supervised.model_diva import DIVA
from paper_experiments.rotated_mnist.dataset.mnist_loader_more_unsup import MnistRotatedMoreUnsup
from paper_experiments.rotated_mnist.dataset.mnist_loader import MnistRotated


def train(data_loaders, model, optimizer, periodic_interval_batches, epoch):
    model.train()
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = 0
    epoch_losses_unsup = 0
    epoch_class_y_loss = 0

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches


        # extract the corresponding batch
        if is_supervised:
            (x, y, d) = next(sup_iter)
            ctr_sup += 1
        else:
            (x, y, d) = next(unsup_iter)

        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        optimizer.zero_grad()

        if is_supervised:
            new_loss, class_y_loss = model.loss_function(d, x, y)
            epoch_losses_sup += new_loss
            epoch_class_y_loss += class_y_loss

        else:
            new_loss = model.loss_function(d, x)
            epoch_losses_unsup += new_loss

        # print(epoch_losses_sup, epoch_losses_unsup)
        new_loss.backward()
        optimizer.step()

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup, epoch_class_y_loss


def get_accuracy(data_loader, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        # use the right data loader
        for (xs, ys, ds) in data_loader:
            # To device
            xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

            # use classification function to compute all predictions for each batch
            pred_d, pred_y = classifier_fn(xs)
            predictions_d.append(pred_d)
            actuals_d.append(ds)
            predictions_y.append(pred_y)
            actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        for pred, act in zip(predictions_d, actuals_d):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_d += (v.item() == 5)

        # calculate the accuracy between 0 and 1
        accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        for pred, act in zip(predictions_y, actuals_y):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds_y += (v.item() == 10)

        # calculate the accuracy between 0 and 1
        accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)

        return accuracy_d, accuracy_y


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")

    # Choose domains
    parser.add_argument('--list_train_domains', type=list, default=['0', '15', '30', '45', '60', '75'],
                        help='domains used during training')
    parser.add_argument('--list_test_domain', type=str, default='75',
                        help='domain used during testing')

    # Model
    parser.add_argument('--d-dim', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--x-dim', type=int, default=784,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')

    # Aux multipliers
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=14000.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.,
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
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')

    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Model name
    args.list_test_domain = [args.list_test_domain]
    print(args.outpath)
    model_name = args.outpath + 'test_domain_' + str(args.list_test_domain[0]) + '_sup_and_unsup_1_to_3_seed_' + str(
        args.seed) + '_adjusted_alpha'
    print(model_name)

    print(args.aux_loss_multiplier_y)

    # Choose training domains
    all_training_domains = ['0', '15', '30', '45', '60', '75']
    all_training_domains.remove(args.list_test_domain[0])
    args.list_train_domains = all_training_domains

    print(args.list_test_domain, args.list_train_domains)

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Empty data loader dict
    data_loaders = {}

    # Load supervised training
    train_loader_sup = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './../dataset/',
                     train=True),
        batch_size=args.batch_size,
        shuffle=True)

    if args.seed == 0:
        all_mnist_subsets = [1, 2, 3]
    elif args.seed == 1:
        all_mnist_subsets = [2, 3, 4]
    elif args.seed == 2:
        all_mnist_subsets = [3, 4, 5]
    elif args.seed == 3:
        all_mnist_subsets = [4, 5, 6]
    elif args.seed == 4:
        all_mnist_subsets = [5, 6, 7]
    elif args.seed == 5:
        all_mnist_subsets = [6, 7, 8]
    elif args.seed == 6:
        all_mnist_subsets = [7, 8, 9]
    elif args.seed == 7:
        all_mnist_subsets = [8, 9, 0]
    elif args.seed == 8:
        all_mnist_subsets = [9, 0, 1]
    elif args.seed == 9:
        all_mnist_subsets = [0, 1, 2]

    # Load unsupervised training
    train_loader_unsup = data_utils.DataLoader(
        MnistRotatedMoreUnsup(args.list_train_domains, args.list_test_domain, args.num_supervised, all_mnist_subsets, './../dataset/',
                     train=True),
        batch_size=args.batch_size,
        shuffle=True)

    data_loaders['sup'] = train_loader_sup
    data_loaders['unsup'] = train_loader_unsup

    # how often would a supervised batch be encountered during inference
    periodic_interval_batches = int(len(data_loaders['unsup'])/len(data_loaders['sup']) + 1)

    # number of unsupervised examples
    sup_num = len(data_loaders['sup']) * args.batch_size
    unsup_num = len(data_loaders['unsup']) * args.batch_size

    # setup the VAE
    model = DIVA(args).to(device)

    # setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # init
    val_total_loss = []
    val_class_err_d = []
    val_class_err_y = []

    best_loss = 1000.
    best_y_acc = 0.

    early_stopping_counter = 1
    max_early_stopping = 100

    # training loop
    print('\nStart training:', args)
    for epoch in range(1, args.epochs + 1):
        # scheduler.step()
        beta = min([args.max_beta, args.max_beta * (epoch * 1.) / args.warmup])
        model.beta_d = beta
        model.beta_y = beta
        model.beta_x = beta

        # train
        epoch_losses_sup, epoch_losses_unsup, epoch_class_y_loss = train(data_loaders, model, optimizer, periodic_interval_batches, epoch)

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = epoch_losses_sup / sup_num
        avg_epoch_losses_unsup = epoch_losses_unsup / unsup_num
        avg_epoch_class_y_loss = epoch_class_y_loss / sup_num

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup = avg_epoch_losses_sup
        str_loss_unsup = avg_epoch_losses_unsup
        str_print = "{} epoch: avg losses {}".format(epoch, "{} {}".format(str_loss_sup, str_loss_unsup))
        str_print += ", class y loss {}".format(avg_epoch_class_y_loss)

        # str_print = str(epoch)
        sup_accuracy_d, sup_accuracy_y = get_accuracy(data_loaders["sup"], model.classifier,
                                                                    args.batch_size)
        str_print += " sup accuracy d {}".format(sup_accuracy_d)
        str_print += ", y {}".format(sup_accuracy_y)

        print(str_print)

        if sup_accuracy_y > best_y_acc:
            early_stopping_counter = 1

            best_y_acc = sup_accuracy_y
            best_loss = avg_epoch_class_y_loss

            torch.save(model, model_name + '.model')

        elif sup_accuracy_y == best_y_acc:
            if avg_epoch_class_y_loss < best_loss:
                early_stopping_counter = 1

                best_loss = avg_epoch_class_y_loss

                torch.save(model, model_name + '.model')

            else:
                early_stopping_counter += 1
                if early_stopping_counter == max_early_stopping:
                    break

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                break
    torch.save(args, model_name + '.config')
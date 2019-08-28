import sys
sys.path.insert(0, "./../../../../../../")

import argparse

import torch.utils.data as data_utils

import numpy as np

import torch

from paper_experiments.rotated_mnist.dataset.mnist_loader import MnistRotated


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
    test_accuracy_y_list = []

    for i in range(10):
        model_name = 'test_domain_75_sup_only_seed_' + str(i) + '_add_60'
        model = torch.load(model_name + '.model')
        args = torch.load(model_name + '.config')

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

        # Load test
        test_loader_sup = data_utils.DataLoader(
            MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './../../../../dataset/',
                         train=False),
            batch_size=args.batch_size,
            shuffle=True)

        # Set seed
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        test_accuracy_d, test_accuracy_y = get_accuracy(test_loader_sup, model.classifier, args.batch_size)
        test_accuracy_y_list.append(test_accuracy_y)

    print(test_accuracy_y_list)
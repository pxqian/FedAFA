#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--alpha', type=float, default=0.2, help="Dirichlet distribution ")
    parser.add_argument('--c', type=float, default=0.6, help="Dirichlet distribution ")
    parser.add_argument('--p', type=float, default=0.5, help="Dirichlet distribution ")

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet32', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--rebuild', action='store_true', help="rebuild train data")
    parser.add_argument('--struct', action='store_true', help="intermediate or raw data in gate model")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")

    parser.add_argument('--imb_factor', type=float, default=0.01, help='imbalanced control')
    parser.add_argument('--gamma', type=int, default=2, help='focal loss')

    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_meta', type=int, default=0, help="number \
                                    of meta data per class")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--debug', action='store_true', help='no runs event')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--embedding_dim', type=int, default=64, help="the dim of label embedding")

    args = parser.parse_args()
    return args

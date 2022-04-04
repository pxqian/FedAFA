import numpy as np
import random
from options import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from collections import defaultdict
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_img_num_per_cls(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def get_train_data(train_dataset,dset,args):
    img_num_list = get_img_num_per_cls(dset, args.imb_factor, args.num_meta * args.num_classes)
    # random.shuffle(img_num_list)
    data_list_val = {}
    for j in range(args.num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]
    print('img_num_list:{},length:{}'.format( img_num_list,len(img_num_list)))
    im_data=[]
    for cls,img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        train_data=np.random.choice(img_id_list,img_num_list[cls],replace=False)
        im_data.extend(train_data)
        data_list_val[cls]=list(train_data)
    random.shuffle(im_data)
    return im_data,data_list_val,img_num_list

def Divide_groups(dataset_train,train_list,dict_per_cls,num_users,args):
    class_num=np.arange(args.num_classes)
    imb_cls_num=list(np.random.randint(2,args.num_classes+1,(args.num_users)))
    num_per_user=int(len(train_list)/args.num_users)
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    num_cls = 0
    while(num_cls<args.num_users):
        user_set = set()
        while(len(user_set)<num_per_user):
            user_cls = np.random.choice(class_num, imb_cls_num[num_cls], replace=False)
            for j in range(len(user_cls)):
                user_set = user_set | dict_per_cls[user_cls[j]]
        dict_users[num_cls] = np.random.choice(np.array(list(user_set)), num_per_user, replace=False)
        random.shuffle(dict_users[num_cls])
        for m in range(len(user_cls)):
            dict_per_cls[user_cls[m]]-=user_set&set(dict_users[num_cls])
        num_cls+=1

    user_class_weight = []
    for i in range(args.num_users):

        image_trainset_weight = np.zeros(args.num_classes)
        for label in np.array(dataset_train.targets)[dict_users[i]]:
            image_trainset_weight[label] += 1

        image_num = image_trainset_weight
        image_trainset_weight = image_trainset_weight / image_trainset_weight.sum()
        user_class_weight.append(image_num)


    return dict_users,user_class_weight

def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_oversample(args,train_cifar,user_idx,user_class,transform_train):

    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)
    user_loader=[]

    for i in range(len(user_idx)):
        user_cifar=datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        user_cifar.targets=targets[user_idx[i]]
        user_cifar.data=train_cifar.data[user_idx[i]]
        assert len(train_cifar.targets) == len(train_cifar.data)
        train_in_idx = get_oversampled_data(user_cifar, user_class[i])
        train_in_loader = DataLoader(user_cifar, batch_size=args.local_bs,
                                     sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=0)
        user_loader.append(train_in_loader)


    return user_loader


def cifar_noniid( no_participants, cifar_classes,img_num_list,args,alpha):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
        """
    # np.random.seed(666)
    # random.seed(666)

    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    datasize = {}
    for n in range(no_classes):
        sampled_probabilities = img_num_list[n] * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = min(len(cifar_classes[n]), no_imgs)
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i, j] for j in range(10)])
    # clas_weight = np.zeros((no_participants, 10))
    # for i in range(no_participants):
    #     for j in range(args.num_classes):
    #         clas_weight[i, j] = float(datasize[i, j]) / float((train_img_size[i]))
    #

    class_num={i:np.zeros(args.num_classes) for i in range (no_participants)}
    class_num1=np.zeros([no_participants,args.num_classes])

    for i in range(no_participants):
        for j in range(args.num_classes):
            class_num[i][j]=datasize[i, j]
            class_num1[i][j]=int(datasize[i, j])

    # for  i in range(len(grad1[0])):
    # sns.heatmap(np.transpose(class_num1),cmap='Reds',vmin=0, vmax=class_num1.max())

    # plt.matshow(np.transpose(class_num1), cmap='RdBu',  vmin=-class_num1.max(), vmax=class_num1.max())
    # plt.colorbar(0)
    # plt.savefig('./cifar10_alpha{}.svg'.format(args.alpha))


    return per_participant_list, class_num,train_img_size

def Divide_groups(dataset_train,train_list,dict_per_cls,num_users,args,m):
    class_num=np.arange(args.num_classes)
    imb_cls_num=list(np.random.randint(2,m,(args.num_users)))
    num_per_user=int(len(train_list)/args.num_users)
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    num_cls = 0
    while(num_cls<args.num_users):
        user_set = set()
        while(len(user_set)<num_per_user):
            user_cls = np.random.choice(class_num, imb_cls_num[num_cls], replace=False)
            for j in range(len(user_cls)):
                user_set = user_set | dict_per_cls[user_cls[j]]
        dict_users[num_cls] = np.random.choice(np.array(list(user_set)), num_per_user, replace=False)
        random.shuffle(dict_users[num_cls])
        for m in range(len(user_cls)):
            dict_per_cls[user_cls[m]]-=user_set&set(dict_users[num_cls])
        num_cls+=1

    return dict_users



import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from sampling import  get_train_data,Divide_groups
from options import args_parser
from Update import *
from util import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset

import time
from sampling import get_oversample,cifar_noniid
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
if __name__ == '__main__':
    args = args_parser()

    args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)
    args.imb_factor = 0.01
        # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/fedfea/local{}_c{}_p{}_alpha{}_localbs{}_lr{}_{}'.format(args.dataset, args.c,args.p, args.alpha,
                                                                        args.local_bs, args.lr,
                                                                        current_time)
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    TAG1 = 'exp/feaperacc/local{}_c{}_p{}_epoch{}_localbs{}_lr{}_{}'.format( args.dataset,args.c,args.p,
                                                                                    args.epochs,
                                                                                    args.local_bs,args.lr,
                                                                                    current_time)
    logdir1 = f'runs/{TAG1}' if not args.debug else f'runs2/{TAG1}'
    writer1 = SummaryWriter(logdir1)


    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    normalizer = InputNormalize(mean, std).cuda()


    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),

        ])
    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    train_group, dict_per_cls, img_num_list = get_train_data(train_dataset, args.dataset, args)
    dict_users, image_num ,dict_user_num= cifar_noniid(args.num_users, dict_per_cls,img_num_list,args,args.alpha)
        # dict_users,image_num = Divide_groups(train_dataset, train_groups, dict_per_cls, args.num_users, args)
    user_loader=get_oversample(args,train_dataset,dict_users,image_num,transform_train)



    model = models.__dict__['resnet32'](args.num_classes).cuda()
    model.train()


    w_glob = \
            torch.load(f'./a_0.2.pkl')['model']


    model.load_state_dict(w_glob)

    paramlist=[]
    for name, param in model.named_parameters():
        paramlist.append(name)
    loss_train=[]
    w_epoch = []
    test_best_acc=0

    for epoch in range(1):
        w_locals, loss_locals = [], []
        testacc_locals=[]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users=np.arange(args.num_users)
        per_acc = np.zeros([args.num_users, args.local_ep])
        per_class_acc = np.zeros([args.num_users, args.num_classes])
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])

            w, loss, user_test, sucess = local_model.train_fea(model=copy.deepcopy(model).to(args.device), idx=idx,
                                                                       train_loader=user_loader[idx],
                                                                       test_loader=test_loader,
                                                                       epoch=epoch, user_class=image_num[idx],
                                                                       paramlist=paramlist,
                                                                       writer=writer1,TAG=TAG1,per_acc=per_acc,per_class_acc=per_class_acc)
            w_locals.append(w)
            loss_locals.append(loss)
            testacc_locals.append(user_test['user_acc'])



            # update global weights
        w_glob = FedAvg(w_locals, dict_user_num, idxs_users)
        model.load_state_dict(w_glob)
        peracc_avg = per_acc.sum(0) / 20
        print("finalacc{}_c{}_p{}_alpha{}".format(peracc_avg,args.c,args.p,args.alpha))
        per_class_acc_avg = per_class_acc.sum(0) / 20
        print("finalacc{}_c{}_p{}_alpha{}".format(per_class_acc_avg, args.c, args.p, args.alpha))
            # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        testacc_avg = sum(testacc_locals) / len(testacc_locals)

        print('Round {:3d}, Train loss {:.3f}'.format(epoch + 1, loss_avg))
        print('Round {:3d}, per Test acc {:.3f}'.format(epoch + 1, testacc_avg))
        writer.add_scalar('per_test_acc', testacc_avg, epoch + 1)

        loss_train.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg, epoch + 1)
        test_acc, test_loss = test_img(model, train_dataset, args)

        writer.add_scalar('test_loss', test_loss, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)


        # testing
    model.eval()

    writer.close()


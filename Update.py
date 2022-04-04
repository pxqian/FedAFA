#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from options import args_parser

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy

from util import sum_t,random_perturb,make_step,InputNormalize
from torch.utils.tensorboard import SummaryWriter
import scipy
import models.resnet32
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
mean = torch.tensor([0.4914, 0.4822, 0.4465])
std = torch.tensor([0.2023, 0.1994, 0.2010])
normalizer = InputNormalize(mean, std).cuda()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,alpha=None, size_average=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
    def train_fea(self, model,idx,train_loader,test_loader,epoch,user_class,paramlist,writer,TAG,per_acc,per_class_acc):
        # idx = 19
        user_class_t = torch.Tensor(user_class).cuda()
        net_train = models.__dict__['resnet32'](self.args.num_classes).cuda()
        net_train.load_state_dict(model.state_dict())
        net_train.train()

        model.eval()
        layer=0
        for name, param in net_train.named_parameters():
            if (layer <= 62):
                layer += 1
                param.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net_train.parameters()),self.args.lr, momentum = self.args.momentum,
        nesterov = self.args.nesterov,
        weight_decay = self.args.weight_decay)

        train_acc=torch.zeros(2)
        epoch_loss = []
        # test_stats = evaluate(self.args, net_train, test_loader, user_class, idx, writer, epoch)

        for ep in range(self.args.local_ep):
            batch_loss = []
            t_success = torch.zeros(2)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                gen_probs = user_class_t[labels] / user_class_t.max()

                gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()
                while (len(gen_index) == 0):  # Generation index
                    gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()

                gen_index = gen_index.view(-1)

                gen_targets = labels[gen_index]
                bs = user_class_t[labels].repeat(gen_index.size(0), 1)
                gs = user_class_t[gen_targets].view(-1, 1)
                delta = F.relu(bs - gs)
                p_accept = 1 - delta/user_class_t.max()
                mask_valid = (p_accept.sum(1) > 0)
                gen_index = gen_index[mask_valid]
                gen_targets = gen_targets[mask_valid]
                p_accept = p_accept[mask_valid]

                select_idx = torch.multinomial(p_accept, 1, replacement=True).view(-1)
                if (len(select_idx) > 0):
                    p_accept = p_accept.gather(1, select_idx.view(-1, 1)).view(-1)
                    gen_inputs, gen_labels, gen_index_c, gen_f, other_idx = train_net(model, self.loss_func, images,
                                                                                      labels, gen_index, gen_targets,
                                                                                      select_idx, p_accept, self.args)

                    log_probs, f = net_train(normalizer(gen_inputs), f=None)
                    gen_f[other_idx] = f[other_idx]
                    probs, f = net_train(normalizer(images), f=gen_f)
                    tprobs, tf = net_train(normalizer(images), f=None)

                    loss0 = F.cross_entropy(probs, gen_labels)
                    loss1 = F.cross_entropy(tprobs, labels)
                    loss = self.args.c*loss0 + (1-self.args.c) * loss1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                    predicted = probs[:, :self.args.num_classes].max(1)[1]
                    t_success[0] += np.count_nonzero(predicted[gen_index_c].cpu() == gen_labels[gen_index_c].cpu())
                    t_success[1] += len(gen_index_c)
                    train_acc[0] += np.count_nonzero(predicted.cpu() == gen_labels.cpu())
                    train_acc[1] += len(labels)
                else:
                    tprobs, tf = net_train(normalizer(images), f=None)
                    net_train.zero_grad()
                    loss = F.cross_entropy(tprobs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            test_stats, class_acc = evaluate(self.args, net_train, test_loader, user_class, idx, writer, epoch)
            per_acc[idx][ep] = test_stats['user_acc']
            per_class_acc[idx] = class_acc



            print('User:{},train_acc:{}'.format(idx,train_acc[0]/train_acc[1]))
            print('attack success:{},{}/{}\n '.format(t_success[0]/t_success[1],t_success[0],t_success[1]))
            writer.add_scalar('user'+str(idx)+' train_loss', sum(batch_loss)/len(batch_loss), ep + 1)
            writer.add_scalar('user'+str(idx)+' train_acc', train_acc[0]/train_acc[1], ep + 1)
            writer.add_scalar('user'+str(idx)+' test_loss', test_stats['user_loss'], ep + 1)
            writer.add_scalar('user'+str(idx)+' test_acc', test_stats['user_acc'], ep+ 1)
        return net_train.state_dict(),sum(epoch_loss) / len(epoch_loss),test_stats,t_success


def train_net(model_gen,criterion,inputs_orig, targets_orig, gen_idx, gen_targets,select_idx,p_accept,args):
    batch_size = inputs_orig.size(0)
    inputs = inputs_orig.clone()
    targets = targets_orig.clone()

    seed_targets = targets_orig[select_idx]
    seed_images = inputs_orig[select_idx]


    gen_f,correct_mask,gen_inputs=generation(model_gen,seed_images, seed_targets, gen_targets,p_accept,args)
    num_gen = sum_t(correct_mask)
    num_others = batch_size - num_gen
    gen_c_idx = gen_idx[correct_mask]
    others_mask = torch.ones(batch_size, dtype=torch.bool).cuda()
    others_mask[gen_c_idx] = 0
    others_idx = others_mask.nonzero().view(-1)
    feature = torch.zeros(batch_size,16,32,32).cuda()


    if num_gen > 0:
        gen_inputs_c = gen_inputs[correct_mask]
        gen_targets_c = gen_targets[correct_mask]
        gen_fea_c=gen_f[correct_mask]

        inputs[gen_c_idx] = gen_inputs_c
        targets[gen_c_idx] = gen_targets_c
        feature[gen_c_idx]=gen_fea_c

    return inputs,targets,gen_c_idx,feature,others_idx


def  generation(model_g, inputs, seed_targets, targets,p_accept,args):

    model_g.eval()


    outputs_g,f_orig = model_g(normalizer(inputs),f=None)
    f=f_orig.clone()
    r=(f.max().item()+f.min().item())/2
    random_noise = random_perturb(f, 'l2', r,1)
    f = torch.clamp(f + random_noise, 0, f.max().item())
    for _ in range(10):
        outputs_g, f = model_g(normalizer(inputs),f)
        loss = F.cross_entropy(outputs_g, targets)
        grad, = torch.autograd.grad(loss, [f])
        f = f - make_step(grad, 'l2', 0.5)
        f = torch.clamp(f , 0, f.max().item())
    outputs_g, _ = model_g(normalizer(inputs),f)
    fea=f.detach()
    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]
    correct = (probs_g >= args.p).byte().cuda()
    return fea,correct,inputs


def evaluate(args,net, dataloader,class_weight,idx,writer,ep):
    is_training = net.training
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct_class = np.zeros(args.num_classes)
    class_loss = np.zeros(args.num_classes)
    correct_class_acc = np.zeros(args.num_classes)
    class_loss_avg = np.zeros(args.num_classes)
    correct_class_size = np.zeros(args.num_classes)
    correct = 0.0
    # dataset_size = len(dataset)
    total_loss = 0.0
    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, target = inputs.to(args.device), targets.to(args.device)

        log_probs, _= net(normalizer(inputs),f=None)
        loss = nn.CrossEntropyLoss(reduction='none')(log_probs, target)
        total_loss += loss.sum().item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        for i in range(args.num_classes):
            class_ind = target.data.view_as(y_pred).eq(i * torch.ones_like(y_pred))
            correct_class_size[i] += class_ind.cpu().sum().item()
            correct_class[i] += (y_pred.eq(target.data.view_as(y_pred)) * class_ind).cpu().sum().item()
            class_idx = torch.reshape(class_ind.float(), (len(target),))
            class_loss[i] += (loss * class_idx).cpu().sum().item()

    for i in range(args.num_classes):
        correct_class_acc[i] = 100*(float(correct_class[i]) / float(correct_class_size[i]))
        class_loss[i] = (loss * class_idx).cpu().sum().item()/ float(correct_class_size[i])

    weight=np.zeros(args.num_classes)
    class_idx=class_weight.nonzero()
    weight[class_idx] = 1
    class_acc=correct_class_acc*weight
    weight=weight/weight.sum()
    user_acc = correct_class_acc * weight
    user_loss = class_loss * weight
    # for ind in class_idx[0]:
    #     writer.add_scalars('user'+str(idx),{'class'+str(ind):class_correct[ind]*100}, epoch + 1)

    results = {
        'global_acc': 100. * correct / len(dataloader.dataset),
        'user_acc': user_acc.sum(),
        'user_loss': user_loss.sum(),


    }

    msg = 'test Global_Acc: %.3f%%  | test  Local ACC: %.3f%%  |  test Local  Loss: %.3f \n'  % \
          (
             results['global_acc'], results['user_acc'],results['user_loss']
          )

    print(msg)

    net.train(is_training)
    return results,class_acc


def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs ,_= net_g(normalizer(data),f=None)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct.item() / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def FedAvg(w,user_num,idx_users):
    total_data_points = user_num[idx_users].sum()
    fed_avg_freqs=user_num[idx_users]/ total_data_points
    w_avg=copy.deepcopy(w[0])
    for net_id in range(len(w)):
        net_para = w[net_id]
        if(net_id==0):
            for key in net_para:
                w_avg[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                w_avg[key] += net_para[key] * fed_avg_freqs[net_id]
    return w_avg


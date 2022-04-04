import torch.backends.cudnn
import torch.cuda
import numpy as np
import random
import torch.nn as nn
import os
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+123)
    np.random.seed(seed+1234)
    random.seed(seed+12345)
    torch.backends.cudnn.deterministic = True


def add_scalar(writer, user_num, test_result, epoch):
    test_loss, test_acc, user_loss, user_acc = test_result
    writer.add_scalar(f'user_{user_num}/global/test_loss', test_loss, epoch)
    writer.add_scalar(f'user_{user_num}/global/test_acc', test_acc, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_loss', user_loss, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_acc', user_acc, epoch)




def sum_t(tensor):
    return tensor.float().sum().item()

def random_perturb(inputs, attack,r, eps):
    if attack == 'inf':
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    else:
        r_inputs = (torch.rand_like(inputs) - r).renorm(p=2, dim=1, maxnorm=eps)
    return r_inputs

def make_step(grad, attack, step_size):
    if attack == 'l2':
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_grad = grad / (grad_norm + 1e-10)
        step = step_size * scaled_grad
    elif attack == 'inf':
        step = step_size * torch.sign(grad)
    else:
        step = step_size * grad
    return step

class InputNormalize(nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None].cuda()
        new_mean = new_mean[..., None, None].cuda()

        # To prevent the updates the mean, std
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized
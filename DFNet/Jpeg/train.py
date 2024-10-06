# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from JDFNet import FuNet
from torch.backends import cudnn
from time import *
import utilmat
import os
from logger import get_logger
#from torch.nn.parallel import DataParallel

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

# cuda related
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


valid_cover_path = "data/val/QF75/J-UNI0.4/0/"
valid_stego_path = "data/val/QF75/J-UNI0.4/1/"
train_cover_path = "data/train/QF75/cover/"
train_stego_path = "data/train/QF75/J-UNI0.4/"

print('torch ', torch.__version__)
print('train_path = ', train_cover_path)
print('train_batch_size = ', args.batch_size)
print('test_batch_size = ', args.test_batch_size)

train_transform = transforms.Compose([utilmat.AugData(), utilmat.ToTensor()])  # utils.AugData(),
train_data = utilmat.DatasetPair(train_cover_path, train_stego_path, train_transform)
valid_transform = transforms.Compose([utilmat.ToTensor()])
valid_data= utilmat.DatasetPair(valid_cover_path,valid_stego_path,valid_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = FuNet()
if args.cuda:
    model.cuda()
    #model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
cudnn.benchmark = True



def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')


model.apply(initWeights)

params = model.parameters()
params_wd, params_rest = [], []
for param_item in params:
    if param_item.requires_grad:
        (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

param_groups = [{'params': params_wd, 'weight_decay': args.weight_decay},
                {'params': params_rest}]


optimizer = optim.Adamax(param_groups, lr=0.001, betas=(0.9, 0.999))  # adjust beta1 to momentum
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=210, gamma=0.1)



def train(epoch):
    total_loss = 0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(lr_train)
    model.train()

    for batch_idx, data in enumerate(train_loader):

        if args.cuda:
            data, label = data['images'].cuda(), data['labels'].cuda()
        data, label = Variable(data), Variable(label)

        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(os.listdir(train_cover_path)) - args.batch_size * (len(train_loader) - 1)
            datas = data.view(last_batch_size * 2, 1, 256, 256)
            labels = label.view(last_batch_size * 2)
        else:
            datas = data.view(args.batch_size * 2, 1, 256, 256)
            labels = label.view(args.batch_size * 2)
        optimizer.zero_grad()
        output = model(datas)
        # print('output = ',output)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()

            b_accu = b_correct / (labels.size(0))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), b_accu, loss.item()))
    logger.info('train Epoch: {}\tavgLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))
    scheduler.step()
    # writer.add_scalar('Train_loss', loss ,epoch)


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            if args.cuda:
                data, target = data['images'].cuda(), data['labels'].cuda()
            data, target = Variable(data), Variable(target)
            data = data.view(10 * 2, 1, 256, 256)
            target = target.view(10 * 2)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            # print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)*2
    logger.info('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset)*2,
        100. * correct / (len(valid_loader.dataset) * 2)))
    accu = float(correct) / len(valid_loader.dataset) * 2
    return accu, valid_loss


t1 = time()
logger = get_logger('J-UNI0.4/QF75/jdfnet.log')
logger.info('start training!')
'''
a ='J-UNI0.2/QF95/funet/174.pkl'

state_dict = torch.load(a)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    if not key.startswith('module.'):
        new_key = 'module.' + key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
'''

for epoch in range(1, args.epochs + 1):  #
    #torch.cuda.empty_cache()
    train(epoch)
    torch.save(model.state_dict(), 'J-UNI0.4/QF75/jdfnet/' + str(epoch) + '.pkl', _use_new_zipfile_serialization=False)
    valid()




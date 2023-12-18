import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit
from utils.data_loader import *
from model.ResNet import *
import argparse

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        with torch.no_grad():
          x_var = Variable(x.type(torch.cuda.FloatTensor))


        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--model', type=str, default='Res20')
parser.add_argument('--data_augment', type=str, default='baseline')
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                           transform=T.ToTensor()) #TAG

cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                           transform=T.ToTensor())

cifar10_test = dset.CIFAR10('./dataset', train=False, download=True,
                          transform=T.ToTensor())

train_dataset_size = len(cifar10_train)
NUM_TRAIN = int(train_dataset_size * 0.98)
NUM_VAL = train_dataset_size - NUM_TRAIN

loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
loader_test = DataLoader(cifar10_test, batch_size=64)
model = resnet20().to(args.device)

# 加载之前保存的模型权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))

# 将模型设置为评估模式
model.eval()
check_accuracy(model, loader_test)
print(NUM_TRAIN)
print(NUM_VAL)
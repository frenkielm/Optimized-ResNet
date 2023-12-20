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
          x_var = Variable(x).to(args.device)


        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Res20')
parser.add_argument('--data_augment', type=str, default='baseline')
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)

if(args.model=='Res20'):
    model = resnet20().to(args.device) 
elif(args.model=='Res32'):
    model = resnet32().to(args.device)
elif(args.model=='Res44'):
    model = resnet44().to(args.device) 
elif(args.model=='Res56'):
    model = resnet56().to(args.device)  
else:
    raise ValueError("Unsupported model: {}".format(args.model))

# 加载之前保存的模型权重
loc = "cuda:0" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load('checkpoint/Res20.pt',map_location=loc))

# 将模型设置为评估模式
model.eval()
check_accuracy(model, loader_test)
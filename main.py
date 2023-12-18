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


## 设立随机种子
import random
seed = 2138
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


## 定义超参数

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--model', type=str, default='Res20')
parser.add_argument('--data_augment', type=str, default='baseline')
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## 加载数据集

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



## 实例化模型
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
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)




## 开始训练
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
accuracy_val = []
valid_loss_min = 1000
counter = 0
lr = args.learning_rate
for epoch in range(args.epochs):
    model.train()
    train_epoch_loss = []
    idx = 0
    for data_x,data_y in loader_train:
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = loss_fn(outputs,outputs)
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx%(len(loader_train)//2)==0:
            print("epoch={}/{},{}/{}of train, loss={}".format(
                epoch, args.epochs, idx, len(loader_train),loss.item()))
        idx += 1
    train_epochs_loss.append(np.average(train_epoch_loss))
    
    #=====================valid============================
    total_sample = 0
    right_sample = 0
    model.eval()
    valid_epoch_loss = []
    for idx,(data_x,data_y) in enumerate(loader_val):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)
        outputs = model(data_x)
        loss = loss_fn(outputs, data_y)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())
        _, preds = torch.max(outputs, 1)
        right_sample += (preds == data_y).sum()
        total_sample += preds.size(0)
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy_val.append(right_sample/total_sample)
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    
    #====================save model=======================
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1
    #====================adjust lr========================
    if counter==10:
        counter =0
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
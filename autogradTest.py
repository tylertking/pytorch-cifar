import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *


class Batchnorm():

    def __init__(self,X_dim):
        self.d_X, self.h_X, self.w_X = X_dim
        self.gamma = np.ones((1, int(np.prod(X_dim)) ))
        self.beta = np.zeros((1, int(np.prod(X_dim))))
        self.params = [self.gamma,self.beta]

    def forward(self,X):
        self.n_X = X.shape[0]
        self.X_shape = X.shape
        
        self.X_flat = X.ravel().reshape(self.n_X,-1)
        self.mu = np.mean(self.X_flat,axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta
        
        return out.reshape(self.X_shape)

    def backward(self,dout):

        dout = dout.ravel().reshape(dout.shape[0],-1)
        X_mu = self.X_flat - self.mu
        var_inv = 1./np.sqrt(self.var + 1e-8)
        
        dbeta = np.sum(dout,axis=0)
        dgamma = dout * self.X_norm

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (self.var + 1e-8)**(-3/2)
        dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/self.n_X * np.sum(-2.* X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / self.n_X) + (dvar * 2/self.n_X * X_mu)
        
        dX = dX.reshape(self.X_shape)
        return dX, [dgamma, dbeta]




print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)


m = Batchnorm([20, 100, 35, 45])
input = torch.randn(20, 100, 35, 45)
output = m(input)



a = torch.tensor([2., 3., 30.], requires_grad=True)
b = torch.median(a)
c = torch.mean(a)
# c = torch.full((1, 2), int(b.detach().numpy().tolist()))
# x = a - b
print(b)
print(c)
b.backward()
c.backward
print(b)
print(c)

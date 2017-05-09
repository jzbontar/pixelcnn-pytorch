#! /usr/bin/env python

import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
backends.cudnn.benchmark = True

def tic():
    cuda.synchronize()
    return time.time()

def toc(t):
    cuda.synchronize()
    return round(time.time() - t, 1)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:,:,kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:,:,kH // 2 + 1:] = 0

    def forward(self, x):
        # self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


fm = 32
num_layers = 5
action = 'train'

layers = []
for i in range(num_layers):
    if i == 0:
        layers.append(MaskedConv2d('A',  1, fm, 7, 1, 3, bias=False))
    else:
        layers.append(MaskedConv2d('B', fm, fm, 3, 1, 1, bias=False))
    layers.extend([nn.InstanceNorm2d(fm), nn.ReLU(True)])
layers.append(MaskedConv2d('B', fm, 256, 3, 1, 1))
net = nn.Sequential(*layers)
net.cuda()

tr = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

if action == 'train':
    # train
    optimizer = optim.Adam(net.parameters())
    for epoch in range(10):

        err = []
        tr_time = tic()
        for input, _ in tr:
            bs, _, w, h = input.size()
            input = Variable(input.cuda(async=True))
            target = Variable((input.data * 255).long())
            loss = F.cross_entropy(net(input), target.view(bs, w, h))
            err.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tr_time = toc(tr_time)

        # test
        input = torch.Tensor(32, 1, 28, 28).cuda()
        input.fill_(0)
        out = net(Variable(input, volatile=True))
        print F.softmax(out[0,:,10,10])

        print epoch, np.mean(err), tr_time

        torch.save(net.state_dict(), 'net.pt')

if action == 'sample':
    # sample
    net.load_state_dict(torch.load('net.pt'))

    input = torch.Tensor(32, 1, 28, 28).cuda()
    input.fill_(0)

    t = tic()
    for i in range(28):
        for j in range(28):
            out = net(Variable(input, volatile=True))
            print F.softmax(out[0,:,10,10])
            sys.exit()
    print toc(t)

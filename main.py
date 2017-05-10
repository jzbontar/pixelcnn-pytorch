#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils

parser = argparse.ArgumentParser()
parser.add_argument('action', choices={'train', 'sample'})
parser.add_argument('--feature-maps', type=int, default=32)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.01)
args = parser.parse_args()
backends.cudnn.benchmark = True

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
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


layers = [MaskedConv2d('A',  1, args.feature_maps, 7, 1, 3), nn.ReLU(True)]
for i in range(args.num_layers):
    layers.extend([MaskedConv2d('B', args.feature_maps, args.feature_maps, 7, 1, 3), nn.ReLU(True)])
layers.append(MaskedConv2d('B', args.feature_maps, 256, 7, 1, 3))
net = nn.Sequential(*layers)
net.cuda()

tr = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

if args.action == 'train':
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    for epoch in range(10):
        err = []
        for input, _ in tr:
            bs, _, w, h = input.size()
            input = Variable(input.cuda(async=True))
            target = Variable((input.data * 255).long())
            loss = F.cross_entropy(net(input), target.view(bs, w, h))
            err.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print epoch, np.mean(err)
        torch.save(net.state_dict(), 'net.pt')

if args.action == 'sample':
    net.load_state_dict(torch.load('net.pt'))
    sample = torch.Tensor(64, 1, 28, 28).cuda()
    sample.fill_(0)
    for i in range(28):
        for j in range(28):
            out = net(Variable(sample, volatile=True))
            probs = F.softmax(out[:,:,i,j]).data
            sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.
    utils.save_image(sample, 'sample.png')

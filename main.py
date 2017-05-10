#! /usr/bin/env python

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

fm = 32
net = nn.Sequential(
    MaskedConv2d('A', 1,   fm, 7, 1, 3), nn.ReLU(True),
    MaskedConv2d('B', fm,  fm, 7, 1, 3), nn.ReLU(True),
    MaskedConv2d('B', fm,  fm, 7, 1, 3), nn.ReLU(True),
    MaskedConv2d('B', fm,  fm, 7, 1, 3), nn.ReLU(True),
    MaskedConv2d('B', fm,  fm, 7, 1, 3), nn.ReLU(True),
    MaskedConv2d('B', fm, 256, 7, 1, 3))
net.cuda()

tr = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
sample = torch.Tensor(144, 1, 28, 28).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
for epoch in range(50):
    # train
    err = []
    for input, _ in tr:
        input = Variable(input.cuda(async=True))
        target = Variable((input.data[:,0] * 255).long())
        loss = F.cross_entropy(net(input), target)
        err.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # sample
    sample.fill_(0)
    for i in range(28):
        for j in range(28):
            out = net(Variable(sample, volatile=True))
            probs = F.softmax(out[:, :, i, j]).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
    utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12)

    print epoch, np.mean(err)

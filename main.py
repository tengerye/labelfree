#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets import CushionDataset, EmbedBatch_collate_fn
from networks import ConstNet, HeightNet
from trainer import train_model
import params



__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017-2019"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    return parser.parse_args()



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embedding_net = HeightNet()
    net = ConstNet(embedding_net)

    cushionDB = CushionDataset(root_dir='./cushion', frame_per_traj=params.frame_per_traj,
                               transform=transforms.Compose([
                                   transforms.Resize((56,56)),
                                   transforms.ToTensor()
                               ]))

    # Load data.
    shuffles = {'train': True, 'eval': False}
    dataloaders = {x: DataLoader(cushionDB, batch_size=1, shuffle=shuffles[x],
                                 num_workers=2, collate_fn=EmbedBatch_collate_fn, drop_last=True)
                   for x in list(shuffles.keys())}

    # Training.
    net.to(device) # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.

    criterion = nn.L1Loss(size_average=False)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer.zero_grad()

    train_model(net, dataloaders, criterion, optimizer, device, num_epochs=params.n_epochs)

    print('Finished training.')



if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
#!/usr/bin/env python
# -*- coding: utf-8 -*-

mean, std = 0.1307, 0.3081

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

n_classes = 10
batch_size = 16
n_epochs = 40#00
log_interval = 10
lr = 1e-2
frame_per_traj = 5

acceleration = -9.8
delta_t = 0.1

margin = 1.0
epsilon = 0.05

epsilons = [0, 5, 10, 20, 40, 60, 80]#[0, .05, .1, .15, .2, .25, .3]

data_root = '/home/tenger/datasets/bird_or_bicycle/0.0.4/'
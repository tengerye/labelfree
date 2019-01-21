import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import params



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

        self.a = params.acceleration * (np.arange(1, 6) * params.delta_t)**2
        self.A = np.vstack((np.arange(1, 6) * params.delta_t, np.ones((1, 5))))
        self.A = self.A.transpose()

    def forward(self, output):
        return nn.L1Loss(size_average=False)


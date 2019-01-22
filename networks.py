import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import params



class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class ConstNet(nn.Module):
    def __init__(self, embedding_net):
        super(ConstNet, self).__init__()
        self.embedding_net = embedding_net

        self.a = params.acceleration * (np.arange(1, 6) * params.delta_t)**2
        self.A = np.vstack((np.arange(1, 6) * params.delta_t, np.ones((1, 5))))
        self.A = self.A.transpose()
        self.proj_mat = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T - np.identity(self.A.shape[0])

        self.a, self.A, self.proj_mat = self.a.astype(np.float32), self.A.astype(np.float32), self.proj_mat.astype(np.float32)
        self.const = self.proj_mat @ self.a

        self.proj_mat, self.const = torch.from_numpy(self.proj_mat).to(params.device), \
                                    torch.from_numpy(self.const).view(-1, 1).to(params.device)

    def forward(self, frames):
        """frames: frame_per_traj x C x W x H """
        x = self.embedding_net(frames) #f(x)

        # x = torch.mm(torch.from_numpy(
        #     np.tile(self.proj_mat, (1, params.batch_size) )), x)
        x = torch.mm(self.proj_mat, x)

        x = x - self.const

        return x



class HeightNet(nn.Module):

    def __init__(self):
        super(HeightNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 11, 5)
        self.conv3 = nn.Conv2d(11, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 3 * 3, 72)
        self.fc2 = nn.Linear(72, 1)

    def forward(self, x):
        """frames: frame_per_traj x C x W x H """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
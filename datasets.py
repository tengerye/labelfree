import os

import numpy as np
from PIL import Image
from skimage import io, transform
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import params

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)



class CushionDataset(Dataset):
    """Cushion dataset for constraint learning."""
    def __init__(self, root_dir, frame_per_traj=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_per_traj = frame_per_traj

        # Get names of image.
        self.img_names = os.listdir(self.root_dir)
        filenames = sorted(self.img_names)

        self.img_names, start_idx = [], 0
        for idx in range(len(filenames)-1):
            if int(filenames[idx][-8:-4]) + 1 < int(filenames[idx+1][-8:-4]):
                self.img_names.append(filenames[start_idx: idx+1])
                start_idx = idx + 1

        self.img_names.append(filenames[start_idx:])


    def __len__(self):
        """The whole trajectory is an example."""
        return int(len(self.img_names))

    def __getitem__(self, idx):
        sample_names = np.random.choice(self.img_names[idx], params.frame_per_traj)

        assert self.transform, 'transform can not be None!'

        images = [self.transform(Image.open(
            os.path.join(self.root_dir, img_name))) for img_name in sample_names]

        labels = torch.from_numpy(np.asarray([0.0] * len(images)).astype(np.float32))

        return {'images': images, 'labels': labels}


def EmbedBatch_collate_fn(batch):
    """Reason of custom collate_fn is batch=[[img1, img2,..., imgN], [img1, img2,..., imgN], ...]."""
    images, labels = [torch.stack(sample['images']) for sample in batch], [sample['labels'] for sample in batch]

    # Merge images.
    return torch.cat(images), torch.cat(labels).view(-1, 1)


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

class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        
        mask = torch.ones_like(img)

        for _ in range(self.n_holes):
            y = torch.randint(0, h - self.length, (1,))
            x = torch.randint(0, w - self.length, (1,))

            mask[:, y:y+self.length, x:x+self.length] = 0

        return img * mask

# class Cutout:
#     def __init__(self, n_holes, length):
#         self.n_holes = n_holes
#         self.length = length

#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)

#         for _ in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)

#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)

#             mask[y1:y2, x1:x2] = 0

#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)

#         return img * mask


transform_augmented = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    Cutout(n_holes=1, length=16),

])

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

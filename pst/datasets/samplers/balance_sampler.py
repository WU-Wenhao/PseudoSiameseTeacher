
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from torch.utils.data.dataset import ConcatDataset

class BalanceSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu):
        ### TODO: drop_last is not needed ???

        assert isinstance(dataset, ConcatDataset), "BalanceSampler can only be implemented on ConcatDataset"

        sizes = [len(dataset.datasets[0]), len(dataset.datasets[1])]
        self.sizes = sizes
        self.max_size = max(sizes[0], sizes[1])
        self.samples_per_gpu = samples_per_gpu
        self.times = [1 if (x == self.max_size) else (self.max_size // x + 1) for x in sizes]

        iter1, iter2 = [], []
        for _ in range(self.times[0]):
            iter1.extend(torch.randperm(self.sizes[0]).tolist())
        for _ in range(self.times[1]):
            iter2.extend(torch.randperm(self.sizes[1]).tolist())
        iter2 = (np.array(iter2) + sizes[0]).tolist()
        self.iter1 = iter1
        self.iter2 = iter2

        self.iter_ids = []
        for i in range(0, min(len(self.iter1), len(self.iter2))*2 // self.samples_per_gpu):
            ### NOTE: "*2" to cover unlabeled data as many as possible
            ### TODO: "*2" may be adatively adjusted by the unlabel_ratio
            self.iter_ids.extend(self.iter1[i*self.samples_per_gpu//2:(i+1)*self.samples_per_gpu//2])
            self.iter_ids.extend(self.iter2[i*self.samples_per_gpu//2:(i+1)*self.samples_per_gpu//2])
        self.size = len(self.iter_ids)

    def __len__(self):
        return self.size
        # return self.size // self.samples_per_gpu

    def __iter__(self):
        iter1, iter2 = [], []
        for _ in range(self.times[0]):
            iter1.extend(torch.randperm(self.sizes[0]).tolist())
        for _ in range(self.times[1]):
            iter2.extend(torch.randperm(self.sizes[1]).tolist())
        iter2 = (np.array(iter2) + self.sizes[0]).tolist()
        self.iter1 = iter1
        self.iter2 = iter2

        indices = []
        for i in range(0, min(len(self.iter1), len(self.iter2))*2 // self.samples_per_gpu):
            ### NOTE: "*2" to cover unlabeled data as many as possible
            ### TODO: "*2" may be adatively adjusted by the unlabel_ratio
            indices.extend(self.iter1[i*self.samples_per_gpu//2:(i+1)*self.samples_per_gpu//2])
            indices.extend(self.iter2[i*self.samples_per_gpu//2:(i+1)*self.samples_per_gpu//2])
        # indices = indices.astype(np.int64).tolist()
        # assert len(indices) == self.num_samples
        self.iter_ids = indices
        return iter(indices)


from collections.abc import Sequence

from mmcv.parallel import collate as mmcv_collate
from mmcv.parallel import DataContainer as DC

def collate(batch: Sequence, samples_per_gpu: int = 1):
    """Fit the case with strong augmented data from labeled dataset, and strong augmented and weak augmented data from unlabeled dataset
    """

    new_samples_per_gpu = samples_per_gpu // 2 * 3      ### NOTE: including weak samples
    new_batch = batch[0:samples_per_gpu // 2]
    strong_batch = []
    weak_batch = []
    for i in range(samples_per_gpu // 2, len(batch)):
        b = batch[i]
        strong_batch.append({k: DC(v.data[0][0]) for k, v in b.items()})
        weak_batch.append({k: DC(v.data[0][1]) for k, v in b.items()})
    new_batch = new_batch + strong_batch + weak_batch

    return mmcv_collate(new_batch, new_samples_per_gpu)
    # return mmcv_collate(batch, samples_per_gpu)

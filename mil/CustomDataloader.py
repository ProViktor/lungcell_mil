import numpy as np


class CustomLoader:
    """Torch dataloaders don't support vectors of different lengths,
    this is supposed to act as a dataloader for batches (lists) of bags.
    """

    def __init__(self, dataset, batchsize, shuffle=True):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle

    def batches(self):
        """Returns a list of lists (batches) of bags in the dataset."""
        batches = None

        batchsize = self.batchsize
        n = len(self.dataset)
        dataset = self.dataset

        index = np.arange(n)
        if self.shuffle:
            np.random.shuffle(index)

        batches = []
        k = 0

        while k * batchsize < n:
            max_idx = min((k + 1) * batchsize, n)
            batch_idx = index[k * batchsize : max_idx]
            batch = list()
            for i in batch_idx:
                batch.append(dataset[i])
            batches.append(batch)
            k += 1
        return batches

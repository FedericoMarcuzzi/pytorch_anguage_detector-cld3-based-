import torch
from torch.utils.data import Dataset
import numpy as np

class Batcher(Dataset):
    def __init__(self, data, freq, labels, batch_size):
        labels = torch.tensor(labels, dtype=torch.uint8)
        freq = np.expand_dims(freq, -1)
        self.n_batches = int(np.ceil(data.shape[0] / batch_size))

        self.list_batch_idxs = []
        self.list_batch_freq = []
        self.list_batch_labels = []

        # precomputes batches
        for i in range(self.n_batches):
            offset = i * batch_size
            batch_idxs = data[offset : offset + batch_size]
            batch_freq = freq[offset : offset + batch_size]
            batch_labels = labels[offset : offset + batch_size]

            # reduces the size of the batch based on the largest n-gram ID in the bach (reduces padding).
            max_useful_idx = np.max(np.argmin(batch_idxs, axis=2)) + 1
            batch_idxs = torch.tensor(batch_idxs[:,:,:max_useful_idx])
            batch_freq = torch.tensor(batch_freq[:,:,:max_useful_idx], dtype=torch.float)

            self.list_batch_idxs.append(batch_idxs)
            self.list_batch_freq.append(batch_freq)
            self.list_batch_labels.append(batch_labels)
        
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, i):
        return self.list_batch_idxs[i], self.list_batch_freq[i], self.list_batch_labels[i]
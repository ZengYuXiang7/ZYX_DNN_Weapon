# coding : utf-8
# Author : yuxiang Zeng

import faiss
import numpy as np
import time
import torch


class IndexHamHNSW:

    def __init__(self, config):
        nbits = 64
        self.encoder = faiss.IndexLSH(config.rank, nbits)
        self.innerIndex = faiss.IndexBinaryHNSW(nbits, 32)

    def train(self, hidden):
        self.encoder.train(hidden)

    def add(self, hidden):
        codes = self.encoder.sa_encode(hidden)
        self.innerIndex.train(codes)
        self.innerIndex.add(codes)

    def search(self, hidden, topk):
        codes = self.encoder.sa_encode(hidden)
        return self.innerIndex.search(codes, topk)


class IndexHamHNSWErrorCompensation:

    def __init__(self, config):
        self.config = config
        self.clear()
        self.perfs = []

    def append(self, hidden, target, predict):
        # hidden as tensors
        hidden = hidden.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        self.hiddens.append(hidden)
        self.targets.append(target)
        self.predicts.append(predict.reshape(target.shape))

    def set_ready(self):
        error = np.abs(np.concatenate(self.predicts) - np.concatenate(self.targets))
        percentile = np.percentile(error, 0)
        maskIdx = error > percentile

        self.ready_hiddens = np.vstack(self.hiddens)[maskIdx]
        self.ready_targets = np.concatenate(self.targets)[maskIdx]
        self.index.train(self.ready_hiddens)
        self.index.add(self.ready_hiddens)

    def clear(self):
        self.index = IndexHamHNSW(self.config)
        self.hiddens = []
        self.targets = []
        self.predicts = []
        self.ready_hiddens = None
        self.ready_targets = None

    def correction(self, hidden):
        hidden = hidden.detach().cpu().numpy()
        start = time.time_ns()
        dists, I = self.index.search(hidden, self.config.topk)
        end = time.time_ns()
        self.perfs.append((end - start) / len(hidden))
        compensation = np.zeros(len(hidden), dtype=np.float32)
        lmdas = np.zeros_like(compensation, dtype=np.float32)
        for i in range(len(lmdas)):
            lmdas[i] = self.config.lmda
            compensation[i] = np.mean(self.ready_targets[I[i, :self.config.topk]])
        return torch.from_numpy(compensation).to(self.config.device), torch.from_numpy(lmdas).to(self.config.device)


class Config:
    def __init__(self):
        self.rank = 40  # Dimensionality of the input vectors
        self.topk = 5  # Number of nearest neighbors to search for
        self.lmda = 0.5  # Lambda parameter for compensation
        self.device = 'cpu'


if __name__ == '__main__':

    config = Config()

    # Initialize the IndexHamHNSWErrorCompensation object
    index = IndexHamHNSWErrorCompensation(config)

    index.clear()

    # Generate some sample data
    hidden_vectors = torch.rand((100, config.rank))  # 100 samples of 128-dimensional vectors
    target_vectors = torch.rand((100, 1))  # 100 target values
    predict_vectors = torch.rand((100, 1))  # 100 predicted values

    # Append data to the index
    for hidden, target, predict in zip(hidden_vectors, target_vectors, predict_vectors):
        index.append(hidden, target, predict)

    # Set the index ready for searching and compensation
    index.set_ready()

    # Generate some new hidden vectors for which we want to find corrections
    new_hidden_vectors = torch.rand((10, config.rank))  # 10 new samples

    # Get the correction values
    compensation, lmdas = index.correction(new_hidden_vectors)
    corrected = lmdas * compensation + (1 - lmdas) * predict_vectors

    # Print the results
    print("Compensation values:", compensation.shape)
    print("Corrected values:", corrected.shape)
    print("Lambda values:", lmdas)

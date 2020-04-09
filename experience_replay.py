from collections import deque
import random
import numpy as np
import pickle as pkl


class Experience:
    def __init__(self, bufferSize=1e5, filepath=None):
        np.random.seed()

        if filepath is not None:
            with open(filepath, "rb") as f:
                self.buffer = pkl.load(f)
        else:
            self.buffer = deque([], int(bufferSize))

    def sample(self, batchSize=1024):
        batchSize = min(len(self.buffer), batchSize)

        batch = random.sample(self.buffer, batchSize)

        S = np.asarray([sample[0] for sample in batch]).reshape(batchSize, -1)
        A = np.asarray([sample[1] for sample in batch]).reshape(batchSize, -1)
        R = np.asarray([sample[2] for sample in batch]).reshape(batchSize, -1)
        S_dash = np.asarray([sample[3] for sample in batch]).reshape(batchSize, -1)
        not_terminal = np.asarray([sample[4] for sample in batch]).reshape(batchSize)

        return S, A, R, S_dash, not_terminal

    def store(self, state, action, reward, nextState, not_terminal):
        self.buffer.append([state, action, reward, nextState, not_terminal])

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pkl.dump(self.buffer, f)

    def __len__(self):
        return len(self.buffer)

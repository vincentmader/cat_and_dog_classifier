import numpy as np
import torch

from config import PATH_TO_TRAINING_DATA


def main():

    training_data = np.load(
        PATH_TO_TRAINING_DATA,
        allow_pickle=True
    )

    X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X / 255
    y = torch.Tensor([i[1] for i in training_data])

    return X, y

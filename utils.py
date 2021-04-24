import random

import torch
import numpy as np


def ratio_split(dataset, frac=[0.8, 0.1, 0.1]):
    np.testing.assert_almost_equal(frac[0] + frac[1] + frac[2], 1.0)

    num_mols = len(dataset)
    all_idx = list(range(num_mols))

    train_idx = all_idx[:int(frac[0] * num_mols)]
    valid_idx = all_idx[int(frac[0] * num_mols):int(frac[1] * num_mols)
                                                   + int(frac[0] * num_mols)]
    test_idx = all_idx[int(frac[1] * num_mols) + int(frac[0] * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

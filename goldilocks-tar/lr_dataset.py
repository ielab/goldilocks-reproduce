import sys

from libact.base.dataset import Dataset

from argparse import Namespace

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
import gzip
import pickle


class LrDataset(Dataset):

    def __init__(self, data_features, labels):
        X = data_features

        assert X.shape[0] == len(labels)

        super(LrDataset, self).__init__(X, labels)
        self._X = X

    def get_labeled_features(self):
        """Return dictionary for transformers.Trainer
        """
        return list(zip(self._X[self._labeled_mask],
                        self._y[self._labeled_mask]))

    @classmethod
    def from_cache_file(cls, file, labels):
        return cls(pd.read_pickle(file), labels=labels)

    def spawn_dataset(self, labels=None, idx=None):
        if idx is None:
            return LrDataset(self._X, labels if labels is not None else ([None] * len(self)))
        else:
            assert labels is not None
            return LrDataset(self._X[idx], labels)


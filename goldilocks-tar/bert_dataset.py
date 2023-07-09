from libact.base.dataset import Dataset

from transformers import PreTrainedTokenizer

import pandas as pd
import numpy as np
import torch

from typing import Union

import pickle


class BertDataset(Dataset):

    def __init__(self, input_ids, attention_mask, labels):
        X = np.stack([input_ids, attention_mask], axis=1)
        
        assert X.shape[0] == len(labels)

        super(BertDataset, self).__init__(X, labels)
        self._X = X
    
    @property
    def _input_ids(self):
        return self._X[:,0]
    
    @property
    def _attention_mask(self):
        return self._X[:,1]

    def append(self):
        raise TypeError("Bert dataset does not support dynamically adding examples.")

    # def format_sklearn(self):
    #     raise TypeError("Bert dataset does not support scikit-learn format.")

    # def get_labeled_entries(self):
    #     raise TypeError("Bert dataset does not support scikit-learn format.")

    def get_labeled_features(self):
        """Return dictionary for transformers.Trainer
        """
        return list(zip(self._X[self._labeled_mask],
                         self._y[self._labeled_mask]))

    # def get_unlabeled_features(self):
    #     """Return dictionary for transformers.Trainer
    #     """
    #     return list(zip( self._input_ids[~self._labeled_mask], 
    #                      self._attention_mask[~self._labeled_mask] ) )
    
    @staticmethod
    def collator(features):
        batch = {}

        if not isinstance(features[0][1], np.ndarray):
            batch['input_ids'] = torch.tensor([f[0][0] for f in features], dtype=torch.long)
            batch['attention_mask'] = torch.tensor([f[0][1] for f in features], dtype=torch.long)
            batch['labels'] = torch.tensor([[f[1]] for f in features], dtype=torch.long)
        else:
            batch['input_ids'] = torch.tensor([f[0] for f in features], dtype=torch.long)
            batch['attention_mask'] = torch.tensor([f[1] for f in features], dtype=torch.long)
        
        return batch    

    @classmethod
    def from_raw_text(cls, 
                      raw_text: Union[pd.Series, np.array],
                      labels: Union[pd.Series, np.array],
                      tokenizer: PreTrainedTokenizer,
                      max_length=512,
                      **kwargs):
        
        kwargs_tokenizer = {'padding': True, 'truncation': True, 'max_length': max_length}
        kwargs_tokenizer.update(kwargs)
        
        # need tokenizer here
        encoded = tokenizer(list(raw_text), **kwargs_tokenizer)
        # encoded = [ tokenizer([ t ]) for t in tqdm(raw_text) ]
        return cls(encoded['input_ids'], encoded['attention_mask'], labels)
        
        # pickle.dump( self._features, gzip.open( feature_cache_file, "wb") )

    @classmethod
    def from_cache_file(cls, file, labels):
        return cls( **pd.read_pickle(file), labels=labels )
    
    def save_encoded_features(self, fp):
        pickle.dump({'input_ids': self._input_ids, 'attention_mask': self._attention_mask}, fp)

    @classmethod
    def from_dataset(cls, ds, *args, **kwargs):
        return ds.spawn_dataset(*args, **kwargs)
    
    def spawn_dataset(self, labels=None, idx=None):
        if idx is None:
            return BertDataset(self._input_ids, self._attention_mask, 
                               labels if labels is not None else ([None] * len(self)) )
        else:
            assert labels is not None
            return BertDataset(self._input_ids[idx], self._attention_mask[idx], labels)
        
    def copy(self):
        raise TypeError("Bert dataset does not support copy; use `spawn_dataset` or `from_dataset`")

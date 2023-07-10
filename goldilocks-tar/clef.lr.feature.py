import os
import glob

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import sys


class SaturatedTf(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    k1 : float, optional (default=1.2)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        return X


def lr_tokenizer():
    raw_text = pd.read_csv(f'./clef_info/{ds_dir}/{topic}/raw_text.csv')
    text = raw_text.raw.to_list()
    vectorizer = CountVectorizer()
    tokenized_text = vectorizer.fit_transform(text)

    Tf = SaturatedTf()
    text_features = Tf.transform(tokenized_text)
    with open(f"./clef_info/lr_features/{run_name}_features.pkl", "wb") as dump_file:
        pickle.dump(text_features, dump_file)


ds_dir = sys.argv[1]
assert ds_dir in ['2017/test', '2017/train', '2018/test', '2019/dta/test', '2019/intervention/test',
                   '2019/intervention/train']

topics_per_set = [dir_[-8:] for dir_ in glob.glob(f'./clef_info/{ds_dir}/*')]
ds_name = ds_dir.replace('/', '_')

for topic in topics_per_set:
    run_name = f'clef{ds_name}_{topic}'
    lr_tokenizer()




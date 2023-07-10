import numpy as np
import pandas as pd
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


ds_name = sys.argv[1]
assert ds_name in ['rcv1', 'jb']


# data processing for logistic regression
# if ds_name == 'rcv1':
#     rcv1_sampled = pd.read_csv(f'./{ds_name}/raw_text.csv')
#     rcv1_sampled["raw"] = rcv1_sampled.raw.str.replace(r'(?:[^\w\s]|_)+', ' ', regex=True).str.strip().str.replace(
#         r'\s+', ' ', regex=True).str.lower()
#     rcv1_sampled.to_csv(f'./{ds_name}_info/raw_text_cleaned.csv')
#
# elif ds_name == 'jb':
#     jb_sampled = pd.read_pickle(f'./{ds_name}_info/raw_text_cleaned.pkl')
#     jb_sampled["raw"] = jb_sampled.raw.str.replace(r'(?:[^\w\s]|_)+', ' ', regex=True).str.strip().str.replace(
#         r'\s+', ' ', regex=True).str.lower()
#     with open(f"./{ds_name}_info/raw_text_processed.pkl", "wb") as output_file:
#         pickle.dump(jb_sampled, output_file)


# feature engineering with BM25 within document saturated term frequencies
sampled_mask = pd.read_pickle(f'./{ds_name}_info/sampled-for-bertal.pkl')

if ds_name == 'rcv1':
    cleaned_text = pd.read_csv('./rcv1_info/raw_text_cleaned.csv').iloc[sampled_mask]

elif ds_name == 'jb':
    cleaned_text = pd.read_pickle('./jb_info/raw_text_processed.pkl').iloc[sampled_mask]

text = cleaned_text.raw.to_list()

vectorizer = CountVectorizer()
tokenized_text = vectorizer.fit_transform(text)

Tf = SaturatedTf()
text_features = Tf.transform(tokenized_text)
with open(f"./{ds_name}_info/{ds_name}_sampled_features.pkl", "wb") as dump_file:
    pickle.dump(text_features, dump_file)



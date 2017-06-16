from functools import partial, wraps

import numpy
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


def predict_k(pipeline, X, with_score=True, k=3):
    y_scores = pipeline.predict_proba(X)
    labels = pipeline.steps[-1][1].classes_
    result = [[(str(l), float(s)) if with_score else str(l) for l, s in
               zip(labels[numpy.array(scores.argsort())], numpy.sort(scores))][-k:][::-1]
              for scores in y_scores]
    return result


def _batch_process(texts, func):
    return [func(x) for x in texts]


def transformerize(function):
    """Creates a sklearn transformer from a python method. Used as a decorator."""
    ft = FunctionTransformer(
        func=partial(_batch_process, afunc=function),
        validate=False
    )

    # TODO: make the transformer pickleable
    return wraps(function)(ft)


class DenseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

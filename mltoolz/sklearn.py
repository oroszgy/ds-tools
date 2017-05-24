from functools import partial

import numpy
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
    return FunctionTransformer(
        func=partial(_batch_process, func=function),
        validate=False
    )

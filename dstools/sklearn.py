from functools import partial

from sklearn.preprocessing import FunctionTransformer


def _batch_process(texts, func):
    return [func(x) for x in texts]


def transformerize(function):
    """Creates a sklearn transformer from a python method. Used as a decorator."""
    return FunctionTransformer(
        func=partial(_batch_process, func=function),
        validate=False
    )

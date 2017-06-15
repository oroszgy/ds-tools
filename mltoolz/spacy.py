from collections import namedtuple, Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from spacy.en import English

from mltoolz.sklearn import transformerize

_nlp = English(parser=False, entity=False)

Token = namedtuple("Token", ("text", "lemma", "tag", "pos"))


@transformerize
def nlp(text):
    # return [Token(tok.text, tok.lemma_, tok.tag_, tok.pos_) for tok in _nlp(text)]
    return list(nlp(str(text)).sents)


class DocFeature(BaseEstimator, TransformerMixin):
    def __init__(self, extractor):
        self.extractor = extractor
        self.vectorizer = DictVectorizer()

    def fit(self, X, y=None):
        self.vectorizer.fit(self._count_features(X))
        return self

    def _count_features(self, docs):
        return [Counter(self.extractor(doc)) for doc in docs]

    def transform(self, X):
        features = self.vectorizer.transform(self._count_features(X))
        return features

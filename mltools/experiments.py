from pathlib import Path

import numpy
import pandas as pd


class ExperimentTracker(object):
    def __init__(self, pkl_path, append=True):
        self._fpath = Path(pkl_path)
        if self._fpath.exists() and append:
            self.data = pd.read_pickle(str(self._fpath))
        else:
            self.data = pd.DataFrame()

    def _store(self, result):
        result["_timestamp"] = pd.to_datetime("now")
        self.data = pd.concat([self.data, pd.DataFrame([result])], ignore_index=True)
        self.data.to_pickle((self._fpath))

    def _print(self, result):
        for key, value in result.items():
            if not key.startswith("_"):
                if type(value) in [int, float, numpy.float64, numpy.float32, numpy.float16, numpy.float128,
                                   numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16,
                                   numpy.uint32, numpy.uint64]:
                    print("{:25s}: {:.4f}".format(key, value))
                else:
                    print("{}:\n\t{}".format(key, value))

    def __call__(self, fun):
        def wrapped(*args, **kwargs):
            result = fun(*args, **kwargs)
            self._store(result)
            self._print(result)

        return wrapped

from pathlib import Path

import numpy
import pandas as pd


class ExperimentTracker(object):
    def __init__(self, pkl_path, append=True):
        self._fpath = Path(pkl_path)
        if self._fpath.exists() and append:
            self.data = pd.read_pickle(self._fpath.open())
        else:
            self.data = pd.DataFrame()

    def __call__(self, fun):
        def wrapped(*args, **kwargs):
            result = fun(*args, **kwargs)
            result["_timestamp"] = pd.to_datetime("now")

            self.data = pd.concat([self.data, pd.DataFrame([result])], ignore_index=True)
            self.data.to_pickle(self._fpath.open("w"))

            for key, value in result.items():
                if not key.startswith("_"):
                    if type(value) in [int, float, numpy.float64, numpy.float32, numpy.float16, numpy.float128,
                                       numpy.float256, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.int128]:
                        print("{}:\t\t{:.2f}".format(key, value))
                    else:
                        print("{}:\n\t{}".format(key, value))

        return wrapped

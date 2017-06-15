from pathlib import Path

import pandas as pd


class ExperimentTracker(object):
    def __init__(self, csv_fpath):
        self._fpath = Path(csv_fpath)
        if self._fpath.exists():
            self.data = pd.read_csv(self._fpath.open(), index_col=None, parse_dates=True)
        else:
            self.data = pd.DataFrame()

    def __call__(self, fun):
        def wrapped(*args, **kwargs):
            result = fun(*args, **kwargs)
            result["_timestamp"] = pd.to_datetime("now")

            self.data = pd.concat([self.data, pd.DataFrame([result])], ignore_index=True)
            self.data.to_csv(self._fpath.open("w"), index=False)

            for key, value in result.items():
                if not key.startswith("_"):
                    if type(value) in [int, float]:
                        print("{}:\t\t{:.2f}".format(key, value))
                    else:
                        print("{}:\n\t{}".format(key, value))

        return wrapped

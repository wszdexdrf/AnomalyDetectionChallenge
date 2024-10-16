import numpy as np


# Simple Consumer class
# Not used in the optimal solution
# Only used for testing
class Consumer:
    def __init__(self, type="zscore", params={"threshold": 3}):
        self.type = type
        self.params = params

    def _zscore(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return np.abs((data - mean) / std)

    def _iqr(self, data):
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return np.abs((data - q75) / iqr)

    def detect_anomaly(self, data):
        if self.type == "zscore":
            return self._zscore(data) > self.params["threshold"]
        elif self.type == "iqr":
            return self._iqr(data) > self.params["threshold"]
        else:
            raise ValueError("Unknown anomaly detection method")

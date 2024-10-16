import numpy as np

# Simple generator class
# Not used in the optimal solution
# Only used for testing
class Generator:
    def __init__(self, type, params):
        self.type = type
        self.params = params

    def generate(self, n):
        if self.type == "normal":
            mean = self.params["mean"]
            std = self.params["std"]
            return np.random.normal(mean, std, n)
        elif self.type == "uniform":
            low = self.params["low"]
            high = self.params["high"]
            return np.random.uniform(low, high, n)
        else:
            raise ValueError("Unknown generator type")

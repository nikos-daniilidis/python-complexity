import numpy as np
from scipy.stats import chi2

__author__ = "nikos.daniilidis"

class EventGenerator:

    def __init__(self, seed=42, num_inputs=11, type='chisq', balance=0.5):
        self.seed = seed
        np.random.seed(seed)
        self.num_inputs = num_inputs
        self.type = type
        if type == 'chisq':
            self.cutoff = chi2.isf(balance, df=num_inputs)

    def get_chisq(self, num_events=100):
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        print nx.shape
        x = nx.reshape((num_events, self.num_inputs))
        y = np.greater_equal(np.sum(np.square(x), axis=1), self.cutoff)
        return x, y.astype(int)

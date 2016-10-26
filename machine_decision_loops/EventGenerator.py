import numpy as np
from scipy.stats import chi2

__author__ = "nikos.daniilidis"

class EventGenerator:
    """
    Class which creates a stream of events. Each event is an input vector and a label
    correspondng to the input vector. Certain categories of events with known statistical
    properties are implemented
    """
    def __init__(self, seed=42, num_inputs=11, kind='chisq', balance=0.5):
        """
        Initialize the instance
        :param seed: Int. The seed for the numpy random state.
        :param num_inputs: Int. Number of inputs in the input vector of each event.
        :param kind:  String. Type of event stream to generate.
        :param balance: Float between 0. and 1.0. Fraction of the 0 class in the event stream
        :return: None
        """
        assert isinstance(seed, int)
        assert isinstance(num_inputs, int)
        assert kind in ('chisq')
        assert isinstance(balance, float)
        assert (balance >= 0.) and (balance <= 1.)
        self.seed = seed
        np.random.seed(seed)
        self.num_inputs = num_inputs
        self.kind = kind
        if kind == 'chisq':
            self.cutoff = chi2.isf(balance, df=num_inputs)

    def get(self, num_events=100):
        """
        Generate a number of events and their labels
        :param num_events: Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        if self.kind == "chisq":
            return self.__get_chisq(num_events)

    def __get_chisq(self, num_events):
        """
        Generate a number of chi-squared type events.
        The inputs are drawn from iid gaussian distributions.
        The class is the condition: sum of squared inputs >= threshold
        :param num_events:  Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        print nx.shape
        x = nx.reshape((num_events, self.num_inputs))
        y = np.greater_equal(np.sum(np.square(x), axis=1), self.cutoff)
        return x, y.astype(int)

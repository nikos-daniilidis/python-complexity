import numpy as np
from scipy.stats import chi2, cauchy

__author__ = "nikos.daniilidis"

class EventGenerator:
    """
    Class which creates a stream of events. Each event is an input vector and a label
    correspondng to the input vector. Certain categories of events with known statistical
    properties are implemented
    """
    def __init__(self, seed=42, num_inputs=11, kind='chisq', balance=0.5, noise='gauss', spread=0.1):
        """
        Initialize the instance
        :param seed: Int. The seed for the numpy random state.
        :param num_inputs: Int. Number of inputs in the input vector of each event.
        :param kind:  String. Type of event stream to generate.
        :param balance: Float between 0. and 1.0. Fraction of the 0 class in the event stream
        :param noise: String. One of gaussian, uniform
        :param spread: Float. Spread of the noise in terms of percentiles of the
        :return:
        """
        assert isinstance(seed, int)
        assert isinstance(num_inputs, int)
        assert kind in ('chisq', 'cauchy')
        assert noise in ('gauss', 'uniform')
        assert isinstance(balance, float)
        assert (balance >= 0.) and (balance <= 1.)
        self.seed = seed
        np.random.seed(seed)
        self.num_inputs = num_inputs
        self.kind = kind
        if kind == 'chisq':
            self.cutoff = chi2.isf(balance, df=num_inputs)
            self.spread = (chi2.isf(balance + spread/2., df=num_inputs) -
                           chi2.isf(balance - spread/2., df=num_inputs))
        elif kind == 'cauchy':
            assert (self.num_inputs % 2 == 0)  # only implemented for even number of inputs
            self.cauchy = cauchy(0., np.sqrt(self.num_inputs)/np.pi)
            self.cutoff = self.cauchy.isf(balance)
            self.spread = self.cauchy.isf(balance + spread/2.) - self.cauchy.isf(balance - spread/2.)
            print 'spread', self.spread
        self.noise = noise

    def get(self, num_events=100):
        """
        Generate a number of events and their labels
        :param num_events: Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        if self.kind == "chisq":
            return self.__get_chisq(num_events)
        elif self.kind == "cauchy":
            return self.__get_cauchy(num_events)

    def __get_noise(self, num_events):
        if self.noise == 'gauss' and self.spread > 0:
            return np.random.normal(0., self.spread, num_events)
        elif self.noise == 'uniform' and self.spread > 0:
            return np.random.uniform(-self.spread/2., self.spread/2., num_events)
        else:
            return np.zeros(num_events)

    def __get_chisq(self, num_events):
        """
        Generate a number of chi-squared type events.
        The inputs are drawn from iid gaussian distributions.
        The class is the condition: sum of squared inputs >= threshold.
        The sum of squares of normal variables is a chi squared random variable.
        :param num_events:  Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        x = nx.reshape((num_events, self.num_inputs))
        noise = self.__get_noise(num_events)
        y = np.greater_equal(np.sum(np.square(x), axis=1), self.cutoff + noise)
        return x, y.astype(int)

    def __get_cauchy(self, num_events):
        """
        Generate a number of Cauchy type events.
        The inputs are drawn from iid gaussian distributions.
        The class is the condition: (sum of n/2 squared inputs)/(sum of other n/2 squared inputs) >= threshold.
        The ratio of normal random variables is a Cauchy random variable.
        :param num_events: Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        n_top = self.num_inputs / 2
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        x = nx.reshape((num_events, self.num_inputs))
        noise = self.__get_noise(num_events)
        y = np.greater_equal(np.sum(x[:, :n_top], axis=1)/np.sum(x[:, n_top:], axis=1),
                             self.cutoff + noise)
        return x, y.astype(int)


def basic_checks():
    eg = EventGenerator(seed=42, num_inputs=10, kind='chisq', balance=0.5, noise='gauss', spread=0.1)
    x, y = eg.get(10000)
    print 'chisq at 0.5 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='cauchy', balance=0.5, noise='gauss', spread=0.1)
    x, y = eg.get(10000)
    print 'cauchy at 0.5 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='chisq', balance=0.3, noise='gauss', spread=0.1)
    x, y = eg.get(10000)
    print 'chisq at 0.3 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='cauchy', balance=0.3, noise='gauss', spread=0.1)
    x, y = eg.get(100000)
    print 'cauchy at 0.3 ->', np.mean(y)


if __name__ == '__main__':
    basic_checks()
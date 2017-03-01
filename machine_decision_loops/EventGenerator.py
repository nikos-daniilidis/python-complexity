import numpy as np
from scipy.stats import chi2, cauchy, norm

__author__ = "nikos.daniilidis"

class EventGenerator:
    """
    Class which creates a stream of events. Each event is an input vector and a label
    correspondng to the input vector. Certain categories of events with known statistical
    properties are implemented
    """
    def __init__(self, seed=42, num_inputs=11, kind='chisq', used_features=0.8,
                 balance=0.5, noise='gauss', spread=0.1, verbose=False):
        """
        Initialize the instance
        :param seed: Int. The seed for the numpy random state.
        :param num_inputs: Int. Number of inputs in the input vector of each event.
        :param kind:  String. Type of event stream to generate.
        :param used_features: Float. Fraction of features to use in labeling the stream (randomly sampled).
        :param balance: Float between 0. and 1.0. Fraction of the 0 class in the event stream
        :param noise: String. One of gaussian, uniform
        :param spread: Float. Spread of the noise in terms of percentiles of the
        :param verbose: Boolean. Print stuff if true
        :return:
        """
        assert isinstance(seed, int)
        assert isinstance(num_inputs, int)
        assert kind in ('gauss', 'chisq', 'cauchy')
        assert noise in ('gauss', 'uniform')
        assert isinstance(balance, float)
        assert (balance >= 0.) and (balance <= 1.)
        self.seed = seed
        np.random.seed(seed)
        self.used_indices = np.random.choice(range(num_inputs),
                                             size=2*int(round(num_inputs*used_features/2.)),
                                             replace=False)
        self.numerator_indices = None  # only used when type is 'cauchy'
        self.denumerator_indices = None  # only used when type is 'cauchy'
        self.coefficients = None  # only used when type is 'gauss'
        self.used_inputs = len(self.used_indices)
        self.num_inputs = num_inputs
        assert (self.used_inputs <= self.num_inputs)
        self.kind = kind
        if kind == 'gauss':
            self.coefficients = 2.*np.random.random(size=self.used_inputs) - 1.  # range is [-1, 1]
            sigma = np.sqrt(np.sum(np.power(self.coefficients, 2)))
            self.spread = sigma*(norm.isf(spread/2.) - norm.isf(-spread/2.))
            self.cutoff = sigma*norm.isf(balance)
        elif kind == 'chisq':
            self.cutoff = chi2.isf(balance, df=self.used_inputs)
            self.spread = (chi2.isf(balance + spread/2., df=self.used_inputs) -
                           chi2.isf(balance - spread/2., df=self.used_inputs))
        elif kind == 'cauchy':
            self.numerator_indices = self.used_indices[:self.used_inputs/2]
            self.denumerator_indices = self.used_indices[self.used_inputs/2:]
            assert (self.used_inputs % 2 == 0)  # only implemented for even number of inputs
            self.cauchy = cauchy(0., np.sqrt(self.used_inputs)/np.pi)
            self.cutoff = self.cauchy.isf(balance)
            self.spread = self.cauchy.isf(balance - spread/2.) - self.cauchy.isf(balance + spread/2.)
            if verbose:
                print 'spread', self.spread
        self.noise = noise

    def get_labeled(self, num_events=100):
        """
        Generate a number of events and their labels
        :param num_events: Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        if self.kind == "gauss":
            return self.__get_gauss(num_events)
        elif self.kind == "chisq":
            return self.__get_chisq(num_events)
        elif self.kind == "cauchy":
            return self.__get_cauchy(num_events)

    def get_unlabeled(self, num_events=100):
        """
        Generate a stream of unlabeled events
        :param num_events:
        :return:
        """
        nx = np.random.normal(0., 1., self.num_inputs * num_events)
        x = nx.reshape((num_events, self.num_inputs))
        return x

    def append_noise(self, x):
        """
        Append noise to a stream of events
        :param x: numpy array of events with dimensions (num_events, num_inputs)
        :return: x, noise. Numpy arrays of float, x is the input events, noise has shape (num_events,)
        """
        num_events = x.shape[0]
        noise = self.__get_noise(num_events)
        return x, noise

    def label(self, x, noise):
        """
        Label a stream of events. Refers to the labeler corresponding to the type of event.
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :param noise: Numpy array of float with dimensions (num_events,).
        :return: x, y. Numpy arrays of float/int.
        """
        if self.kind == "gauss":
            return self.__label_gauss(x, noise)
        elif self.kind == "chisq":
            return self.__label_chisq(x, noise)
        elif self.kind == "cauchy":
            return self.__label_cauchy(x, noise)

    def __get_noise(self, num_events):
        """
        Get a stream of noise events. Can be gaussian or uniformly distributed.
        :param num_events: int.
        :return: Numpy array of noise value for each event. Shape is (num_events,).
        """
        if self.noise == 'gauss' and self.spread > 0:
            return np.random.normal(0., self.spread, num_events)
        elif self.noise == 'uniform' and self.spread > 0:
            return np.random.uniform(-self.spread/2., self.spread/2., num_events)
        else:
            return np.zeros(num_events)

    def __label_gauss(self, x, noise):
        """
        Label a stream of gaussian type events. The class is the condition: sum of coefficients*inputs >= threshold.
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :param noise: Numpy array of float with dimensions (num_events,).
        :return: x, y. Numpy arrays of float/int.
        """
        # TODO: Add assertions to check sizes
        y = np.greater_equal(
            np.inner(x[:, self.used_indices], self.coefficients),
            self.cutoff + noise)
        return x, y.astype(int)

    def __get_gauss(self, num_events):
        """
        Generate a number of gauss type events.
        The inputs are drawn from iid gaussian distributions.
        The class is the condition: sum of coefficients*inputs >= threshold.
        :param num_events:  Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        x = self.get_unlabeled(num_events)
        noise = self.__get_noise(num_events)
        return self.__label_gauss(x, noise)

    def __label_chisq(self, x, noise):
        """
        Label a stream of chi-squared type events. The class is the condition: sum of squared inputs >= threshold.
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :param noise: Numpy array of float with dimensions (num_events,).
        :return: x, y. Numpy arrays of float/int.
        """
        # TODO: Add assertions to check sizes
        y = np.greater_equal(np.sum(np.square(x[:, self.used_indices]), axis=1), self.cutoff + noise)
        return x, y.astype(int)

    def __get_chisq(self, num_events):
        """
        Generate a number of chi-squared type events.
        The inputs are drawn from iid gaussian distributions.
        The class is the condition: sum of squared inputs >= threshold.
        The sum of squares of normal variables is a chi squared random variable.
        :param num_events:  Int. Number of events to generate
        :return: x, y. Numpy arrays of float/int
        """
        x = self.get_unlabeled(num_events)
        noise = self.__get_noise(num_events)
        return self.__label_chisq(x, noise)

    def __label_cauchy(self, x, noise):
        """
        Label a stream of Cauchy type events.
        The class is the condition: (sum of n/2 squared inputs)/(sum of other n/2 squared inputs) >= threshold.
        :param x: Numpy array of float with dimensions (num_events, num_inputs).
        :param noise: Numpy array of float with dimensions (num_events,).
        :return: x, y. Numpy arrays of float/int.
        """
        # TODO: Add assertions to check sizes
        n_top = self.used_inputs / 2
        y = np.greater_equal(np.sum(x[:, self.used_indices[:n_top]],
                                    axis=1) / np.sum(x[:, self.used_indices[n_top:]],
                                                     axis=1),
                             self.cutoff + noise)
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
        x = self.get_unlabeled(num_events)
        noise = self.__get_noise(num_events)
        return self.__label_cauchy(x, noise)


def basic_checks():
    eg = EventGenerator(seed=42, num_inputs=10, kind='gauss', balance=0.5, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(10000)
    print 'gauss at 0.5 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='chisq', balance=0.5, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(10000)
    print 'chisq at 0.5 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='cauchy', balance=0.5, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(10000)
    print 'cauchy at 0.5 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='gauss', balance=0.3, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(10000)
    print 'gauss at 0.3 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='chisq', balance=0.3, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(10000)
    print 'chisq at 0.3 ->', np.mean(y)

    eg = EventGenerator(seed=42, num_inputs=10, kind='cauchy', balance=0.3, noise='gauss', spread=0.1)
    x, y = eg.get_labeled(100000)
    print 'cauchy at 0.3 ->', np.mean(y)


if __name__ == '__main__':
    basic_checks()
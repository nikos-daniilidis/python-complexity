import numpy as np
import warnings
from config import GREEDY, EPS_GREEDY, SOFTMAX, THOMPSON, THOMPSON_SOFTMAX, SINGLE_STREAM, COMPETING_STREAMS

__author__ = "nikos.daniilidis"


class EventSelector:
    """
    Class which filters streams of events based on model scores. If a single event stream is
    fed to the filter, we use a threshold on the model score for each event. if a number of
    event streams and the corresponding models are fed to the filter, the event with the
    highest weighted model score is selected for each round of events.
    """
    def __init__(self, selection_type=SINGLE_STREAM, method=GREEDY):
        """
        :param selection_type: String. Support for single streams and competing event streams
        :param method: String. Methods for selecting among competing events, if criterion is competing streams
        :return: None
        """
        assert selection_type in (SINGLE_STREAM, COMPETING_STREAMS)
        assert method in (GREEDY, EPS_GREEDY, SOFTMAX, THOMPSON, THOMPSON_SOFTMAX)  # TODO: Add 'ucb'
        self.selection_type = selection_type
        self.method = method

    def filter(self, xs, ys, models, **kwargs):
        """
        The main method of this class. Filters event streams. Applies to both single event streams
        and competing event streams.
        :param xs: List of numpy array. Each item in the list is a segment of the inputs of an event stream.
        :param ys: List of numpy array. Each item in the list is a segment of the labels of an event stream.
        :param models: List of trained sklearn classifier models. Each model corresponds to an event stream.
        :param kwargs: Parameters required for single stream or multiple streams.
        :return: (List of numpy array, List of numpy array).
        """
        if self.selection_type == SINGLE_STREAM:
            return self.__filter_single_stream(xs, ys, models, **kwargs)
        elif self.selection_type == COMPETING_STREAMS:
            return self.__filter_multiple_streams(xs, ys, models, **kwargs)

    @staticmethod
    def __filter_single_stream(xs, ys, models, threshold=0.4):
        """
        Filter a single event stream.
        :param xs: List of numpy array of length one. The item in the list is a segment of the inputs of an event stream.
        :param ys: List of numpy array of length one. The item in the list is a segment of the labels of an event stream.
        :param models: List of trained sklearn classifier models of length one . The model corresponds to the event stream.
        :param threshold: Float. If model.predict_proba(x) >= threshold, event x and its label
                          remain in the stream.
        :return: Tuple of numpy arrays. The event inputs and the event labels.
        """
        assert len(xs) == 1
        assert len(ys) == 1
        assert len(models) == 1  # list of trained scikit classifiers
        assert type(threshold) is float
        x = xs[0]
        y = ys[0]
        model = models[0]
        if model is None:
            return x, y
        else:
            ix_keep = np.where(model.predict_proba(x)[:, 0] > threshold)[0]
            return x[ix_keep, ], y[ix_keep, ]

    def __filter_multiple_streams(self, xs, ys, models, sd_estimator, event_gains=None, verbose=False, **kwargs):
        """
        Filter competing event streams. Events arrive in rounds, one event per stream. At each round,
        a single event from one stream is selected. The critersion is: choose the event with the highest
        weighted probability of being in class 1.
        :param xs: List of numpy array. Each item in the list is a segment of the inputs of an event stream.
        :param ys: List of numpy array. Each item in the list is a segment of the labels of an event stream.
        :param models: List of trained sklearn classifier models. Each model corresponds to an event stream.
        :param sd_estimator: Trained model variance estimator or None. This function predicts variances for
                    input with shape [num_events, num_streams]. Predictor must have a method
                    predict_sd : numpy array [n_events, n_features] -> numpy array [n_events, n_streams]
        :param event_gains: List of float. Weights to use when selecting which event to select in each round.
        :return: Tuple of lists of numpy arrays. The event inputs and the event labels for each stream.
        """
        # event_gains is iterable of float
        assert len(xs) > 1
        assert len(xs) == len(models)
        assert len(xs) == len(ys)
        assert self.selection_type == COMPETING_STREAMS
        if sd_estimator is not None:
            assert sd_estimator.predict_sd(None) == 'standard deviation estimator'
        # TODO: fix these assertions
        #for ix in range(len(models)):
        #    assert callable(models[ix]) or models[ix] is None
        #assert callable(sds) or sds is None
        num_events = ys[0].shape[0]
        num_streams = len(models)
        if None in models:
            # if any model is None, revert to random choice between all streams
            ixs = np.random.choice(np.arange(num_streams), num_events, replace=True)
        else:
            if event_gains is None:
                event_gains = np.ones(num_streams)

            size = (num_events, num_streams)
            ps = np.zeros(size)

            for ix, model in enumerate(models):  # to get predictions, run through streams
                ps[:, ix] = event_gains[ix] * model.predict_proba(xs[ix])[:, 1].T
                if not np.array_equal(xs[ix], xs[0]):
                    warnings.warn("EventSelector.filter() encountered inconsistent input streams (0, %d)\n" % ix)

            if self.method in (THOMPSON, THOMPSON_SOFTMAX):
                for ix, x in enumerate(xs):
                    assert np.array_equal(x, xs[0])
                vs = sd_estimator.predict_sd(xs[ix])  # standard deviation of predictions for each event in each stream
                assert vs.shape == size
            elif self.method in (GREEDY, SOFTMAX):
                # if running greedy optimization, the variances are set to 0 and random factors are set to 1
                vs = np.zeros(size)
            else:
                warnings.warn("method = %s not implemented in EventSelector. Fallback to greedy\n" % self.method)
                vs = np.zeros(size)

            fs = np.random.normal(loc=0., scale=1., size=size)
            rfs = np.ones(size) + np.multiply(fs, vs)
            ps = np.multiply(ps, rfs)

            # index of winner for each row in events stream
            ixs = self.__choose_indices(ps=ps, num_events=num_events, num_streams=num_streams, **kwargs)

        xlst = []
        ylst = []
        for ix, xx in enumerate(xs):
            xlst.append(xs[ix][np.equal(ixs, ix), :])
            ylst.append(ys[ix][np.equal(ixs, ix),])
            #TODO pout[ix] = ps[...]

        if verbose:
            print 'weighted ps are: ', ps
            print 'selected indices are: ', ixs
            print 'selected labels are: ', ylst

        return xlst, ylst

    def __choose_indices(self, ps, num_events, num_streams, tau=1., eps=0.05):
        """
        Choose indices for a number of events, given the score of each event
        :param ps: numpy array of shape [num_events, num_streams]. The scores of the events in each stream
        :param num_events: int. The number of events
        :param num_streams: int. The number of streams to choose from
        :param tau: float. Temperature to use in the softmax selection
        :param tau: float. Percentage of cases in which to choose randomly in epsilon greedy exploration
        :return: numpy array of int, of length [num_events]. The stream indices of selected events
        """
        size = (num_events, num_streams)
        if self.method in (SOFTMAX, THOMPSON_SOFTMAX):
            p = np.exp(ps/tau)
            psum = np.sum(p, axis=1)
            pcum = np.zeros(size)
            pcum[:, 0] = p[:, 0] / psum
            for ix in range(num_streams)[1:]:
                pcum[:, ix] = pcum[:, ix - 1] + p[:, ix] / psum[:, ]
            dice = np.random.uniform(size=num_events)

            low = np.roll(pcum, shift=1, axis=1)
            low[:, 0] = 0.
            ixs = np.where((low <= dice[:, None])
                           & (pcum > dice[:, None]))[1]
        elif self.method in (GREEDY, EPS_GREEDY, THOMPSON):
            ixs = np.argmax(ps, axis=1)
            if self.method == EPS_GREEDY:
                num_explore = int(round(eps * num_events))
                explore_ixs = np.random.choice(range(num_events), size=num_explore, replace=False)
                random_choices = np.random.randint(0, num_streams, num_explore)
                ixs[explore_ixs] = random_choices
        else:
            warnings.warn("method = %s not implemented in EventSelector. Fallback to greedy\n" % self.method)
            ixs = np.argmax(ps, axis=1)

        return ixs


class DummyModel:
    """
    A class with dummy model predictions for testing.
    """
    def __init__(self, kind='uniform'):
        assert kind in ('uniform', 'logit_gaussian', 'parity_sum')
        self.kind = kind

    def predict_proba(self, x):
        assert isinstance(x, (np.ndarray, np.generic))
        if self.kind == 'uniform':
            p = np.random.uniform(0., 1., x.shape[1])
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3, 1), buffer=p.T),
                              np.ndarray(shape=(3, 1), buffer=pp.T)))
        elif self.kind == 'logit_gaussian':
            p = 1./(1. + np.exp(-1.*np.random.normal(0., 1., x.shape[1])))
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3, 1), buffer=p.T),
                              np.ndarray(shape=(3, 1), buffer=pp.T)))
        elif self.kind == 'parity_sum':
            s = np.sum(x, axis=1)
            p = 1. * (s % 2) / s
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3, 1), buffer=p.T),
                              np.ndarray(shape=(3, 1), buffer=pp.T)))
        else:
            return 0., 0.

class DummyVarianceEstimator:
    """
    A class with dummy model variances for testing.
    """
    def __init__(self, kind='uniform', magnitude=0.01):
        assert kind in ('uniform', 'logit_gaussian', 'parity_sum')
        self.kind = kind
        self.magnitude = magnitude

    def predict_sd(self, x):
        if x is None:
            return 'standard deviation estimator'
        if self.kind == 'uniform':
            p = np.random.uniform(0., self.magnitude, x.shape[1])
            return np.power(np.ndarray(shape=(3, 1), buffer=p.T), 2)


if __name__ == "__main__":
    """Set of tests for the EventSelector class"""
    x1 = np.array([[0, 1, 0], [1, 2, 0], [1, 2, 3]])
    y1 = np.array([0.1, 0.2, 0.3])
    x2 = np.array([[0, 1, 1], [1, 2, 1], [2, 2, 3]])
    y2 = np.array([0.2, 0.3, 0.4])
    es1 = EventSelector(selection_type='single_stream', method='greedy')
    es2 = EventSelector(selection_type='competing_streams', method='thompson')
    dm = DummyModel(kind='parity_sum')
    ve = DummyVarianceEstimator(kind='uniform', magnitude=2.)

    print('Single stream-single cutoff (@0.5) example:')
    print('Inputs: ')
    print x1, y1
    print ('Scores: ')
    print dm.predict_proba(x1)
    x11, y11 = es1.filter([x1], [y1], [dm], threshold=0.5)
    print('Outputs: ')
    print x11, y11

    print('Multiple streams-no cutoff example:')
    print('Inputs: ')
    print x1, y1
    print x2, y2
    print ('Scores: ')
    print dm.predict_proba(x1)
    print dm.predict_proba(x2)
    print ('Uncertainties:')
    print ve.predict_sd(x1)
    print ve.predict_sd(x2)
    xs, ys = es2.filter(xs=[x1, x2], ys=[y1, y2], models=[dm, dm], sd_estimator=ve, event_gains=[1., 3.])
    print('Outputs: ')
    print xs, ys,

import numpy as np


class EventSelector:
    """
    Class which filters streams of events based on model scores. If a single event stream is
    fed to the filter, we use a threshold on the model score for each event. if a number of
    event streams and the corresponding models are fed to the filter, the event with the
    highest weighted model score is selected for each round of events.
    """
    def __init__(self, criterion='single_stream'):
        """
        Initializatio. Only the criterion is needed here.
        :param criterion: String. Support for single streams and competing event streams.
        :return: None
        """
        self.criterion = criterion
        assert self.criterion in ('single_stream', 'competing_streams')

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
        if self.criterion == 'single_stream':
            return self.__filter_single_stream(xs, ys, models, **kwargs)
        elif self.criterion == 'competing_streams':
            return self.__filter_multiple_streams(xs, ys, models, **kwargs)

    def __filter_single_stream(self, xs, ys, models, threshold=0.4):
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
        ix_keep = np.where(model.predict_proba(x)[:, 0] > threshold)[0]
        return x[ix_keep, ], y[ix_keep, ]

    def __filter_multiple_streams(self, xs, ys, models, event_gains=None):
        """
        Filter competing event streams. Events arrive in rounds, one event per stream. At each round,
        a single event from one stream is selected. The critersion is: choose the event with the highest
        weighted probability of being in class 1.
        :param xs: List of numpy array. Each item in the list is a segment of the inputs of an event stream.
        :param ys: List of numpy array. Each item in the list is a segment of the labels of an event stream.
        :param models: List of trained sklearn classifier models. Each model corresponds to an event stream.
        :param event_gains: List of float. Weights to use when selecting which event to select in each round.
        :return: Tuple of lists of numpy arrays. The event inputs and the event labels for each stream.
        """
        # event_gains is iterable of float
        assert len(xs) > 1
        assert len(xs) == len(models)
        assert len(xs) == len(ys)
        assert self.criterion == 'competing_streams'
        num_events = ys[0].shape[0]
        num_streams = len(models)
        if event_gains is None:
            event_gains = np.ones(num_streams)

        ps = np.zeros((num_events, num_streams))
        for ix, model in enumerate(models):
            ps[:, ix] = event_gains[ix] * model.predict_proba(xs[ix])[:, 0].T

        print 'weighted ps are: ', ps
        ixs = np.argmax(ps, axis=1)  # index of winner for each row in events stream
        xout = []
        yout = []
        for ix, xx in enumerate(xs):
            xout.append(xs[ix][np.equal(ixs, ix), :])
            yout.append(ys[ix][np.equal(ixs, ix),])
            #TODO pout[ix] = ps[...]

        return xout, yout


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

if __name__ == "__main__":
    """Set of tests for the EventSelector class"""
    x1 = np.array([[0, 1, 0], [1, 2, 0], [1, 2, 3]])
    y1 = np.array([0.1, 0.2, 0.3])
    x2 = np.array([[0, 1, 1], [1, 2, 1], [2, 2, 3]])
    y2 = np.array([0.2, 0.3, 0.4])
    es1 = EventSelector(criterion='single_stream')
    es2 = EventSelector(criterion='competing_streams')
    dm = DummyModel(kind='parity_sum')

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
    print x1, y1, x2, y2
    print ('Scores: ')
    print dm.predict_proba(x1), dm.predict_proba(x2)
    xs, ys = es2.filter([x1, x2], [y1, y2], [dm, dm], event_gains=[1., 3.])
    print('Outputs: ')
    print xs, ys,

import numpy as np


class EventSelector:
    def __init__(self, criterion='single_stream'):
        self.criterion = criterion
        assert self.criterion in ('single_stream', 'competing_streams')

    def filter(self, xs, ys, models, **kwargs):
        if self.criterion == 'single_stream':
            return self.__filter_single_stream(xs, ys, models, **kwargs)
        elif self.criterion == 'competing_streams':
            return self.__filter_multiple_streams(xs, ys, models, **kwargs)

    def __filter_single_stream(self, xs, ys, models, threshold=0.4):
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
    def __init__(self, kind='uniform'):
        assert kind in ('uniform', 'logit_gaussian', 'parity_sum')
        self.kind = kind

    def predict_proba(self, x):
        assert isinstance(x, (np.ndarray, np.generic))
        if self.kind == 'uniform':
            p = np.random.uniform(0., 1., x.shape[1])
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3,1), buffer=p.T),
                              np.ndarray(shape=(3,1), buffer=pp.T)))
        elif self.kind == 'logit_gaussian':
            p = 1./(1. + np.exp(-1.*np.random.normal(0., 1., x.shape[1])))
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3,1), buffer=p.T),
                              np.ndarray(shape=(3,1), buffer=pp.T)))
        elif self.kind == 'parity_sum':
            s = np.sum(x, axis=1)
            p = 1. * (s % 2) / s
            pp = 1. - p
            return np.hstack((np.ndarray(shape=(3,1), buffer=p.T),
                              np.ndarray(shape=(3,1), buffer=pp.T)))
        else:
            return 0., 0.

if __name__ == "__main__":
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

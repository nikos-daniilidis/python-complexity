import numpy as np
import scipy.stats as stats
import sklearn

__author__ = "nikos.daniilidis"


class DataTomographer:
    def __init__(self, x_train, y_train, model=None, ntiles=10):
        # TODO: some of these need to move to the metods
        assert isinstance(x_train, np.array)
        assert isinstance(y_train, np.array)
        self.xtr = x_train
        self.ytr = y_train
        self.model = model
        self.ntiles=ntiles
        ##self.ntiles = np.percentile(self.xtr, q=[100.*(i+1)/ntiles for i in range(ntiles-1)], axis=0)

    def kl(self, x, rule=None):
        assert rule in (None, 'fd')
        if rule is None:
            pre, bns = np.histogram(self.xtr, self.ntiles, normed=True)
            pst, bns = np.histogram(x, bns, normed=True)
        elif rule == 'fd':
            pre, bns = np.histogram(self.xtr, bins=rule, normed=True)
            pst, bns = np.histogram(x, bns, normed=True)
        return stats.entropy(pre, pst)

    def stagewise_metric(self, step=10, metric='logloss'):
        assert metric in ('auc', 'logloss')
        n_estinmators = self.model.get_params['n_estimators']
        return  # TODO: Finsh this
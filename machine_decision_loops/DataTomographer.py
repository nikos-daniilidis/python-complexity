import numpy as np
import scipy.stats as stats
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
from matplotlib import pyplot as plt

__author__ = "nikos.daniilidis"


class DataTomographer:
    """
    Methods for measuring changes to the distribution of data streams. Implemented are:
    KL: Kuhlback-Leibler divergence for the distribution of each feature in each data stream.
    stagewise_logloss: Log loss for the stagewise predictions of the existing models for each data stream.
    """
    def __init__(self, xrefs, yrefs, xus, yus, models=None):
        """
        Initialize the class.
        :param xrefs: List of numpy array. The train data Xs for the last model update for each stream.
        :param yrefs: List of numpy array. The train data labels for the last model update for each stream.
        :param xus: List of numpy array. The incoming data Xs for each stream.
        :param yus: List of numpy array. The incoming data labels for each stream.
        :param models: List of trained classifier models. One for each stream.
        """
        #assert all(isinstance(xref, np.array) for xref in xrefs)
        #assert all(isinstance(yref, np.array) for yref in yrefs)
        #assert all(isinstance(xu, np.array) for xu in xus)
        #assert all(isinstance(yu, np.array) for yu in yus)
        self.xrefs = xrefs
        self.yref = yrefs
        self.xus = xus
        self.yus = yus
        self.models = models

    def kl(self, ntiles=10, rule=None):
        """
        Compute the KL divergence between the distribution of values during training and during model serving for each
        feature in each data stream. Uses train data as the basis for the distribution against which to compare.
        :param ntiles: Integer. Number of buckets to use for a histogram of the distribution.
        :param rule: String. The rule to use to compute the distribution bins for the train data.
        :return: List of numpy array. List of the KL divergences for the features in each stream.
        """
        assert rule is None or (rule in ('fd', 'auto'))
        all_kls = []
        for ix, xref in enumerate(self.xrefs):  # loop over streams
            stream_kl = np.zeros(xref.shape[1])   # the KL divergences for all features in the current stream (ix)
            xu = self.xus[ix]
            for jx in range(xref.shape[1]):  # loop over features in the stream
                xxref = xref[:, jx]
                xxu = xu[:, jx]
                if rule is None:
                    pre, bns = np.histogram(xxref, bins=ntiles, normed=True)
                    pst, bns = np.histogram(xxu, bns, normed=True)
                elif rule in ('fd', 'auto'):
                    pre, bns = np.histogram(xxref, bins=rule, normed=True)
                    pst, bns = np.histogram(xxu, bns, normed=True)
                stream_kl[jx] = stats.entropy(pre, pst)
            all_kls.append(stream_kl)
        return all_kls

    def stagewise_metric(self,  metric='logloss'):
        """
        Compute the stage wise log loss (or auc if you must) of the trained model on the train data and on the incoming
        unseen data.
        :param metric: String. One of 'auc' or 'logloss'
        :return: metrics_tr, metrics_upd. A tuple of lists of numpy arrays with the metric for each stage of the
                estimator. _tr is on the _train data, upd is on the incoming data stream.
        """
        assert metric in ('auc', 'logloss')
        n_estimators = self.model.get_params['n_estimators']
        metrics_tr = [np.zeros(n_estimators) for ix in range(len(self.models))]
        metrics_upd = [np.zeros(n_estimators) for ix in range(len(self.models))]
        for ix, model in self.models:  # loop through streams
            xref, yref = self.xrefs[ix], self.yrefs[ix]
            for j, ptr in enumerate(self.model.staged_predict_proba(xref)):  # loop through estimator iterations/train
                if metric == 'logloss':
                    metrics_tr[ix][j] = log_loss(yref, ptr)
                elif metric == 'auc':
                    metrics_tr[ix][j] = roc_auc_score(yref, ptr)

            xu, yu = self.xus[ix], self.yus[ix]
            for j, pu in enumerate(self.model.staged_predict_proba(xu)):  # loop through estimator iterations/unseen
                if metric == 'logloss':
                    metrics_upd[ix][j] = log_loss(yu, pu)
                elif metric == 'auc':
                    metrics_tr[ix][j] = roc_auc_score(yref, ptr)

        return metrics_tr, metrics_upd

    def plot_kl(self, ntiles=10, rule=None, **kwargs):
        """
        Plot the KL divergence for all features in all streams.
        :param ntiles: Integer. Number of percentiles to use.
        :param rule: String. The rule to use to compute the distribution bins for the train data.
        :return:
        """
        kls = self.kl(ntiles=ntiles, rule=rule)
        for ix, kl in enumerate(kls):
            nm = 'KL' + str(ix)
            df = pd.DataFrame(data={'feature': ['X'+str(ii)
                                                for ii in range(self.xrefs[0].shape[1])],
                                    nm: [k for k in kl]})
            if ix == 0:
                ax = df.plot(x='feature', y=nm, kind='barh', **kwargs)
            else:
                ax = df.plot(x='feature', y=nm, kind='barh', ax=ax, **kwargs)

        plt.show()
        plt.pause(1.0)

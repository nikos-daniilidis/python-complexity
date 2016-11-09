import numpy as np
import scipy.stats as stats
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
from matplotlib import pyplot as plt
from time import time

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
        # TODO: Fix these assertions
        #assert all(isinstance(xref, np.array) for xref in xrefs)
        #assert all(isinstance(yref, np.array) for yref in yrefs)
        #assert all(isinstance(xu, np.array) for xu in xus)
        #assert all(isinstance(yu, np.array) for yu in yus)
        self.xrefs = xrefs
        self.yrefs = yrefs
        self.xus = xus
        self.yus = yus
        self.models = models
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # for plotting

    def stagewise_metric(self, metric='logloss', verbose=False):
        """
        Compute the stage wise log loss (or auc if you must) of the trained model on the train data and on the incoming
        unseen data.
        :param metric: String. One of 'auc' or 'logloss'
        :param verbose: Boolean. If true, print messages.
        :return: metrics_ref, metrics_upd. A tuple of lists of numpy arrays with the metric for each stage of the
                estimator. _ref is on the _reference data, _upd is on the incoming data stream.
        """
        assert metric in ('auc', 'logloss')
        metrics_ref, metrics_upd = [], []
        for model in self.models:
            n_estimators = model.best_params_['n_estimators']
            metrics_ref.append(np.zeros(n_estimators))
            metrics_upd.append(np.zeros(n_estimators))
        for ix, model in enumerate(self.models):  # loop through streams
            xref, yref = self.xrefs[ix], self.yrefs[ix]
            for j, pref in enumerate(model.best_estimator_.staged_predict_proba(xref)):  # loop through estimator iterations/train
                if metric == 'logloss':
                    metrics_ref[ix][j] = log_loss(yref, pref)
                elif metric == 'auc':
                    metrics_ref[ix][j] = roc_auc_score(yref, pref)

            xu, yu = self.xus[ix], self.yus[ix]
            for j, pu in enumerate(model.best_estimator_.staged_predict_proba(xu)):  # loop through estimator iterations/unseen
                if metric == 'logloss':
                    metrics_upd[ix][j] = log_loss(yu, pu)
                elif metric == 'auc':
                    metrics_ref[ix][j] = roc_auc_score(yref, pref)

        if verbose:
            print 'Reference metrics: '
            print metrics_ref
            print 'Update metrics: '
            print metrics_upd

        return metrics_ref, metrics_upd

    def kuhl_leib(self, ntiles=10, rule=None, prior=1e-6, verbose=False):
        """
        Compute the KL divergence between the distribution of values during training and during model serving for each
        feature in each data stream. Uses train data as the basis for the distribution against which to compare.
        :param ntiles: Integer. Number of buckets to use for a histogram of the distribution.
        :param rule: String. The rule to use to compute the distribution bins for the train data.
        :param prior: Float. Prior density value to use for histogram bins with 0 counts.
        :param verbose: Boolean. If tue, show results during execution.
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
                pst[pst < prior] = prior
                stream_kl[jx] = stats.entropy(pre, pst)
                if verbose and stream_kl[jx] == np.inf:
                    print 'ooooooooooooooooooooooooooooooo\n'
                    print 'Encountered infinite entropy at:\n', pre, pst, bns
                elif verbose:
                    print '-------------------------------\n'
                    print 'Encountered regular entropy at:\n', pre, pst, bns
                else:
                    pass
            all_kls.append(stream_kl)
        return all_kls

    def plot_stagewise(self, metric='logloss', saveas=None, verbose=False, **kwargs):
        """

        :param metric: String. Metric to plot.
        :param saveas: String. Filename to use for savig the figure (UTC timestamp will be appended to the name).
                        If None, only produce plots and do not save.
        :param verbose: Boolean. If true, print messages.
        :param kwargs: dict of keyword arguments for the pandas plotting method.
        :return: Nothing. Side effects only :-)
        """
        ref, upd = self.stagewise_metric(metric=metric, verbose=verbose)
        for ix, ll_ref in enumerate(ref):
            ll_upd = upd[ix]
            x, y = 'reference_' + metric, 'update_' + metric
            df = pd.DataFrame({x: ll_ref, y: ll_upd})
            df.sort_values(by=x, inplace=True)
            label = 'Stream ' + str(ix)
            if ix == 0:

                ax = df.plot(x=x, y=y, color=self.colors[ix%len(ref)], label=label, **kwargs)# kind='barh',
            else:
                ax = df.plot(x=x, y=y, ax=ax, color=self.colors[ix%len(ref)], label=label, **kwargs)
            plt.plot([0., 1.], [0., 1.], color='k', linestyle='--')

        if saveas is not None:
            assert isinstance(saveas, str)
            if '/' in saveas:
                saveas = saveas.strip('.').strip('/')
            name = './figures/' + saveas + str(int(time())) + '.png'
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(1.0)
        plt.close()

    def plot_kl(self, ntiles=10, rule=None, prior=1e-6, verbose=False, saveas=None, **kwargs):
        """
        Plot the KL divergence for all features in all streams.
        :param ntiles: Integer. Number of percentiles to use.
        :param rule: String. The rule to use to compute the distribution bins for the train data.
        :param prior: Float. Prior density value to use for histogram bins with 0 counts.
        :param verbose: Boolean. If tue, show results during execution.
        :param saveas: String. Filename to use for savig the figure (UTC timestamp will be appended to the name).
                        If None, only produce plots and do not save.
        :param kwargs: dict of keyword arguments for the pandas plotting method.
        :return: Nothing. Side effects only :-)
        """
        kls = self.kuhl_leib(ntiles=ntiles, rule=rule, prior=prior, verbose=verbose)
        for ix, kl in enumerate(kls):
            nm = 'Stream ' + str(ix)
            df = pd.DataFrame(data={'feature': ['X'+str(ii)
                                                for ii in range(self.xrefs[0].shape[1])],
                                    nm: [k for k in kl]})
            if verbose:
                print df.head(10)
            label = 'Stream ' + str(ix)
            if ix == 0:
                ax = df.plot(x='feature', y=nm, color=self.colors[ix%len(kls)], label=label, **kwargs)# kind='barh',
            else:
                ax = df.plot(x='feature', y=nm, ax=ax, color=self.colors[ix%len(kls)], label=label, **kwargs)

        if saveas is not None:
            assert isinstance(saveas, str)
            if '/' in saveas:
                saveas = saveas.strip('.').strip('/')
            name = './figures/' + saveas + str(int(time())) + '.png'
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()
            plt.pause(1.0)
        plt.close()

    @staticmethod
    def __hist_plotter(xx, bns, normed, minimal, ix, jx, ix_vals, jx_vals, saveas_plus, x_axis, **kwargs):
        """
        Plot a histogram of xx using preselected bins.
        :param xx: numpy array. The data to plot as a histogram
        :param bns: numpy array. The bins for the histogram
        :param normed: boolean. Normalized histogram
        :param minimal: boolean. If true, remove all axes and marks from the histogram
        :param ix: int. If ix is in ix_vals, make a plot, else not
        :param jx: int. If jx is in jx_vals, make a plot, else not
        :param ix_vals: list of int.
        :param jx_vals: list of int.
        :param saveas_plus: string. String to use in filename for saving the plots
        :param x_axis: list of float. x axis limits
        :param kwargs: arguments to pass to matplotlib plot
        :return: nothing
        """
        if (ix in ix_vals) and (jx in jx_vals):
            plt.hist(xx, bins=bns, normed=normed, **kwargs)
            if x_axis is not None:
                ax = plt.gca()
                ax.set_xlim(x_axis)
            if minimal:
                plt.axis('off')
            else:
                ttl = 'Stream %d, X%d - reference' % (ix, jx)
                plt.title(ttl)
            if saveas_plus is not None:
                assert isinstance(saveas_plus, str)
                if '/' in saveas_plus:
                    saveas_plus = saveas_plus.strip('.').strip('/')
                name = './figures/' + saveas_plus + str(int(time())) + '.png'
                plt.savefig(name, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.pause(1.0)
                plt.close()

    def plot_hist(self, ntiles=10, rule=None, saveas=None,
                  minimal=True, plot_selection=None, x_axis=None,
                  **kwargs):
        """
        Plot histograms for a selection of features from the reference and the new data.
        :param ntiles: int. Number of bins in the histogram
        :param rule: string or None. Rule to pass to np.hist for automatic calculation of the bins
        :param saveas: string. String to use for saving the plots
        :param minimal: boolean. If true, remove all axes, ticks, etc from the plots
        :param plot_selection: tuple of lists of int. The features to plot histograms of.
                            e.g. ([2], [1]) means histograms of feature X1 from stream 2 only
        :param x_axis: list of float. x axis limits
        :param kwargs: arguments to pass to matplotlib plot
        :return: Nothing
        """
        assert rule is None or (rule in ('fd', 'auto'))
        if plot_selection is None:
            ix_vals = range(len(self.xrefs))
            jx_vals = range(self.xrefs[0].shape[1])
        else:
            ix_vals, jx_vals = plot_selection
        for ix, xref in enumerate(self.xrefs):  # loop over streams
            xu = self.xus[ix]
            for jx in range(xref.shape[1]):  # loop over features in the stream
                xxref = xref[:, jx]
                xxu = xu[:, jx]
                if rule is None:   # find the bin sizes
                    pre, bns = np.histogram(xxref, bins=ntiles, normed=True)
                elif rule in ('fd', 'auto'):
                    pre, bns = np.histogram(xxref, bins=rule, normed=True)

                # plot and save
                self.__hist_plotter(xxref, bns, normed=True, minimal=minimal,
                                    ix=ix, jx=jx, ix_vals=ix_vals, jx_vals=jx_vals,
                                    saveas_plus=saveas + 'stream%d-x%d-reference' % (ix, jx), x_axis=x_axis,
                                    **kwargs)

                self.__hist_plotter(xxu, bns, normed=True, minimal=minimal,
                                    ix=ix, jx=jx, ix_vals=ix_vals, jx_vals=jx_vals,
                                    saveas_plus=saveas + 'stream%d-x%d-new' % (ix, jx), x_axis=x_axis,
                                    **kwargs)

import platform
import datetime
import os
from time import time
import numpy as np
import pandas as pd
from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT
from loop_helper import df_init, df_append_events, df_append_score, save_metadata, plot_namer
if "centos" in platform.platform():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

__author__ = "nikos.daniilidis"


def main():
    note = 'parallel'
    readme = '''n parallel streams, all chisq, different balances, same weights'''
    event_types = 3 * ['chisq']  # distribution of the hidden score for each stream
    seed_events = 20  # number of events to use on the first round of training
    update_events = 20  # number of total events occurring in each round of batch update
    analysis_events = 20  # number of events to use on each round of analysis
    ps = [0.47, 0.48, 0.49]  # fraction of class 1 examples in each stream
    seeds = [11, 10, 9]  # random seeds for each stream
    gs = 3 * [1.]  # gains to use in weighing each stream probability
    num_inputs = 10  # number of inputs in each stream
    classifier_kinds = 3 * ['gbm']  # classifier to use
    criterion = 'competing_streams'  # type of selection condition
    batch_updates = 30  # number of batch updates to run for the models
    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)  # will be used for figure names
    datetimestr = datetime.datetime.now().strftime("%Y%B%d-%H%M")
    dirname = os.path.join('runs', str(len(event_types)) + '_streams-' + note + '-' + datetimestr)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_metadata(event_types, seed_events, update_events, analysis_events, ps, seeds, num_inputs,
                  classifier_kinds, criterion, batch_updates, file_descriptor, dirname, note, readme)
    pn = plot_namer(dirname=dirname)

    # EventGenerators
    egs = []
    for ix, event_type in enumerate(event_types):
        egs.append(EG(seed=seeds[ix], num_inputs=num_inputs, kind=event_type, balance=ps[ix]))

    # ModelUpdaters
    mus = []
    for ix, classifier_kind in enumerate(classifier_kinds):
        mus.append(MU(kind=classifier_kind))

    # EventSelector
    es = ES(criterion=criterion)
    # TrainDataUpdaters
    tdu = TDU(num_events=seed_events)
    tdua = TDU(num_events=analysis_events)

    nones = [None for e in event_types]
    xolds = [None for e in event_types]
    yolds = [None for e in event_types]
    xold_ans = [None for e in event_types]
    yold_ans = [None for e in event_types]

    # global behavior: optimal logloss, and KL distributions at each batch update
    ll_cols = ['update_index'] + ['logloss_S%d' % ix for ix, e in enumerate(event_types)]
    kl_cols = ['update_index'] + ['KL_S%d' % ix for ix, e in enumerate(event_types)]
    df_lgls = pd.DataFrame(columns=ll_cols)
    df_kl = pd.DataFrame(columns=kl_cols)

    for batch_update in range(batch_updates):
        if batch_update == 0:  # on the first iteration use seed events, otherwise use update_event
            events = seed_events
        else:
            events = update_events
        # create train stream events
        xrs, yrs = [], []
        for eg in egs:
            xi, yi = eg.get_labeled(events)
            xrs.append(xi)
            yrs.append(yi)
        # create analysis stream events
        xas, yas = [], []
        for eg in egs:
            xi, yi = eg.get_labeled(events)
            xas.append(xi)
            yas.append(yi)

        # pass events through current models filter
        if batch_update == 0:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=nones, event_gains=gs, verbose=True)
            xafs, yafs = es.filter(xs=xas, ys=yas,
                                   models=nones, event_gains=gs)
        else:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=ms, event_gains=gs, verbose=True)
            xafs, yafs = es.filter(xs=xas, ys=yas,
                                   models=ms, event_gains=gs)
            # print xafs[0]
            # print yafs[0]
        msg = ''
        for ix, xi in enumerate(xs):
            msg += str(xi.shape[0]) + '(%3.2f) ' % (1.*sum(ys[ix])/(xi.shape[0]+1e-8))
        print '---- Event Selector (balance in parentheses)----'
        print 'New events at %d:' % batch_update
        print msg

        # update train data
        Xus, Yus = [], []
        for ix, x in enumerate(xs):
            Xi, Yi = tdu.update(xolds[ix], yolds[ix], xs[ix], ys[ix])
            Xus.append(Xi)
            Yus.append(Yi)
        # update analysis data
        Xuas, Yuas = [], []
        for ix, x in enumerate(xafs):
            Xi, Yi = tdu.update(xold_ans[ix], yold_ans[ix], xafs[ix], yafs[ix])
            Xuas.append(Xi)
            Yuas.append(Yi)

        # update models using new data
        ms = []
        for ix, mu in enumerate(mus):
            ms.append(mu.train(Xus[ix], Yus[ix], learning_rate=[0.005, 0.01, 0.03, 0.06, 0.1], n_estimators=[5],
                               subsample=0.5, max_depth=[2, 3], random_state=13, folds=5))
            print ms[ix].best_params_

        # lookahead: pass events through updated models filter
        xafnews, yafnews = es.filter(xs=xas, ys=yas,
                                     models=ms, event_gains=gs, verbose=False)
        #xafnews = xafs
        #yafnews = yafs

        # look at distribution shifts and algorithm performance
        msg = ''
        for ix, x in enumerate(xafs):
            msg += str(x.shape[0]) + '(%3.2f) ' % (1.*sum(yafs[ix])/(x.shape[0]+1e-8))
        print '--- Data Tomographer (balance in parentheses) ---'
        print 'Old model events at %d:' % batch_update
        print msg
        print ''

        # unbiased data vs old biased data on updated model
        dt = DT(xrefs=xafs, yrefs=yafs,
                xus=xas, yus=yas,
                models=ms)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False,
                   saveas=pn('unbiased_feature_kl_' + file_descriptor + str(int(time()))))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('unbiased_stagewise_logloss_' + file_descriptor + str(int(time()))))

        # question: Does the distribution of data through model converge to some value?
        kls = dt.kuhl_leib(ntiles=10, rule='auto', prior=1e-8, verbose=False)
        mean_kls = [np.mean(kl) if kl is not None else np.nan for kl in kls]
        df = pd.DataFrame(data=[[batch_update] + mean_kls], columns=kl_cols)
        df_kl = df_kl.append(df, ignore_index=True)

        # lookahead: old biased data vs new biased data on updated model
        dt = DT(xrefs=xafs, yrefs=yafs,
                xus=xafnews, yus=yafnews,
                models=ms)
        dt.plot_hist(ntiles=10, rule='auto', minimal=True, plot_selection=([2], [9]), x_axis=(-3.5, 3.5),
                     saveas=pn('biased_feature_histogram_' + str(int(time()))), color='b', edgecolor='none', alpha=0.5)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas=pn('biased_feature_kl_'+file_descriptor))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('biased_stagewise_logloss_'+file_descriptor + str(int(time()))))

        # question: Does the logloss on future data converge to some value?
        ll_af, ll_afnew = dt.stagewise_metric(metric='logloss', verbose=False)
        d = [[batch_update] + [lls[-1] if lls is not None else np.nan for lls in ll_afnew]] # this might not be necessary out here
        df = pd.DataFrame(data=d, columns=ll_cols)
        df_lgls = df_lgls.append(df, ignore_index=True)

        # create "old" data for next iteration
        xolds = Xus
        yolds = Yus

        xold_ans = Xuas
        yold_ans = Yuas

        plt.figure()
        df_kl[kl_cols[1:]].plot()
        plt.savefig(pn(event_type + 'mean_kl_' + file_descriptor), bbox_inches='tight')
        plt.close()

        plt.figure()
        df_lgls[ll_cols[1:]].plot()
        plt.savefig(pn(event_type + 'logloss_' + file_descriptor), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()

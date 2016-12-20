import platform
import datetime
import json
import os
from time import time
if "centos" in platform.platform():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT

__author__ = "nikos.daniilidis"


def save_metadata(event_types, seed_events, update_events, analysis_events, ps, seeds, num_inputs,
                  classifier_kind, criterion, batch_updates, file_descriptor, dirname):
    """
    Save the metadata for a run to a json file. See main for parameter explanation.
    """
    d = {'event_types': event_types,
         'seed_events': seed_events,
         'update_events': update_events,
         'analysis_events': analysis_events,
         'ps': ps,
         'seeds': seeds,
         'num_inputs': num_inputs,
         'classifier_kind': classifier_kind,
         'criterion': criterion,
         'batch_updates': batch_updates,
         'file_descriptor': file_descriptor}
    with open(os.path.join(dirname, file_descriptor+'.json'), 'w') as f:
        f.write(json.dumps(d, indent=4))


def plot_namer(dirname, suffix='.png'):
    """
    Minimal utility for formatting plot names.
    """
    return lambda fname: os.path.join(dirname, fname + suffix)


def main():
    event_types = ['chisq', 'chisq', 'chisq']  # distribution of the hidden score for each stream
    seed_events = 500  # number of events to use on the first round of training
    update_events = 1500  # number of total events occurring in each round of batch update
    analysis_events = 1000  # number of events to use on each round of analysis
    ps = [0.4, 0.5, 0.6]  # fraction of class 1 examples in each stream
    seeds = [42, 13, 79]  # random seeds for each stream
    gs = [1., 1., 1.]  # gains to use in weighing each stream probability
    num_inputs = 10  # number of inputs in each stream
    classifier_kinds = ['gbm', 'gbm', 'gbm']  # classifier to use
    criterion = 'competing_streams'  # type of selection condition
    batch_updates = 12  # number of batch updates to run for the models
    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)  # will be used for figure names
    datetimestr = datetime.datetime.now().strftime("%Y%B%d-%H%M")
    dirname = str(len(event_types)) + '_streams-' + '-' + datetimestr
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_metadata(event_types, seed_events, update_events, analysis_events, ps, seeds, num_inputs,
                  classifier_kinds, criterion, batch_updates, file_descriptor, dirname)
    pn = plot_namer(dirname=dirname)

    # EventGenerators
    egs = []
    for ix, event_type in event_types:
        egs.append(EG(seed=seeds[ix], num_inputs=num_inputs, kind=event_type, balance=ps[ix]))

    # ModelUpdaters
    mus = []
    for ix, classifier_kind in classifier_kinds:
        mus.append(MU(kind=classifier_kind))

    # EventSelector
    es = ES(criterion=criterion)
    # TrainDataUpdaters
    tdu = TDU(num_events=seed_events)
    tdua = TDU(num_events=analysis_events)

    xolds = [None for e in event_types]
    yolds = [None for e in event_types]
    xold_ans = [None for e in event_types]
    yold_ans = [None for e in event_types]

    # global behavior: optimal logloss, and KL distributions at each batch update
    ll_cols = ['update_index'] + ['logloss_S%d' % ix for ix, e in event_types]
    kl_cols = ['update_index'] + ['KL_S%d' % ix for ix, e in event_types]
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
            xi, yi = eg.get(events)
            xrs.append(xi)
            yrs.append(yi)
        # create analysis stream events
        xas, yas = [], []
        for eg in egs:
            xi, yi = eg.get(events)
            xas.append(xi)
            yas.append(yi)

        # pass events through current models filter
        if batch_update == 0:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=[None for mu in mus], event_gains=gs)
            xsaf, ysaf = es.filter(x=xas, ys=yas,
                                   models=[None for mu in mus], event_gains=gs)
        else:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=ms, event_gains=gs)
            xsaf, ysaf = es.filter(xs=xas, ys=yas,
                                   models=ms, event_gains=gs)
        msg = ''
        for xi in xs:
            msg += str(xi.shape[0]) + ' '
        print '---- Event Selector ----'
        print 'New events at %d:' % batch_update
        print msg

        #######################################################################################################333
        # update train data
        X1u, Y1u = tdu.update(x1old, y1old, x1, y1)
        X2u, Y2u = tdu.update(x2old, y2old, x2, y2)
        X3u, Y3u = tdu.update(x3old, y3old, x3, y3)
        X1ua, Y1ua = tdua.update(x1old_an, y1old_an, x1af, y1af)
        X2ua, Y2ua = tdua.update(x2old_an, y2old_an, x2af, y2af)
        X3ua, Y3ua = tdua.update(x3old_an, y3old_an, x3af, y3af)

        # update models using new data
        m1 = mu1.train(X1u, Y1u, learning_rate=[0.005, 0.01, 0.03, 0.06, 0.1], n_estimators=[250],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=5)
        m2 = mu2.train(X2u, Y2u, learning_rate=[0.005, 0.01, 0.03, 0.06, 0.1], n_estimators=[250],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=5)
        m3 = mu3.train(X3u, Y3u, learning_rate=[0.005, 0.01, 0.03, 0.06, 0.1], n_estimators=[250],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=5)

        # lookahead: pass events through updated models filter
        xsaf, ysaf = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                               models=(m1, m2, m3), event_gains=ps)
        x1afnew, x2afnew, x3afnew = xsaf
        y1afnew, y2afnew, y3afnew = ysaf

        # look at distribution shifts and algorithm performance
        print '--- Data Tomographer ---'
        print 'Old model events at %d:' % batch_update
        print x1af.shape[0], x2af.shape[0], x3af.shape[0]
        print ''

        # unbiased data vs old biased data on updated model
        dt = DT(xrefs=[x1af, x2af, x3af], yrefs=[y1af, y2af, y3af],
                xus=[x1a, x2a, x3a], yus=[y1a, y2a, y3a],
                models=[m1, m2, m3])
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False,
                   saveas=pn('unbiased_feature_kl_' + file_descriptor + str(int(time()))))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('unbiased_stagewise_logloss_' + file_descriptor + str(int(time()))))
        # question: Does the distribution of data through model converge to some value?
        kls = dt.kuhl_leib(ntiles=10, rule='auto', prior=1e-8, verbose=False)
        mean_kls = [np.mean(kl) for kl in kls]
        df = pd.DataFrame(data=[[batch_update] + mean_kls], columns=kl_cols)
        df_kl = df_kl.append(df, ignore_index=True)

        # lookahead: old biased data vs new biased data on updated model
        dt = DT(xrefs=[x1af, x2af, x3af], yrefs=[y1af, y2af, y3af],
                xus=[x1afnew, x2afnew, x3afnew], yus=[y1afnew, y2afnew, y3afnew],
                models=[m1, m2, m3])
        dt.plot_hist(ntiles=10, rule='auto', minimal=True, plot_selection=([2], [9]), x_axis=(-3.5, 3.5),
                     saveas=pn('biased_feature_histogram_' + str(int(time()))), color='b', edgecolor='none', alpha=0.5)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas=pn('biased_feature_kl_'+file_descriptor))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('biased_stagewise_logloss_'+file_descriptor + str(int(time()))))
        # question: Does the logloss on future data converge to some value?
        ll_af, ll_afnew = dt.stagewise_metric(metric='logloss', verbose=False)
        df = pd.DataFrame(data=[[batch_update] + [lls[-1] for lls in ll_afnew]], columns=ll_cols)
        df_lgls = df_lgls.append(df, ignore_index=True)

        # create "old" data for next iteration
        x1old, x2old, x3old = X1u, X2u, X3u
        y1old, y2old, y3old = Y1u, Y2u, Y3u
        x1old_an, x2old_an, x3old_an = X1ua, X2ua, X3ua
        y1old_an, y2old_an, y3old_an = Y1ua, Y2ua, Y3ua

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
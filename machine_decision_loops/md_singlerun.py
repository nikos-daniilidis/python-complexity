import datetime
import platform
import os
from time import time
import numpy as np
import pandas as pd
from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT
from loop_helper import save_metadata, df_append_score, df_append_events, df_init, plot_namer
if "centos" in platform.platform():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

__author__ = "nikos.daniilidis"


def main_run(note, readme, event_types, seed_events, update_events, analysis_events, ps, noise_spread, seeds,
              gs, num_inputs, classifier_kinds, train_args, criterion, method):
    """
    This is the main loop. Prepares all streams and runs a number of batch updates.
    :param note: str. Note to use in the configuration json
    :param readme: str. Human readable note
    :param event_types: list of string. Distribution of the hidden score for each stream
    :param seed_events: int. Number of events to use on the first round of training (shared among all streams)
    :param update_events: int. Number of total events occurring in each round of batch update (shared among all streams)
    :param analysis_events:int. Number of events to use on each round of analysis (again, shared)
    :param ps: list of float. Fraction of class 1 examples in each stream (class balance)
    :param noise_spread: float. Magnitude of noise to add to the event class, expressed as a fraction of the underlying
                                score
    :param seeds: list of int. Random seeds for each stream
    :param gs: list of float. Gains to use in weighing each stream probability
    :param num_inputs: int. Number of inputs for each stream (all streams take same number of inputs)
    :param classifier_kinds: list of string. Type of classifier to use
    :param train_args: dict. Preset arguments to use for the cross validation part of model train
    :param criterion: string. Type of selection condition (single stream or multiple streams)
    :param method: string. Method used to select between multiple streams
    :return: Nothing
    """

    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)  # will be used for figure names
    # create directory to save results
    datetimestr = datetime.datetime.now().strftime("%Y%B%d-%H%M")
    dirname = os.path.join('runs', str(len(event_types)) + '_streams-' + note + '-' + datetimestr)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_metadata(event_types, seed_events, update_events, analysis_events, ps, noise_spread, seeds, num_inputs,
                  classifier_kinds, criterion, batch_updates, file_descriptor, dirname, note, readme)
    pn = plot_namer(dirname=dirname)
    pn_csv = plot_namer(dirname=dirname, suffix='.csv')

    # EventGenerators
    mother_eg = EG(seed=42, num_inputs=num_inputs, spread=noise_spread)  # this is the mother egg, the source of events for all!
    egs = []
    for ix, event_type in enumerate(event_types):
        egs.append(EG(seed=seeds[ix], num_inputs=num_inputs, kind=event_type, balance=ps[ix]))

    # ModelUpdaters
    mus = []
    for ix, classifier_kind in enumerate(classifier_kinds):
        mus.append(MU(kind=classifier_kind, cv_folds=10, n_jobs=20))

    # EventSelector
    es = ES(criterion=criterion)

    # TrainDataUpdaters
    tdu = TDU(num_events=seed_events)
    tdua = TDU(num_events=analysis_events)  # TODO: Make use of tdua where appropriate

    nones = [None for e in event_types]
    xolds = [None for e in event_types]
    yolds = [None for e in event_types]
    xold_ans = [None for e in event_types]
    yold_ans = [None for e in event_types]

    # data frames to log metrics:
    df_params, c_params = df_init(event_types, 'best_parameters')  # best parameters for each stream
    df_ll_train, cll_train = df_init(event_types, 'logloss_train_Stream')  # log loss on the train data
    df_ll_unb, cll_unb = df_init(event_types, 'logloss_unbiased_Stream')  # log loss on unseen and unbiased data
    df_ll_pre, cll_pre = df_init(event_types, 'logloss_biased_unseen_Stream')  # log loss on unseen data pre-update
    df_ll_post, cll_post = df_init(event_types, 'logloss_biased_future_Stream')  # log loss on unseen data post-update
    df_kl_train, ckl_train = df_init(event_types, 'KL_train_Stream')  # KL divergence of train data wrt unbiased data
    df_n_tr_events, ctr_ev = df_init(event_types, 'num_train_events_Stream')  # num of all train events in the stream
    df_n_tr_success, ctr_succ = df_init(event_types, 'num_train_success_Stream')  # num of all train 1s in the stream
    df_n_events, cn_events = df_init(event_types, 'num_events_Stream')  # num of selected new events in the stream
    df_n_success, cn_success = df_init(event_types, 'num_success_Stream')  # num of successful new events in the stream
    df_n_events_uniq, cn_events_uniq = df_init(event_types, 'num_events_uniq_Stream')  # num of selected
    df_n_success_uniq, cn_success_uniq = df_init(event_types, 'num_success_uniq_Stream')  # num of successful
    #                                                              new events in stream when always choosing this stream
    df_n_events_rand, cn_events_rand = df_init(event_types, 'num_events_random_Stream')  # num of selected
    df_n_success_rand, cn_success_rand = df_init(event_types, 'num_success_random_Stream')  # num of successful
    #                                                              new events in stream when choosing stream at random
    df_n_success_max, cn_event_max = df_init(event_types[:1], 'num_success_max_All')  # num of maximum successful events
    #                                                              when choosing the stream optimally

    # create analysis stream events
    # This set of events is the same for all batch updates, but the assignment changes depending on the models.
    xas, yas = [], []
    xi = mother_eg.get_unlabeled(analysis_events)
    xi, ni = mother_eg.append_noise(xi)
    for eg in egs:
        xi, yi = eg.label(xi, ni)
        xas.append(xi)
        yas.append(yi)

    # main loop
    for batch_update in range(batch_updates):
        if batch_update == 0:  # on the first iteration use seed events, otherwise use update_event
            events = seed_events
        else:
            events = update_events

        # create train stream events
        xrs, yrs = [], []
        xi = mother_eg.get_unlabeled(events)
        xi, ni = mother_eg.append_noise(xi)
        for eg in egs:
            xi, yi = eg.label(xi, ni)
            xrs.append(xi)
            yrs.append(yi)

        # pass events through current models filter
        if batch_update == 0:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=nones, event_gains=gs)
            xafs, yafs = es.filter(xs=xas, ys=yas,
                                   models=nones, event_gains=gs)
            xr, yr = xs, ys  # stream of randomly chosen events is identical to model-chosen events if model is none
        else:
            xs, ys = es.filter(xs=xrs, ys=yrs,
                               models=ms, event_gains=gs)
            xafs, yafs = es.filter(xs=xas, ys=yas,
                                   models=ms, event_gains=gs)
            xr, yr = es.filter(xs=xrs, ys=yrs,
                               models=nones, event_gains=gs)  # stream of randomly chosen events

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
        # If using static analysis events, xold_ans yold_ans should be None
        Xuas, Yuas = [], []
        for ix, x in enumerate(xafs):
            assert xold_ans[ix] is None and yold_ans[ix] is None
            Xi, Yi = tdu.update(xold_ans[ix], yold_ans[ix], xafs[ix], yafs[ix])
            Xuas.append(Xi)
            Yuas.append(Yi)

        # num of selected/successful new events in the stream when using models to select
        df_n_events = df_append_events(df_n_events, y=ys, batch_index=batch_update, kind='events')
        df_n_success = df_append_events(df_n_success, y=ys, batch_index=batch_update, kind='successes')

        # total number of all/successful events in the stream, used for training (old + update)
        df_n_tr_events = df_append_events(df_n_tr_events, y=Xus, batch_index=batch_update, kind='events')
        df_n_tr_success = df_append_events(df_n_tr_success, y=Yus, batch_index=batch_update, kind='successes')

        # number of selected/successful new events in stream when when always choosing this stream
        df_n_events_uniq = df_append_events(df_n_events_uniq, y=yrs, batch_index=batch_update, kind='events')
        df_n_success_uniq = df_append_events(df_n_success_uniq, y=yrs, batch_index=batch_update, kind='successes')

        # number of selected/successful new events in stream when when choosing stream at random
        df_n_events_rand = df_append_events(df_n_events_rand, y=yr, batch_index=batch_update, kind='events')
        df_n_success_rand = df_append_events(df_n_success_rand, y=yr, batch_index=batch_update, kind='successes')

        # number of selected/successful new events in stream when when choosing stream optimally
        df_n_success_max = df_append_events(df_n_success_max, y=yrs, batch_index=batch_update, kind='max_successes')

        # update models using new data
        ms, b_p = [], []  # models, best parameters
        for ix, mu in enumerate(mus):
            ms.append(mu.train(Xus[ix], Yus[ix], train_args))
            b_p.append(ms[ix].best_params_)
            print ms[ix].best_params_

        # record best parameters for each stream
        df = pd.DataFrame(data=[[batch_update] + b_p], columns=c_params)
        df_params = df_params.append(df, ignore_index=True)

        # record log loss on train data for each stream
        df_ll_train = df_append_score(df_ll_train, models=ms, x=Xus, y=Yus, batch_index=batch_update, factor=-1.)
        # print df_ll_train.head()

        # lookahead: pass events through updated models filter
        xafnews, yafnews = es.filter(xs=xas, ys=yas, models=ms, event_gains=gs)

        # look at distribution shifts and algorithm performance
        msg = ''
        for ix, x in enumerate(xafs):
            msg += str(x.shape[0]) + '(%3.2f) ' % (1.*sum(yafs[ix])/(x.shape[0]+1e-8))
        print '--- Data Tomographer (balance in parentheses) ---'
        print 'Old model events at %d:' % batch_update
        print msg
        print ''

        # log loss on unseen, unbiased data
        df_ll_unb = df_append_score(df_ll_unb, x=xas, y=yas, models=ms, batch_index=batch_update, factor=-1.)

        # log loss on unseen data pre-update
        df_ll_pre = df_append_score(df_ll_pre, x=xafs, y=yafs, models=ms, batch_index=batch_update,  factor=-1.)
        # print df_ll_pre.head()

        # log loss on unseen data post-update
        df_ll_post = df_append_score(df_ll_post, x=xafnews, y=yafnews, models=ms, batch_index=batch_update, factor=-1.)

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
        mean_kls = [np.mean(kl) for kl in kls]
        df = pd.DataFrame(data=[[batch_update] + mean_kls], columns=ckl_train)
        df_kl_train = df_kl_train.append(df, ignore_index=True)

        # lookahead: old biased data vs new biased data on updated models
        dt = DT(xrefs=xafs, yrefs=yafs,
                xus=xafnews, yus=yafnews,
                models=ms)

        # dt.plot_hist(ntiles=10, rule='auto', minimal=True, plot_selection=([2], [9]), x_axis=(-3.5, 3.5),
        #             saveas=pn('biased_feature_histogram_' + str(int(time()))), color='b', edgecolor='none', alpha=0.5)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas=pn('biased_feature_kl_'+file_descriptor))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('biased_stagewise_logloss_'+file_descriptor + str(int(time()))))

        # create "old" data for next iteration
        xolds = Xus
        yolds = Yus

        # no need to update the analysis events if using static analysis events. They shall always be None
        # xold_ans = Xuas
        # yold_ans = Yuas

        # plot and save KL divergences
        plt.figure()
        df_kl_train[ckl_train[1:]].plot()
        plt.savefig(pn(event_type + 'mean_kl_' + file_descriptor), bbox_inches='tight')
        plt.close()

        # plot and save log loss on train data
        plt.figure()
        df_ll_train[cll_train[1:]].plot()
        plt.savefig(pn(event_type + 'logloss_' + file_descriptor), bbox_inches='tight')
        plt.close()

        # save metrics to csv files
        df_params.to_csv(pn_csv(event_type + '_best_params_' + file_descriptor), index=False)
        df_ll_train.to_csv(pn_csv(event_type + '_logloss_train_' + file_descriptor), index=False)
        df_ll_unb.to_csv(pn_csv(event_type + '_logloss_unbiased_' + file_descriptor), index=False)
        df_ll_pre.to_csv(pn_csv(event_type + '_logloss__biased_unseen_' + file_descriptor), index=False)
        df_ll_post.to_csv(pn_csv(event_type + '_logloss_biased_future_' + file_descriptor), index=False)
        df_kl_train.to_csv(pn_csv(event_type + '_mean_kl_' + file_descriptor), index=False)
        df_n_events.to_csv(pn_csv(event_type + '_num_events_' + file_descriptor), index=False)
        df_n_success.to_csv(pn_csv(event_type + '_num_success_' + file_descriptor), index=False)
        df_n_tr_events.to_csv(pn_csv(event_type + '_num_train_events_' + file_descriptor), index=False)
        df_n_tr_success.to_csv(pn_csv(event_type + '_num_train_success_' + file_descriptor), index=False)
        df_n_events_uniq.to_csv(pn_csv(event_type + '_num_events_uniq_' + file_descriptor), index=False)
        df_n_success_uniq.to_csv(pn_csv(event_type + '_num_success_uniq_' + file_descriptor), index=False)
        df_n_events_rand.to_csv(pn_csv(event_type + '_num_events_rand_' + file_descriptor), index=False)
        df_n_success_rand.to_csv(pn_csv(event_type + '_num_success_rand_' + file_descriptor), index=False)
        df_n_success_max.to_csv(pn_csv(event_type + '_num_success_max_' + file_descriptor), index=False)

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


def save_metadata(event_types, seed_events, update_events, analysis_events, ps, noise_spread, seeds, num_inputs,
                  classifier_kind, criterion, batch_updates, file_descriptor, dirname, note, readme):
    """
    Save the metadata for a run to a json file. See main for parameter explanation.
    """
    d = {'event_types': event_types,
         'seed_events': seed_events,
         'update_events': update_events,
         'analysis_events': analysis_events,
         'ps': ps,
         'noise_spread': noise_spread,
         'seeds': seeds,
         'num_inputs': num_inputs,
         'classifier_kind': classifier_kind,
         'criterion': criterion,
         'batch_updates': batch_updates,
         'file_descriptor': file_descriptor,
         'note': note,
         'readme': readme}
    with open(os.path.join(dirname, file_descriptor+'.json'), 'w') as f:
        f.write(json.dumps(d, indent=4))


def df_init(stream_list, data_kind):
    """
    Initialize empty DataFrame
    :param stream_list: list of length equal to the number of streams
    :param data_kind: string. The type of informarion stored in this dataframe
    :return:
    """
    cols = ['update_index'] + \
           [data_kind + str(ix) for ix, e in enumerate(stream_list)]  # list of column names
    return pd.DataFrame(columns=cols), cols


def df_append_score(df_in, models, x, y, batch_index, factor=1.):
    """
    Update a data frame of score values for each model using provided data (score is defined by GridSearchCV object).
    :param df_in: data frame. Input data frame
    :param models: list of sklearn GridSearch classifiers
    :param x: list of numpy array. The X values for each stream
    :param y: list of numpy array. The y values for each stream
    :param batch_index: int. index to use in the 'update_index' column of the data frame
    :param factor: float. factor to multiply scores by
    :return: df_in. DataFrame with one appended row (index, and log loss on each stream. )
    """
    assert (abs(abs(factor)-1.)) < 1e-9
    scores = []
    cols = df_in.columns.tolist()
    for ix, model in enumerate(models):
        scores.append(factor*model.score(x[ix], y[ix]))
    df = pd.DataFrame(data=[[batch_index] + scores], columns=cols)
    df_in = df_in.append(df, ignore_index=True)
    return df_in


def df_append_events(df_in, y, batch_index, kind):
    """
    Append the number of events or the number of successful events for each stream in the stream labels y.
    :param df_in: dataframe. Gets updated with number of events for each stream
    :param y: list of numpy array. Labels of events for each stream
    :param batch_index: int. index to use in the 'update_index' column of the data frame
    :param kind: string.
    :return:
    """
    ns = []
    cols = df_in.columns.tolist()
    if kind == 'events':
        for yi in y:
            ns.append(yi.shape[0])
    elif kind == 'successes':
        for yi in y:
            ns.append(sum(yi))
    else:
        'df_append_events encountered unknown "kind" option'
        return df_in
    df = pd.DataFrame(data=[[batch_index] + ns], columns=cols)
    df_in = df_in.append(df, ignore_index=True)
    return df_in


def plot_namer(dirname, suffix='.png'):
    """
    Minimal utility for formatting plot names.
    """
    return lambda fname: os.path.join(dirname, fname + suffix)


def main():
    note = 'shared-x'
    readme = '''n streams sharing x values, all chisq, different balances, same weights'''
    event_types = 3 * ['chisq']  # distribution of the hidden score for each stream
    seed_events = 20  # number of events to use on the first round of training
    update_events = 20  # number of total events occurring in each round of batch update
    analysis_events = 20  # number of events to use on each round of analysis
    ps = [0.47, 0.48, 0.49]  # fraction of class 1 examples in each stream
    noise_spread = 0.01  # magnitude of noise to add to the event class, expressed as a fraction of the underlying score
    seeds = [11, 10, 9]  # random seeds for each stream
    gs = 3 * [1.]  # gains to use in weighing each stream probability
    num_inputs = 10  # number of inputs in each stream
    classifier_kinds = 3 * ['gbm']  # classifier to use
    criterion = 'competing_streams'  # type of selection condition
    batch_updates = 10  # number of batch updates to run for the models
    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)  # will be used for figure names

    # create directory to save results
    datetimestr = datetime.datetime.now().strftime("%Y%B%d-%H%M")
    dirname = str(len(event_types)) + '_streams-' + note + '-' + datetimestr
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
        mus.append(MU(kind=classifier_kind, cv_folds=4, n_jobs=-1))

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
    df_n_events, cn_events = df_init(event_types, 'num_events_Stream')  # num of selected new events in the stream
    df_n_success, cn_success = df_init(event_types, 'num_success_Stream')  # num of successful new events in the stream
    df_n_events_max, cn_events_max = df_init(event_types, 'num_events_max_Stream')  # num of selected
    df_n_success_max, cn_success_max = df_init(event_types, 'num_success_max_Stream')  # num of successful
    # new events in stream when always choosing this stream
    df_n_events_rand, cn_events_rand = df_init(event_types, 'num_events_random_Stream')  # num of selected
    df_n_success_rand, cn_success_rand = df_init(event_types, 'num_success_random_Stream')  # num of successful
    # new events in stream when choosing stream at random

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
        # create analysis stream events
        xas, yas = [], []
        xi = mother_eg.get_unlabeled(events)  # TODO: Consider moving this outside the loop to always have same analysis
        xi, ni = mother_eg.append_noise(xi)
        for eg in egs:
            xi, yi = eg.label(xi, ni)
            xas.append(xi)
            yas.append(yi)

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

        # num of selected/successful new events in the stream when using models to select
        df_n_events = df_append_events(df_n_events, y=ys, batch_index=batch_update, kind='events')
        df_n_success = df_append_events(df_n_success, y=ys, batch_index=batch_update, kind='successes')

        # number of selected/successful new events in stream when when always choosing this stream
        df_n_events_max = df_append_events(df_n_events_max, y=yrs, batch_index=batch_update, kind='events')
        df_n_success_max = df_append_events(df_n_success_max, y=yrs, batch_index=batch_update, kind='successes')

        # number of selected/successful new events in stream when when choosing stream at random
        df_n_events_rand = df_append_events(df_n_events_rand, y=yr, batch_index=batch_update, kind='events')
        df_n_success_rand = df_append_events(df_n_success_rand, y=yr, batch_index=batch_update, kind='successes')

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
        ms, b_p = [], []  # models, best parameters
        for ix, mu in enumerate(mus):
            ms.append(mu.train(Xus[ix], Yus[ix], learning_rate=[0.005, 0.01, 0.03, 0.06, 0.1], n_estimators=[5],
                               subsample=0.5, max_depth=[2, 3], random_state=13, scoring='neg_log_loss'))
            print ms[ix].best_params_

        # double check that cv optimizes log_loss
        print "------------ gals, here comes the cowboy ----------------"
        dt = DT(xrefs=Xus, yrefs=Yus,
                xus=Xus, yus=Yus,
                models=ms)
        ll, ll_new = dt.stagewise_metric(metric='logloss', verbose=False)
        df = pd.DataFrame(data=[[batch_update] + [lls[-1] for lls in ll_new]], columns=cll_train)
        df_ll_train = df_ll_train.append(df, ignore_index=True)
        print df_ll_train.head()
        print "------------------- eeeeeeee haah -----------------------"

        # record best parameters for each stream
        df = pd.DataFrame(data=[[batch_update] + b_p], columns=c_params)
        df_params = df_params.append(df, ignore_index=True)

        # record log loss on train data for each stream
        df_ll_train = df_append_score(df_ll_train, models=ms, x=Xus, y=Yus, batch_index=batch_update, factor=-1.)
        print df_ll_train.head()

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

        # lookahead: old biased data vs new biased data on updated model
        dt = DT(xrefs=xafs, yrefs=yafs,
                xus=xafnews, yus=yafnews,
                models=ms)
        dt.plot_hist(ntiles=10, rule='auto', minimal=True, plot_selection=([2], [9]), x_axis=(-3.5, 3.5),
                     saveas=pn('biased_feature_histogram_' + str(int(time()))), color='b', edgecolor='none', alpha=0.5)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas=pn('biased_feature_kl_'+file_descriptor))
        dt.plot_stagewise(metric='logloss', verbose=False,
                          saveas=pn('biased_stagewise_logloss_'+file_descriptor + str(int(time()))))

        ### question: Does the log-loss on future data converge to some value?
        ##ll_af, ll_afnew = dt.stagewise_metric(metric='logloss', verbose=False)
        ##df = pd.DataFrame(data=[[batch_update] + [lls[-1] for lls in ll_afnew]], columns=cll_train)
        ##df_ll_train = df_ll_train.append(df, ignore_index=True)
        ##df_ll_train = df_append_score(df_ll_train, models=ms, x=xafnews, y=yafnews, batch_index=batch_update)
        ##print df_ll_train.head()

        # create "old" data for next iteration
        xolds = Xus
        yolds = Yus

        xold_ans = Xuas
        yold_ans = Yuas

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
    df_params.to_csv(pn_csv((event_type + '_best_params_' + file_descriptor)))
    df_ll_train.to_csv(pn_csv((event_type + '_logloss_train_' + file_descriptor)))
    df_ll_unb.to_csv(pn_csv((event_type + '_logloss_unbiased_' + file_descriptor)))
    df_ll_pre.to_csv(pn_csv((event_type + '_logloss__biased_unseen_' + file_descriptor)))
    df_ll_post.to_csv(pn_csv((event_type + '_logloss_biased_future_' + file_descriptor)))
    df_kl_train.to_csv(pn_csv((event_type + '_mean_kl_' + file_descriptor)))
    df_n_events.to_csv((event_type + '_num_events_' + file_descriptor))
    df_n_success.to_csv((event_type + '_num_success_' + file_descriptor))
    df_n_events_max.to_csv((event_type + '_num_events_max_' + file_descriptor))
    df_n_success_max.to_csv((event_type + '_num_success_max_' + file_descriptor))
    df_n_events_rand.to_csv((event_type + '_num_events_rand_' + file_descriptor))
    df_n_success_rand.to_csv((event_type + '_num_success_rand_' + file_descriptor))


if __name__ == '__main__':
    main()

import platform
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


def main():
    seed_events = 500
    update_events = 300
    analysis_events = 1000
    p1, p2, p3 = 0.4, 0.5, 0.6
    assert seed_events >= update_events
    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)  # will be used for figure names

    # EventGenerators
    eg1 = EG(seed=42, num_inputs=10, kind='chisq', balance=p1)
    eg2 = EG(seed=13, num_inputs=10, kind='chisq', balance=p2)
    eg3 = EG(seed=79, num_inputs=10, kind='chisq', balance=p3)

    # ModelUpdaters
    mu1 = MU(kind='gbm')
    mu2 = MU(kind='gbm')
    mu3 = MU(kind='gbm')

    # EventSelector
    es = ES(criterion='competing_streams')
    # TrainDataUpdaters
    tdu = TDU(num_events=seed_events)
    tdua = TDU(num_events=analysis_events)

    x1old, x2old, x3old = None, None, None
    y1old, y2old, y3old = None, None, None
    x1old_an, x2old_an, x3old_an = None, None, None
    y1old_an, y2old_an, y3old_an = None, None, None

    # global behavior: optimal logloss, and KL distributions at each batch update
    ll_cols = ['update_index', 'logloss_S1', 'logloss_S2', 'logloss_S3']
    kl_cols = ['update_index', 'KL_S1', 'KL_S2', 'KL_S3']
    df_lgls = pd.DataFrame(columns=ll_cols)
    df_kl = pd.DataFrame(columns=kl_cols)

    for chunk in range(12):
        # create train stream events
        if chunk == 0: # on the first iteration use seed events, otherwise use update_event
            events = seed_events
        else:
            events = update_events
        x1r, y1r = eg1.get(events)
        x2r, y2r = eg2.get(events)
        x3r, y3r = eg3.get(events)
        # create analysis stream events
        x1a, y1a = eg1.get(analysis_events)
        x2a, y2a = eg2.get(analysis_events)
        x3a, y3a = eg3.get(analysis_events)

        # pass events through current models filter
        if chunk == 0:
            xs, ys = es.filter(xs=(x1r, x2r, x3r), ys=(y1r, y2r, y3r),
                               models=(None, None, None), event_gains=(p1, p2, p3))
            xsaf, ysaf = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                                 models=(None, None, None), event_gains=(p1, p2, p3))
        else:
            xs, ys = es.filter(xs=(x1r, x2r, x3r), ys=(y1r, y2r, y3r),
                               models=(m1, m2, m3), event_gains=(p1, p2, p3))
            xsaf, ysaf = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                                 models=(m1, m2, m3), event_gains=(p1, p2, p3))
        x1, x2, x3 = xs
        y1, y2, y3 = ys
        x1af, x2af, x3af = xsaf
        y1af, y2af, y3af = ysaf
        print '---- Event Selector ----'
        print 'New events at %d:' % chunk
        print x1.shape[0], x2.shape[0], x3.shape[0]

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
                               models=(m1, m2, m3), event_gains=(p1, p2, p3))
        x1afnew, x2afnew, x3afnew = xsaf
        y1afnew, y2afnew, y3afnew = ysaf

        # look at distribution shifts and algorithm performance
        print '--- Data Tomographer ---'
        print 'Old model events at %d:' % chunk
        print x1af.shape[0], x2af.shape[0], x3af.shape[0]
        print ''

        # unbiased data vs old biased data on updated model
        dt = DT(xrefs=[x1af, x2af, x3af], yrefs=[y1af, y2af, y3af],
                xus=[x1a, x2a, x3a], yus=[y1a, y2a, y3a],
                models=[m1, m2, m3])
        #dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='unbiased_feature_kl_'+file_descriptor)
        #dt.plot_stagewise(metric='logloss', verbose=False, saveas='unbiased_stagewise_logloss_'+file_descriptor)
        # question: Does the distribution of data through model converge to some value?
        kls = dt.kuhl_leib(ntiles=10, rule='auto', prior=1e-8, verbose=False)
        median_kls = [np.median(kl) for kl in kls]
        df = pd.DataFrame(data=[[chunk] + median_kls], columns=kl_cols)
        df_kl = df_kl.append(df, ignore_index=True)

        # lookahead: old biased data vs new biased data on updated model
        dt = DT(xrefs=[x1af, x2af, x3af], yrefs=[y1af, y2af, y3af],
                xus=[x1afnew, x2afnew, x3afnew], yus=[y1afnew, y2afnew, y3afnew],
                models=[m1, m2, m3])
        #dt.plot_hist(ntiles=10, rule='auto', minimal=True, plot_selection=([2], [9]), x_axis=(-3.5, 3.5),
        #             saveas='biased_feature_histogram_', color='b', edgecolor='none', alpha=0.5)
        #dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='biased_feature_kl_'+file_descriptor)
        #dt.plot_stagewise(metric='logloss', verbose=False, saveas='biased_stagewise_logloss_'+file_descriptor)
        # question: Does the logloss on future data converge to some value?
        ll_af, ll_afnew = dt.stagewise_metric(metric='logloss', verbose=False)
        df = pd.DataFrame(data=[[chunk] + [lls[-1] for lls in ll_afnew]], columns=ll_cols)
        df_lgls = df_lgls.append(df, ignore_index=True)

        # create "old" data for next iteration
        x1old, x2old, x3old = X1u, X2u, X3u
        y1old, y2old, y3old = Y1u, Y2u, Y3u
        x1old_an, x2old_an, x3old_an = X1ua, X2ua, X3ua
        y1old_an, y2old_an, y3old_an = Y1ua, Y2ua, Y3ua

    plt.figure()
    df_kl[kl_cols[1:]].plot()
    plt.savefig('./figures/median_kl_' + file_descriptor + '.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    df_lgls[ll_cols[1:]].plot()
    plt.savefig('./figures/logloss_' + file_descriptor + '.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
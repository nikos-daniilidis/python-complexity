from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT

__author__ = "nikos.daniilidis"

def main():
    seed_events = 100
    update_events = 30
    analysis_events = 1000
    p1, p2, p3 = 0.5, 0.5, 0.5

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
    atdu = TDU(num_events=analysis_events)

    # create events
    X1, Y1 = eg1.get(seed_events)
    X2, Y2 = eg2.get(seed_events)
    X3, Y3 = eg3.get(seed_events)

    # train models
    m1 = mu1.train(X1, Y1,learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
    m2 = mu2.train(X2, Y2, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
    m3 = mu3.train(X3, Y3, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)

    for chunk in range(10):
        # create events
        x1r, y1r = eg1.get(update_events)
        x2r, y2r = eg2.get(update_events)
        x3r, y3r = eg3.get(update_events)

        # pass events through current models filter
        xs, ys = es.filter(xs=(x1r, x2r, x3r), ys=(y1r, y2r, y3r),
                           models=(m1, m2, m3), event_gains=(p1, p2, p3))
        x1, x2, x3 = xs
        y1, y2, y3 = ys
        print 'New events at %d:' % chunk
        print x1.shape[0], x2.shape[0], x3.shape[0]

        # update train data
        X1u, Y1u = tdu.update(X1, Y1, x1, y1)
        X2u, Y2u = tdu.update(X2, Y2, x2, y2)
        X3u, Y3u = tdu.update(X3, Y3, x3, y3)

        # update models using new data
        m1o, m2o, m3o = m1, m2, m3

        m1 = mu1.train(X1u, Y1u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
        m2 = mu2.train(X2u, Y2u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
        m3 = mu3.train(X3u, Y3u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)

        # create "old" data for next iteration
        X1, X2, X3 = X1u, X2u, X3u
        Y1, Y2, Y3 = Y1u, Y2u, Y3u

        # look at distribution shifts and algorithm performance
        print '--- Data Tomographer ---'
        # create events
        x1a, y1a = eg1.get(analysis_events)
        x2a, y2a = eg2.get(analysis_events)
        x3a, y3a = eg3.get(analysis_events)

        # pass events through updated models filter
        xs, ys = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                           models=(m1o, m2o, m3o), event_gains=(1., 1., 1.))
        x1o, x2o, x3o = xs
        y1o, y2o, y3o = ys
        print 'Old model events at %d:' %chunk
        print x1o.shape[0], x2o.shape[0], x3o.shape[0]

        # pass events through updated models filter
        xs, ys = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                           models=(m1, m2, m3), event_gains=(1., 1., 1.))
        x1, x2, x3 = xs
        y1, y2, y3 = ys
        print 'New model events at %d:' %chunk
        print x1.shape[0], x2.shape[0], x3.shape[0]

        dt = DT([x1o, x2o, x3o], [y1o, y2o, y3o], [x1, x2, x3], [y1, y2, y3], [m1o, m2o, m3o])
        file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='feature_kl_'+file_descriptor)
        dt.plot_stagewise(metric='logloss', verbose=False, saveas='stagewise_logloss_'+file_descriptor)


def analyze(eg1, eg2, eg3, analysis_events, es, m1, m2, m3, m1u, m2u, m3u, chunk, seed_events, update_events):
    print '--- Data Tomographer ---'
    # create events
    x1a, y1a = eg1.get(analysis_events)
    x2a, y2a = eg2.get(analysis_events)
    x3a, y3a = eg3.get(analysis_events)

    # pass events through models filter
    xs, ys = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                       models=(m1, m2, m3), event_gains=(1., 1., 1.))
    x1, x2, x3 = xs
    y1, y2, y3 = ys
    print 'Current model events at %d:' % chunk
    print x1.shape[0], x2.shape[0], x3.shape[0]

    dt = DT([x1a, x2a, x3a], [y1a, y2a, y3a], [x1, x2, x3], [y1, y2, y3], [m1, m2, m3])
    file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)
    dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='unbiased_feature_kl_'+file_descriptor)
    dt.plot_stagewise(metric='logloss', verbose=False, saveas='unbiased_stagewise_logloss_'+file_descriptor)

    if chunk > 0:
        assert m1u is not None
        assert m2u is not None
        assert m3u is not None
        # pass events through models filter
        xs, ys = es.filter(xs=(x1a, x2a, x3a), ys=(y1a, y2a, y3a),
                           models=(m1u, m2u, m3u), event_gains=(1., 1., 1.))
        x1u, x2u, x3u = xs
        y1u, y2u, y3u = ys
        print 'Updated model events at %d:' % chunk
        print x1u.shape[0], x2u.shape[0], x3u.shape[0]

        dt = DT([x1, x2, x3], [y1, y2, y3], [x1u, x2u, x3u], [y1u, y2u, y3u], [m1u, m2u, m3u])
        file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='unbiased_feature_kl_'+file_descriptor)
        dt.plot_stagewise(metric='logloss', verbose=False, saveas='unbiased_stagewise_logloss_'+file_descriptor)


if __name__=='__main__':
    main()
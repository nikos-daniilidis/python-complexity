from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT

__author__ = "nikos.daniilidis"

def main():
    seed_events = 500
    update_events = 100

    eg1 = EG(seed=42, num_inputs=10, kind='chisq', balance=0.5)
    eg2 = EG(seed=13, num_inputs=10, kind='chisq', balance=0.5)
    eg3 = EG(seed=79, num_inputs=10, kind='chisq', balance=0.5)

    X1, Y1 = eg1.get(seed_events)
    X2, Y2 = eg2.get(seed_events)
    X3, Y3 = eg3.get(seed_events)

    mu1 = MU()
    mu2 = MU()
    mu3 = MU()

    m1 = mu1.train(X1, Y1, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
    m2 = mu2.train(X2, Y2, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
    m3 = mu3.train(X3, Y3, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                   subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)

    es = ES(criterion='competing_streams')
    tdu = TDU(num_events=update_events)

    for chunk in range(10):
        x1r, y1r = eg1.get(update_events)
        x2r, y2r = eg2.get(update_events)
        x3r, y3r = eg3.get(update_events)

        xs, ys = es.filter(xs=(x1r, x2r, x3r), ys=(y1r, y2r, y3r),
                           models=(m1, m2, m3), event_gains=(1., 1., 1.))
        x1, x2, x3 = xs
        y1, y2, y3 = ys
        print 'New events at %d:' %chunk
        print x1.shape[0], x2.shape[0], x3.shape[0]

        X1u, Y1u = tdu.update(X1, Y1, x1, y1)
        X2u, Y2u = tdu.update(X2, Y2, x2, y2)
        X3u, Y3u = tdu.update(X3, Y3, x3, y3)

        dt = DT([X1, X2, X3], [Y1, Y2, Y3], [X1u, X2u, X3u], [Y1u, Y2u, Y3u], [m1, m2, m3])
        file_descriptor = 'seed%d_update%d_' % (seed_events, update_events)
        dt.plot_kl(ntiles=10, rule='auto', prior=1e-8, verbose=False, saveas='feature_kl_'+file_descriptor)
        dt.plot_stagewise(metric='logloss', verbose=False, saveas='stagewise_logloss_'+file_descriptor)

        m1 = mu1.train(X1u, Y1u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
        m2 = mu2.train(X2u, Y2u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)
        m3 = mu3.train(X3u, Y3u, learning_rate=[0.01, 0.03, 0.1], n_estimators=[50, 100, 150, 200, 300],
                       subsample=0.5, max_depth=[2, 3], random_state=13, folds=3)

        X1, X2, X3 = X1u, X2u, X3u
        Y1, Y2, Y3 = Y1u, Y2u, Y3u

if __name__=='__main__':
    main()
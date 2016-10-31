from EventGenerator import EventGenerator as EG
from ModelUpdater import ModelUpdater as MU
from EventSelector import EventSelector as ES
from TrainDataUpdater import TrainDataUpdater as TDU
from DataTomographer import DataTomographer as DT

__author__ = "nikos.daniilidis"

def main():
    eg1 = EG(seed=42, num_inputs=10, kind='chisq', balance=0.4)
    eg2 = EG(seed=42, num_inputs=10, kind='chisq', balance=0.5)
    eg3 = EG(seed=42, num_inputs=10, kind='chisq', balance=0.6)

    X1, Y1 = eg1.get(12)
    X2, Y2 = eg2.get(12)
    X3, Y3 = eg3.get(12)

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
    tdu = TDU(num_events=100)

    for chunk in range(10):
        x1r, y1r = eg1.get(100)
        x2r, y2r = eg2.get(100)
        x3r, y3r = eg3.get(100)

        xs, ys = es.filter(xs=(x1r, x2r, x3r), ys=(y1r, y2r, y3r),
                           models=(m1, m2, m3), event_gains=(1./0.4, 1./0.5, 1./0.6))
        x1, x2, x3 = xs
        y1, y2, y3 = ys

        X1u, Y1u = tdu.update(X1, Y1, x1, y1)
        X2u, Y2u = tdu.update(X1, Y1, x1, y1)
        X3u, Y3u = tdu.update(X1, Y1, x1, y1)

        dt = DT([X1, X2, X3], [Y1, Y2, Y3], [X1u, X2u, X3u], [Y1u, Y2u, Y3u], [m1, m2, m3])
        dt.plot_kl(rule='fd')

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
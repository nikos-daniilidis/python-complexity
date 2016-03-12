import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from ComplexAgents import ComplexAgent


def get_data(agent_rule=30, agent_size=50, num=300, bounded=True, fraction_rows=1., fraction_columns=1., seed=42):
    ca = ComplexAgent(rule=agent_rule, time_range=agent_size, num_blocks=num, bounded=bounded, seed=seed)
    ca.start_random()
    ca.cycle()
    x = np.concatenate((ca.column_sums(columns=range(int(fraction_columns * (ca.get_block_width()-2))),
                                       exclude_last=True),
                        ca.row_sums(rows=range(int(fraction_rows * (ca.get_time_range()-1))))
                        ), axis=1)
    y = ca.output_balance()
    return x, y


def update_data(x_old, y_old, x_new, y_new, model, threshold=0.3):
    ix_keep = np.where(model.predict_proba(x_new)[:, 0] > threshold)[0]
    return np.concatenate((x_old, x_new[ix_keep, ]), axis=0), np.concatenate((y_old, y_new[ix_keep, ]), axis=0)


def filter_data(x, y, model, threshold=0.3):
    ix_keep = np.where(model.predict_proba(x)[:, 0] > threshold)[0]
    return x[ix_keep, ], y[ix_keep, ]


def update_model(model, Xi, yi, Xt, yt, metrics, num=1000, threshold=0.3, seed=42):
    Xn, yn = get_data(num=num, seed=seed)
    X, y = update_data(Xi, yi, Xn, yn, model, threshold=threshold)
    clf = GradientBoostingClassifier(n_estimators=N_EST, learning_rate=L_RT)
    clf.fit(X, y)
    y_pred = clf.predict(Xt)
    auc = roc_auc_score(yt, y_pred)
    logloss = log_loss(yt, y_pred)
    print "ROC-AUC: %4.3f" % auc
    print "log-loss: %4.3f" % logloss
    metrics = pd.concat([metrics, pd.DataFrame({"AUC": [auc], "LogLoss": [logloss],
                                                "Threshold": [threshold], "Num_Rows": [X.shape[0]]})])
    return X, y, metrics


if __name__ == "__main__":
    """
    EXAMPLE OUTPUT
            AUC    LogLoss  Num_Rows  Threshold
    0  0.657104  11.881496       500        0.0
    0  0.665103  11.587906       563        0.3
    0  0.664566  11.622451       633        0.3
    0  0.652012  12.054190       704        0.3
    0  0.646015  12.278702       777        0.3
    0  0.667700  11.553389       842        0.3
    0  0.657010  11.933324       909        0.3
    0  0.650548  12.157829       973        0.3
    0  0.648829  12.209635      1037        0.3
    0  0.649162  12.175087      1109        0.3
    0  0.659432  11.829698      1175        0.3
    0  0.667838  11.536114      1239        0.3
    0  0.663863  11.674271      1309        0.3
    0  0.670724  11.432495      1378        0.3
    0  0.669665  11.467033      1446        0.3
    0  0.680674  11.087102      1518        0.3
    0  0.664065  11.605168       600        0.0
    0  0.672790  11.311588       700        0.0
    0  0.654268  11.967839       800        0.0
    0  0.647370  12.192338       900        0.0
    0  0.667975  11.518839      1000        0.0
    0  0.653195  12.036930      1100        0.0
    0  0.647828  12.244174      1200        0.0
    0  0.671101  11.432500      1300        0.0
    0  0.660629  11.777885      1400        0.0
    0  0.673125  11.346146      1500        0.0
    0  0.674785  11.294339      1600        0.0
    0  0.673241  11.346147      1700        0.0
    0  0.680645  11.087102      1800        0.0
    0  0.686854  10.879870      1900        0.0
    0  0.689138  10.793519      2000        0.0
    """
    N_EST = 400
    L_RT = 0.05
    X, y = get_data(num=500, seed=42)
    X_test, y_test = get_data(num=2000, seed=98765421)
    clf = GradientBoostingClassifier(n_estimators=N_EST, learning_rate=L_RT)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    print "Initial ROC-AUC: %4.3f" % auc
    print "Initial log-loss: %4.3f" % logloss
    metrics = pd.DataFrame({"AUC": [auc], "LogLoss": [logloss],
                            "Threshold": [0.], "Num_Rows": [X.shape[0]]})

    for sd in [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000, 1111, 1222, 1333, 1444, 1555]:
        X, y, metrics = update_model(clf, X, y, X_test, y_test, metrics, num=100, threshold=0.3, seed=sd)

    print "\nThreshold <- 0"

    X, y = get_data(num=500, seed=42)
    for sd in [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000, 1111, 1222, 1333, 1444, 1555]:
        X, y, metrics = update_model(clf, X, y, X_test, y_test, metrics, num=100, threshold=0., seed=sd)

    print metrics.head(50)

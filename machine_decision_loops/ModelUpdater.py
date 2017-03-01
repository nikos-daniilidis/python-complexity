from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from loop_helper import safe_fit
from config import GBM, RF, LOGISTIC

__author__ = "nikos.daniilidis"


class ModelUpdater:
    def __init__(self, kind=GBM, cv_folds=10, n_jobs=10):
        self.kind = kind
        self.folds = cv_folds
        self.n_jobs = n_jobs

    def train(self, x, y, train_dict, safe=True, **kwargs):
        if safe:
            try:
                return self.raw_train(x, y, train_dict, **kwargs)
            except ValueError:
                return None  # this error should catch most cases where training data is too small (single class)
        else:
            return self.raw_train(x, y, train_dict, **kwargs)

    def raw_train(self, x, y, train_dict, **kwargs):
        if self.kind == GBM:
            if train_dict is not None:
                learning_rate = train_dict['learning_rate']
                n_estimators = train_dict['n_estimators']
                max_depth = train_dict['max_depth']
                subsample = train_dict['subsample']
                random_state = train_dict['random_state']
                scoring = train_dict['scoring']
                return self.__train_gbm(x, y, learning_rate=learning_rate, n_estimators=n_estimators,
                                        max_depth=max_depth, subsample=subsample, random_state=random_state,
                                        scoring=scoring, **kwargs)
            else:
                return self.__train_gbm(x, y, **kwargs)
        elif self.kind == RF:
            if train_dict is not None:
                n_estimators = train_dict['n_estimators']
                max_depth = train_dict['max_depth']
                random_state = train_dict['random_state']
                class_weight = train_dict['class_weight']
                scoring = train_dict['scoring']
                return self.__train_random_forest(x, y, n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=random_state, class_weight=class_weight,
                                                  scoring=scoring, **kwargs)
            else:
                return self.__train_random_forest(x, y, **kwargs)
        elif self.kind == LOGISTIC:
            if train_dict is not None:
                cs = train_dict['cs']
                scoring = train_dict['scoring']
                return self.__train_logistic(x, y, cs=cs, scoring=scoring, **kwargs)
            else:
                return self.__train_logistic(x, y, **kwargs)

    def __train_gbm(self, x, y, learning_rate=(0.01, 0.03, 0.1), n_estimators=(50, 100, 150, 200, 300),
                    max_depth=(2, 3), subsample=0.5, random_state=13, **kwargs):
        parameters = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth}
        folds = min(self.folds, len(y))
        clf = GridSearchCV(GradientBoostingClassifier(random_state=random_state,
                                                      subsample=subsample),
                           param_grid=parameters, cv=folds, n_jobs=self.n_jobs, **kwargs)
        clf = safe_fit(clf, x, y, safe_min=folds)  # fit adding dummy elements of minority class if too few
        return clf

    def __train_random_forest(self, x, y, n_estimators=(50, 100, 150, 200, 300), max_depth=(2, 3, 4, 5),
                              min_samples_split=20, min_samples_leaf=10,
                              random_state=13, class_weight=None, **kwargs):
        parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth}
        folds = min(self.folds, len(y))
        clf = GridSearchCV(RandomForestClassifier(random_state=random_state, class_weight=class_weight,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf),
                           param_grid=parameters, cv=folds, n_jobs=self.n_jobs, **kwargs)
        clf = safe_fit(clf, x, y, safe_min=folds)  # fit adding dummy elements of minority class if too few
        return clf

    def __train_logistic(self, x, y, cs=(0.01, 0.1, 1., 10), **kwargs):
        parameters = {'C': cs}
        folds = min(self.folds, len(y))
        clf = GridSearchCV(LogisticRegression(),
                           param_grid=parameters, cv=folds, n_jobs=self.n_jobs, **kwargs)
        clf = safe_fit(clf, x, y, safe_min=folds)  # fit adding dummy elements of minority class if too few
        return clf

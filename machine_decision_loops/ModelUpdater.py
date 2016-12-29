from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

__author__ = "nikos.daniilidis"


class ModelUpdater:
    def __init__(self, kind='gbm', cv_folds=10, n_jobs=10):
        self.kind = kind
        self.folds = cv_folds
        self.n_jobs = n_jobs

    def train(self, x, y, **kwargs):
        if self.kind== 'gbm':
            return self.__train_gbm(x, y, **kwargs)

    def __train_gbm(self, x, y, learning_rate=(0.01, 0.03, 0.1), n_estimators=(50, 100, 150, 200, 300),
                    max_depth=(2, 3), subsample=0.5, random_state=13, **kwargs):
        parameters = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth}
        clf = GridSearchCV(GradientBoostingClassifier(random_state=random_state,
                                                      subsample=subsample),
                           param_grid=parameters, cv=self.folds, n_jobs=self.n_jobs, **kwargs)
        clf.fit(x, y)
        return clf

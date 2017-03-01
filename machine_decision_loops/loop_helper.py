import json
import os
import sys
import numpy as np
import pandas as pd
import sklearn
import warnings


__author__ = "nikos.daniilidis"


def save_metadata(event_types, seed_events, update_events, analysis_events, ps, noise_spread, seeds, num_inputs,
                  classifier_kind, ve_config, criterion, method, batch_updates, file_descriptor, dirname, note,
                  tau_dict, readme):
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
         'variance_estimator': ve_config,
         'criterion': criterion,
         'method': method,
         'tau_dict': tau_dict,
         'batch_updates': batch_updates,
         'file_descriptor': file_descriptor,
         'note': note,
         'readme': readme,
         'sklearn': sklearn.__version__,
         'numpy': np.__version__,
         'pandas': pd.__version__,
         'python': sys.version}
    with open(os.path.join(dirname, file_descriptor+'.json'), 'w') as f:
        f.write(json.dumps(d, indent=4))


def df_init(stream_list, data_kind):
    """
    Initialize empty DataFrame
    :param stream_list: list of length equal to the number of streams
    :param data_kind: string. The type of information stored in this dataframe
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
        if len(np.unique(y[ix])) <= 1:  # handle cases where array has only one class, or has zero length
            scores.append(np.nan)
        else:
            try:
                scores.append(factor*model.score(x[ix], y[ix]))
            except ValueError:  # handle ValueErrors I did not predict
                print ("ValueError encountered at stream %d, batch update %d" % (ix, batch_index))
                scores.append(np.nan)
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

    Usage:
    y1 = np.array([1,0,0,0])
    y2 = np.array([1,1,0,0])
    y3 = np.array([1,1,1,0])
    y4 = np.array([1,1,1,0])
    y = [y1, y2, y3, y4]
    df, l = df_init('chisq', 'num_success_max_All')
    df_append_events(df, y, 0, 'max_successes')

    """
    ns = []
    cols = df_in.columns.tolist()
    if kind == 'events':
        for yi in y:
            ns.append(yi.shape[0])
    elif kind == 'successes':
        for yi in y:
            ns.append(sum(yi))
    elif kind == 'max_successes':
        # check that all streams of labels have same length in this case
        lengths = [yi.shape[0] for yi in y]
        for ix, l in enumerate(lengths[:-1]):
            assert l == lengths[ix+1]
        Y = np.column_stack(y)
        ns = [np.sum(np.max(Y, axis=1))]
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


def safe_fit(clf, x, y, safe_min=1, n=20):
    """
    Fit a classifier with checks to avoid bad class errors. If the labels contain a single class, fit the classifier
    with the label of the first n elements flipped.
    :param clf: scikit classifier to be trained
    :param x: numpy array. The feature vectors
    :param y: numpy array. The labels
    :param safe_min: int. Minimum number of elements in each class
    :param n: int. Number of elements to flip
    :return: trained classifier
    """
    assert "fit" in dir(clf)
    y_safe = y
    if len(y[y == 0]) < safe_min or len(y[y == 1]) < safe_min:
        n = min(n, len(y)/2)
        if n > safe_min:
            warnings.warn("safe_fit is flipping %d labels of y, while safe_min is set to %d" % (n, safe_min))
        y_safe[:n] = 1. - y_safe[:n]
    print (len(y), len(np.unique(y)), n)
    clf.fit(x, y_safe)
    return clf

__author__ = 'brandonkelly'

import numpy as np
import pandas as pd
import os

base_dir = os.environ['HOME'] + '/Projects/Kaggle/big_data_combine/'


def boxcox(x):
    if np.any(x < 0):
        u = x
    elif np.any(x == 0):
        lamb = 0.5
        u = (x ** lamb - 1.0) / lamb
    else:
        u = np.log(x)

    return u


def build_dataframe(file=None):

    # grab the data
    df = pd.read_csv(base_dir + 'data/' + '1.csv')
    df['day'] = pd.Series(len(df[df.columns[0]]) * [1])
    df['time index'] = pd.Series(df.index)
    df = df.set_index(['day', 'time index'])
    files = [str(d) + '.csv' for d in range(2, 511)]
    for f in files:
        print 'Getting data for day ' + f.split('.')[0] + '...'
        this_df = pd.read_csv(base_dir + 'data/' + f)
        this_df['day'] = pd.Series(len(df[this_df.columns[0]]) * [int(f.split('.')[0])])
        this_df['time index'] = pd.Series(this_df.index)
        this_df = this_df.set_index(['day', 'time index'])
        df = df.append(this_df)

    # find the columns corresponding to the securities and predictors
    colnames = df.columns
    feature_index = [c[0] == 'I' for c in colnames]
    nfeatures = np.sum(feature_index)
    security_index = [c[0] == 'O' for c in colnames]
    nsecurities = np.sum(security_index)

    feature_labels = []
    for c in colnames:
        if c[0] == 'I':
            feature_labels.append(c)

    for f in feature_labels:
        print 'Transforming data for ', f
        df[f] = df[f].apply(boxcox)

    if file is not None:
        df.to_pickle(file)

    return df

if __name__ == "__main__":
    build_dataframe(base_dir + 'data/BDC_dataframe.p')
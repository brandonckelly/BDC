__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hmlinmae_gibbs as hmlin
import os
import multiprocessing as mp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import cPickle

base_dir = os.environ['HOME'] + '/Projects/Kaggle/big_data_combine/'
plags = 3
ndays = 510
ntime = 54

# get the data
fname = base_dir + 'data/BDC_dataframe.p'
df = pd.read_pickle(fname)

train_file = base_dir + 'data/trainLabels.csv'
train = pd.read_csv(train_file)
train = train.drop(train.columns[0], axis=1)

ntrain = len(train)

# global MCMC parameters
nsamples = 10000
nburnin = 10000
nthin = 5
tdof = 10000


def build_submission_file(yfit, snames, filename):

    header = 'FileID'
    for s in snames:
        header += ',' + s
    header += '\n'

    # check for zero standard deviation
    for i in xrange(yfit.shape[0]):
        for j in xrange(yfit.shape[1]):
            # stock j on day i
            ystd = np.std(df[snames[j]].ix[200 + i + 1][1:])
            if ystd < 1e-6:
                yfit[i, j] = df[snames[j]].ix[200 + i + 1, 54]

    submission = np.insert(yfit, 0, np.arange(201, 511), axis=1)

    submission_file = base_dir + 'data/submissions/' + filename
    sfile = open(submission_file, 'w')
    sfile.write(header)
    for i in xrange(submission.shape[0]):
        this_row = str(submission[i, 0])
        for j in xrange(1, submission.shape[1]):
            this_row += ',' + str(submission[i, j])
        this_row += '\n'
        sfile.write(this_row)

    sfile.close()


def boost_residuals(args):
    resid, stock_idx = args

    X, Xpredict = build_design_matrix(stock_idx)

    gbr = GradientBoostingRegressor(loss='lad', max_depth=2, subsample=0.5, learning_rate=0.001,
                                    n_estimators=400)

    gbr.fit(X, resid)

    oob_error = -np.cumsum(gbr.oob_improvement_)
    #plt.plot(oob_error)
    #plt.show()

    ntrees = np.max(np.array([np.argmin(oob_error) + 1, 5]))

    print "Using ", ntrees, " trees for stock ", cnames[stock_idx]

    gbr.n_estimators = ntrees

    # get cross-validation accuracy
    print "Getting CV error..."
    cv_error = cross_validation.cross_val_score(gbr, X, y=resid, score_func=mean_absolute_error,
                                                cv=10)

    gbr.fit(X, resid)

    fimportance = gbr.feature_importances_
    fimportance /= fimportance.max()

    pfile_name = base_dir + 'data/GBR_O' + str(stock_idx+1) + '.p'
    pfile = open(pfile_name, 'wb')
    cPickle.dump(gbr, pfile)
    pfile.close()

    return gbr.predict(Xpredict), cv_error, fimportance


def build_design_matrix(stock_idx):

    # construct array of predictors
    fnames = []
    for f in df.columns:
        if f[0] == 'I':
            fnames.append(f)

    cnames = df.columns

    npredictors = len(fnames)
    two_hours = 24

    thisX = np.empty((ntrain, npredictors))
    thisXpredict = np.empty((ndata - ntrain, npredictors))
    for j in xrange(len(fnames)):
        thisX[:, j] = df[fnames[j]].ix[:, 54][:ntrain]
        thisXpredict[:, j] = df[fnames[j]].ix[:, 54][ntrain:]

    # remove day 22
    thisX = np.delete(thisX, 21, axis=0)

    return thisX, thisXpredict


if __name__ == "__main__":

    # get the stock labels
    snames = []
    for c in df.columns:
        if c[0] == 'O':
            snames.append(c)

    nstocks = len(snames)
    ndata = ndays

    # construct the response arrays
    y = []
    for i in xrange(nstocks):
        thisy = np.empty(ndata)
        thisy[:ntrain] = train[snames[i]]
        thisy[ntrain:] = df[snames[i]].ix[:, 54][ntrain:]
        thisy = np.delete(thisy, (21, 421), axis=0)
        y.append(thisy)

    # construct the predictor arrays
    two_hours = 24
    args = []
    print 'Building data arrays...'
    mfeat = 1 + plags
    X = []
    for i in xrange(nstocks):
        thisX = np.empty((ndata, mfeat))
        thisX[:, 0] = 1.0  # first column corresponds to constant
        for j in xrange(plags):
            thisX[:ntrain, j + 1] = df[snames[i]].ix[:, 54 - j][:ntrain]
            thisX[ntrain:, j + 1] = df[snames[i]].ix[:, 54 - two_hours - j][ntrain:]
        thisX = np.delete(thisX, (21, 421), axis=0)
        X.append(thisX)

    # run the MCMC sampler to get linear predictors based on previous values
    samples = hmlin.run_gibbs(y, X, nsamples, nburnin, nthin, tdof)

    sfile = open(base_dir + 'data/linmae_samples.p', 'wb')
    cPickle.dump(samples, sfile)
    sfile.close()

    print 'Getting predictions from MCMC samples ...'

    # boost residuals from predicted values at 2pm and 4pm for the training set, and 2pm for the test set
    y = np.empty((ndays + ntrain, nstocks))
    yfit = np.empty((ndays + ntrain, nstocks))
    ysubmit = np.empty((ndays - ntrain, nstocks))
    Xsubmit = np.empty((ndays - ntrain, mfeat, nstocks))
    Xfit = np.empty((ndays + ntrain, mfeat, nstocks))
    # Xfit[0:ndata, :, :] = the predictors for the 2pm values for the entire data set
    # Xfit[ndata:, :, :] = the predictors for the 4pm values for the training data set

    ndata = ndays

    for i in xrange(nstocks):
        print '... ', snames[i], ' ...'
        y[:ndays, i] = df[snames[i]].ix[:, 54]  # value at 2pm
        y[ndays:, i] = train[snames[i]]  # value at 4pm
        Xfit[:, 0, i] = 1.0  # first column corresponds to constant
        Xsubmit[:, 0, i] = 1.0
        for j in xrange(plags):
            # value at 12pm
            Xfit[:ndays, j + 1, i] = df[snames[i]].ix[:, 54 - j - two_hours]
            # value at 2pm
            Xfit[ndays:, j + 1, i] = df[snames[i]].ix[:, 54 - j][:ntrain]
            Xsubmit[:, j + 1, i] = df[snames[i]].ix[:, 54 - j][ntrain:]

    for d in xrange(len(snames)):
        for k in xrange(yfit.shape[0]):
            ypredict, ypvar = samples.predict(Xfit[k, :, d], d)
            yfit[k, d] = np.median(ypredict)
        for k in xrange(ysubmit.shape[0]):
            ypredict, ypvar = samples.predict(Xsubmit[k, :, d], d)
            ysubmit[k, d] = np.median(ypredict)

    build_submission_file(ysubmit, snames, 'hmlin_mae.csv')

    resid = y - yfit
    resid = resid[ndata:, :]
    #remove days 22 and 422
    resid = np.delete(resid, 21, axis=0)

    # compare histogram of residuals with expected distribution
    print 'Comparing histogram of residuals against model distributions...'
    for d in xrange(len(snames)):
        this_resid = y[:, d] - yfit[:, d]
        # rmax = np.percentile(this_resid, 0.99)
        # rmin = np.percentile(this_resid, 0.01)
        # rrange = rmax - rmin
        # rmax += 0.05 * rrange
        # rmin -= 0.05 * rrange
        plt.clf()
        n, bins, patches = plt.hist(this_resid, bins=30, normed=True)
        bins = np.linspace(np.min(bins), np.max(bins), 100)
        sigsqr = samples.get_samples('sigsqr ' + str(d))
        pdf = np.zeros(len(bins))
        for b in xrange(len(bins)):
            pdf[b] = np.mean(1.0 / 2.0 / np.sqrt(sigsqr) * np.exp(-np.abs(bins[b]) / np.sqrt(sigsqr)))
        plt.plot(bins, pdf, 'r-', lw=2)
        plt.savefig(base_dir + 'plots/residual_distribution_' + snames[d] + '.png')
        plt.close()

    # plot model values vs true values at 4pm

    sidx = 0
    for s in snames:
        plt.clf()
        plt.plot(yfit[ndata:, sidx], train[s], 'b.')
        xlower = np.percentile(train[s], 1.0)
        xupper = np.percentile(train[s], 99.0)
        xr = xupper - xlower
        plt.xlim(xlower - 0.05 * xr, xupper + 0.05 * xr)
        plt.ylim(xlower - 0.05 * xr, xupper + 0.05 * xr)
        plt.plot(plt.xlim(), plt.xlim(), 'r-', lw=2)
        sidx += 1
        plt.xlabel('Estimated value at 4pm')
        plt.ylabel('True value at 4pm')
        plt.savefig(base_dir + 'plots/model_vs_true_' + s + '.png')
        plt.close()


    # construct array of predictors
    fnames = []
    for f in df.columns:
        if f[0] == 'I':
            fnames.append(f)

    # now gradient boost the residuals
    print "Fitting gradient boosted trees..."

    cnames = df.columns
    npredictors = 2 * (len(df.columns) - 1)
    args = []
    for i in xrange(nstocks):
        args.append((resid[:, i], i))

    pool = mp.Pool(mp.cpu_count()-1)
    results = pool.map(boost_residuals, args)

    #print np.mean(abs(resid[:, 1]))
    #results = boost_residuals(args[1])

    print 'Training error:', np.mean(abs(resid))

    cverror = 0.0
    fimportance = 0.0
    for r in results:
        cverror += np.mean(r[1])
        fimportance += r[2]

    fimportance /= nstocks
    sort_idx = fimportance.argsort()
    print "Sorted feature importances: "
    for s in sort_idx:
        print fnames[s], fimportance[s]

    print ''
    print 'CV error is: ', cverror / len(results)

    idx = 0
    for r in results:
        # add AR(p) contribution back in
        ysubmit[:, idx] = r[0] + ysubmit[:, idx]
        idx += 1

    subfile = 'hmlin_mae_boost.csv'
    build_submission_file(ysubmit, snames, subfile)

    # compare submission file with last observed value as a sanity check
    dfsubmit = pd.read_csv(base_dir + 'data/submissions/' + subfile)

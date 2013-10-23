__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import lib_hmlinmae as maeLib
import yamcmcpp


class LinMAESample(yamcmcpp.MCMCSample):

    def __init__(self, y, X):
        super(LinMAESample, self).__init__()
        self.y = y
        self.X = X
        self.mfeat = X[0].shape[1]
        self.nobjects = len(y)

    def generate_from_file(self, filename):
        pass

    def generate_from_trace(self, trace):
        pass

    def set_logpost(self, logpost):
        pass

    def predict(self, x_predict, obj_idx, nsamples=None):
        if nsamples is None:
            # Use all of the MCMC samples
            nsamples = self.nsamples
            index = np.arange(nsamples)
        else:
            try:
                nsamples <= self.nsamples
            except ValueError:
                "nsamples must be less than the total number of MCMC samples."

            nsamples0 = self.nsamples
            index = np.arange(nsamples) * (nsamples0 / nsamples)

        beta = self.get_samples('coefs ' + str(obj_idx))[index, :]
        sigsqr = self.get_samples('sigsqr ' + str(obj_idx))[index]

        y_predict = beta.dot(x_predict)

        return y_predict, sigsqr


def run_gibbs(y, X, nsamples, nburnin, nthin=1, tdof=8):

    nobjects = len(y)

    mfeat = X[0].shape[1]

    # convert from numpy to vec3D format needed for C++ extension
    X3d = maeLib.vec3D()
    y2d = maeLib.vec2D()
    for i in xrange(nobjects):
        y_i = y[i]
        X_i = X[i]
         # store response in std::vector<std::vector<double> >
        y1d = maeLib.vec1D()
        y1d.extend(y_i)
        y2d.append(y1d)
        X2d = maeLib.vec2D()
        ndata = y_i.size
        for j in xrange(ndata):
            # store predictors in std::vector<std::vector<std::vector<double> > >
            X1d = maeLib.vec1D()
            X1d.extend(X_i[j, :])
            X2d.append(X1d)
        X3d.append(X2d)

    # run the gibbs sampler
    Sampler = maeLib.MaeGibbs(tdof, y2d, X3d)  # C++ gibbs sampler object
    Sampler.RunMCMC(nsamples, nburnin, nthin)

    print "Getting MCMC samples..."

    # grab the MCMC samples and store them in a python class
    samples = LinMAESample(y, X)

    samples._samples['coefs mean'] = np.empty((nsamples, mfeat))
    samples._samples['sigsqr mean'] = np.empty(nsamples)
    for d in xrange(nobjects):
        samples._samples['weights ' + str(d)] = np.empty(nsamples)
        samples._samples['coefs ' + str(d)] = np.empty((nsamples, mfeat))
        samples._samples['sigsqr ' + str(d)] = np.empty(nsamples)

    samples.parameters = samples._samples.keys()
    samples.nsamples = nsamples

    print "Storing MCMC samples..."

    trace = Sampler.GetCoefsMean()
    samples._samples['coefs mean'] = np.asarray(trace)
    trace = Sampler.GetNoiseMean()
    samples._samples['sigsqr mean'] = np.asarray(trace)

    for d in xrange(nobjects):
        if d % 100 == 0:
            print "...", d, "..."
        trace = Sampler.GetWeights(d)
        samples._samples['weights ' + str(d)] = np.asarray(trace)
        trace = Sampler.GetCoefs(d)
        samples._samples['coefs ' + str(d)] = np.asarray(trace)
        trace = Sampler.GetSigSqr(d)
        samples._samples['sigsqr ' + str(d)] = np.asarray(trace)

    return samples


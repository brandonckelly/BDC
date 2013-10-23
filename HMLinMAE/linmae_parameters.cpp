//
//  parameters.cpp
//  HMVAR
//
//  Created by Brandon Kelly on 9/13/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include "linmae_parameters.hpp"
#include <random.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/gamma.hpp>

using namespace HMLinMAE;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

double tWeights::StartingValue()
{
    double w = RandGen.scaled_inverse_chisqr(dof, 1.0);
    return w;
}

double tWeights::RandomPosterior()
{
    arma::vec beta_cent = pCoef_->Value() - pCoefMean_->Value();
    arma::mat beta_prec = arma::inv(arma::sympd(pCoefVar_->Value()));
    double zsqr = arma::as_scalar(beta_cent.t() * beta_prec * beta_cent);
    int post_dof = mfeat + dof;
    double post_ssqr = (zsqr + dof) / post_dof;
    double w = RandGen.scaled_inverse_chisqr(post_dof, post_ssqr);
    return w;
}

arma::vec CoefsMean::StartingValue()
{
    arma::vec mu(value_.n_elem);
    arma::mat this_covar = pCoefsVar_->Value();
    mu = RandGen.normal(this_covar);
    return mu;
}

arma::vec CoefsMean::RandomPosterior()
{
    int nobjects = pCoefs_.size();
    arma::vec mu(value_.n_elem);
    double wsum = 0.0;
    arma::vec mean_beta(mfeat);
    mean_beta.zeros();
    for (int i=0; i<nobjects; i++) {
        arma::vec this_beta = pCoefs_[i]->Value();
        wsum += 1.0 / pWeights_[i]->Value();
        mean_beta += this_beta / pWeights_[i]->Value();
    }
    mean_beta /= wsum;
    arma::mat this_covar = pCoefsVar_->Value() / wsum;
    mu = mean_beta + RandGen.normal(this_covar);
    return mu;
}

arma::mat CoefsVar::StartingValue()
{
    arma::mat varmat = RandGen.inv_wishart(100 * prior_dof, prior_scale);
    return varmat;
}

arma::mat CoefsVar::RandomPosterior()
{
    int post_dof = prior_dof + pCoefs_.size();
    int nobjects = pCoefs_.size();
    arma::mat Smat(prior_scale.n_rows, prior_scale.n_cols);
    Smat.zeros();
    arma::vec mu = pCoefsMean_->Value();
    for (int i=0; i<nobjects; i++) {
        arma::vec this_beta = pCoefs_[i]->Value();
        arma::vec beta_cent = this_beta - mu;
        Smat += beta_cent * beta_cent.t() / pWeights_[i]->Value();
    }
    arma::mat post_scale = Smat + prior_scale;
    arma::mat Sigma = RandGen.inv_wishart(post_dof, Smat + prior_scale);
    return Sigma;
}

// just do a random draw from the prior
double NoiseMean::StartingValue()
{
    double mu_ssqr = RandGen.scaled_inverse_chisqr(4, 1.0);
    return mu_ssqr;
}

double NoiseMean::RandomPosterior()
{
    int nobjects = pNoise_.size();
    double mu_ssqr;

    double ssqr_sum = 0.0;
    for (int i=0; i<nobjects; i++) {
        ssqr_sum += log(pNoise_[i]->Value());
    }
    double log_mu_ssqr = RandGen.normal(ssqr_sum / nobjects, sqrt(pNoiseVar_->Value() / nobjects));
    mu_ssqr = exp(log_mu_ssqr);
    
    return mu_ssqr;
}

// just do a random draw from the prior
double NoiseVar::StartingValue()
{
    return RandGen.scaled_inverse_chisqr(prior_dof_, prior_ssqr_);
}

double NoiseVar::RandomPosterior()
{
    int data_dof = pNoise_.size();
    int post_dof = prior_dof_ + data_dof;
    
    double ssqr = 0.0;
    for (int i=0; i<pNoise_.size(); i++) {
        double ncent = log(pNoise_[i]->Value()) - log(pNoiseMean_->Value());
        ssqr += ncent * ncent;
    }
    ssqr /= data_dof;
    
    double post_ssqr = (prior_dof_ * prior_ssqr_ + data_dof * ssqr) / post_dof;
    return RandGen.scaled_inverse_chisqr(post_dof, post_ssqr);
}

// do random draw from conditional posterior
arma::vec Coefs::StartingValue()
{
    arma::vec beta(mfeat);
    double weights = pWeights_->Value();
    for (int j=0; j<mfeat; j++) {
        arma::vec mu = pCoefMean_->Value();
        arma::mat popvar = weights * pCoefVar_->Value();
        // posterior precision matrix of VAR(p) coefficients corresponding y_j
        arma::mat Bprec = X_.t() * X_;
        Bprec += arma::eye(popvar.n_rows, popvar.n_cols); // add in contribution from population variance
        // get cholesky factor
        arma::mat Bvar;
        try {
            Bvar = arma::inv(arma::sympd(Bprec));
        } catch (std::runtime_error& e) {
            std::cout << "Caught runtime error when trying to compute starting values for coefficients: " << e.what() << std::endl;
            std::cout << "Just using the identity matrix for starting value covariance..." << std::endl;
            Bvar = arma::eye(Bprec.n_rows, Bprec.n_cols);
        }

        arma::mat Ubeta = arma::chol(Bvar); // upper triangular matrix returned
        
        // get posterior mean for coefficients
        arma::vec post_mean = Bvar * (X_.t() * y_ + popvar.i() * mu);
        // draw coefficients from posterior, a multivariate normal
        arma::vec snorm(mfeat);
        
        for (int k=0; k<snorm.n_elem; k++) {
            snorm(k) = RandGen.normal();
        }
        beta = post_mean + Ubeta.t() * snorm;
    }
    
    return beta;
}

// return conditional posterior of beta given population values and data
double Coefs::LogDensity(arma::vec beta)
{
    arma::vec mu = pCoefMean_->Value();
    arma::mat covar = pCoefVar_->Value();
    double sigsqr = pNoise_->Value();
    double w = pWeights_->Value();
    
    arma::vec yhat = X_ * beta;
    // get residual MAE
    double rad = arma::norm(y_ - yhat, 1);
    // add log-likelihood contribution
    double loglik = -0.5 * y_.n_rows * log(sigsqr) - rad / sqrt(sigsqr);
    // add log-population level contribution to log-posterior, a log-normal distribution
    arma::vec beta_cent = beta - mu;
    double logprior = -0.5 * log(arma::det(w * covar)) -
        0.5 * arma::as_scalar(beta_cent.t() * arma::inv(arma::sympd(w * covar)) * beta_cent);

    double logpost = logprior + loglik;
    
    return logpost;
}


// for initial value just draw from scaled-inverse chi-square, ignoring the prior distribution
double Noise::StartingValue()
{
    arma::vec beta = pCoef_->Value();
    arma::mat y = pCoef_->GetY();
    arma::mat X = pCoef_->GetXmat();

    double sigsqr;
    // get residual sum of squares for y_j
    double rss = arma::norm(y - X * beta, 2);
    rss *= rss / y.n_rows;
    sigsqr = RandGen.scaled_inverse_chisqr(y.n_rows, rss);
    
    return sigsqr;
}

// return conditional posterior of white noise variance given population values and estimated time series
double Noise::LogDensity(double sigsqr)
{
    arma::vec beta = pCoef_->Value();
    arma::mat y = pCoef_->GetY();
    arma::mat X = pCoef_->GetXmat();
    double ssqr_mean = pNoiseMean_->Value();
    double ssqr_var = pNoiseVar_->Value();
    
    // get sum of absolute deviations for y_j
    double rad = arma::norm(y - X * beta, 1);
    // add log-likelihood contribution
    double loglik = -0.5 * pCoef_->GetY().n_rows * log(sigsqr) - rad / sqrt(sigsqr);
    // add log-population level contribution to log-posterior, a log-normal distribution
    double logprior = -log(sigsqr) - 0.5 * (log(sigsqr) - log(ssqr_mean)) * (log(sigsqr) - log(ssqr_mean)) / ssqr_var;
    
    double logpost = loglik + logprior;
    if (sigsqr <= 0.0) {
        // don't let sigsqr go negative
        logpost = -arma::datum::inf;
    }
    
    return logpost;
}
//
//  unit_test.cpp
//  HMLinMAE
//
//  Created by Brandon Kelly on 9/27/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

//#define CATCH_CONFIG_MAIN
//#include <catch.hpp>

#include <iostream>
#include <vector>
#include <armadillo>
#include <random.hpp>
#include <boost/math/distributions.hpp>
#include "linmae_parameters.hpp"
#include "MaeGibbs.hpp"

using namespace HMLinMAE;

// Global random number generator object, instantiated in random.cpp
extern boost::random::mt19937 rng;

// Object containing some common random number generators.
extern RandomGenerator RandGen;

int main(int argc, const char * argv[])
{
    double cmean = 2.0;
    double bmean = 1.0;
    double logssqr_mean = log(0.3);
    double logssqr_var = 0.1;
    double cvar = 0.3 * 0.3;
    double bvar = 0.05 * 0.05;
    
    int ndata0 = 300;
    int nobjects = 200;
    int mfeat = 6;
    
    std::vector<vec2d> Xmats;
    std::vector<vec1d> Yvecs;
    std::vector<arma::vec> betas;
    std::vector<double> sigsqrs;
    for (int i=0; i<nobjects; i++) {
        int ndata = 0;
        if (i == nobjects - 1) {
            ndata = 200;
        } else {
            ndata = ndata0;
        }
        arma::mat thisX = arma::zeros(ndata, mfeat);
        thisX.col(0) = arma::ones(ndata);
        for (int k=0; k<mfeat-1; k++) {
            thisX.col(k+1) = 3.0 * 2.0 * arma::randn(ndata);
        }
        vec2d stdX;
        for (int j=0; j<ndata; j++) {
            vec1d Xrow = arma::conv_to<vec1d>::from(thisX.row(j));
            stdX.push_back(Xrow);
        }
        Xmats.push_back(stdX);
        arma::vec this_beta(mfeat);
        this_beta.zeros();
        this_beta(0) = RandGen.normal(cmean, sqrt(cvar));
        this_beta(1) = RandGen.normal(bmean, sqrt(bvar));
        betas.push_back(this_beta);
        double this_ssqr = exp(RandGen.normal(logssqr_mean, sqrt(logssqr_var)));
        sigsqrs.push_back(this_ssqr);
        arma::vec unifs = arma::randu(ndata) - 0.5;
        arma::vec thisY = thisX * this_beta - sqrt(this_ssqr) * unifs / arma::abs(unifs) % arma::log(1.0 - 2.0 * arma::abs(unifs));
        vec1d stdY = arma::conv_to<vec1d>::from(thisY);
        Yvecs.push_back(stdY);
    }
    
    int tdof = 8;
    MaeGibbs Model(tdof, Yvecs, Xmats);
    
    int nsamples = 10000;
    int nburn = 20000;
    int nthin = 5;
    
    ////// RUN THE GIBBS SAMPLER ///////
    
    Model.RunMCMC(nsamples, nburn, nthin);
    
    
    
    // make sure each value of betas are within 3-sigma of true values
    arma::running_stat_vec<double> mcmc_samples(true);
    boost::math::chi_squared_distribution<> chisqr_dist(mfeat);
    double lower_bound = boost::math::quantile(chisqr_dist, 0.01);
    double upper_bound = boost::math::quantile(chisqr_dist, 0.99);
    
    for (int i=0; i<nobjects; i++) {
        mcmc_samples.reset();
        vec2d bsamples = Model.GetCoefs(i); // bsamples is of dimension [nsamples][mfeat]
        for (int k=0; k<nsamples; k++) {
            mcmc_samples(arma::conv_to<arma::vec>::from(bsamples[k]));
        }
        arma::vec post_mean = mcmc_samples.mean();
        arma::mat post_cov = mcmc_samples.cov();
        arma::vec post_cent = post_mean - betas[i];
        double zsqr = arma::as_scalar(post_cent.t() * arma::inv(arma::sympd(post_cov)) * post_cent);
        
        if ((zsqr > upper_bound) || (zsqr < lower_bound)) {
            std::cout << "Coefficient test failed for object " << i << std::endl;
            std::cout << "Reduced chis-square: " << zsqr / mfeat << std::endl;
            post_mean.print("posterior mean: ");
            betas[i].print("true value: ");
            mcmc_samples.stddev().print("posterior stdev: ");
        }
        
        // now test sigsqr
        arma::vec ssqr_samples(Model.GetSigSqr(i));
        ssqr_samples = arma::log(ssqr_samples);
        double post_smean = arma::mean(ssqr_samples);
        double post_stdev = arma::stddev(ssqr_samples);
        double zscore = (post_smean - log(sigsqrs[i])) / post_stdev;
        
        if (std::abs(zscore) > 3.0) {
            std::cout << "Noise Variance test failed for object " << i << std::endl;
            std::cout << "Z-score: " << zscore << std::endl;
            std::cout << "log true value: " << log(sigsqrs[i]) << std::endl;
            std::cout << "posterior mean: " << post_smean << std::endl;
            std::cout << "posterior stdev: " << post_stdev << std::endl;
            
        }
    }
    
    mcmc_samples.reset();
    vec2d musamples = Model.GetCoefsMean(); // musamples is of dimension [nsamples][mfeat]
    for (int k=0; k<nsamples; k++) {
        mcmc_samples(arma::conv_to<arma::vec>::from(musamples[k]));
    }
    arma::vec post_mean = mcmc_samples.mean();
    arma::mat post_cov = mcmc_samples.cov();
    arma::vec true_mean(mfeat);
    true_mean.zeros();
    true_mean(0) = cmean;
    true_mean(1) = bmean;
    arma::vec post_cent = post_mean - true_mean;
    double zsqr = arma::as_scalar(post_cent.t() * arma::inv(arma::sympd(post_cov)) * post_cent);
    
    if ((zsqr > upper_bound) || (zsqr < lower_bound)) {
        std::cout << "Coefficient test failed for mean of coefficients." << std::endl;
        std::cout << "Reduced chi-square: " << zsqr / mfeat << std::endl;
        post_mean.print("posterior mean: ");
        true_mean.print("true value: ");
        mcmc_samples.stddev().print("posterior stdev: ");
    }
    
    // now test sigsqr
    arma::vec ssqr_sample(Model.GetNoiseMean());
    ssqr_sample = arma::log(ssqr_sample);
    double post_smean = arma::mean(ssqr_sample);
    double post_stdev = arma::stddev(ssqr_sample);
    double zscore = (post_smean - logssqr_mean) / post_stdev;
    
    if (std::abs(zscore) > 3.0) {
        std::cout << "Noise Variance test failed geometric mean of noise variance." << std::endl;
        std::cout << "Z-score: " << zscore << std::endl;
        std::cout << "true value: " << logssqr_mean << std::endl;
        std::cout << "posterior mean: " << post_smean << std::endl;
        std::cout << "posterior stdev: " << post_stdev << std::endl;
        
    }

}


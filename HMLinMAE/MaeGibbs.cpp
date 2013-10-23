//
//  MaeGibbs.cpp
//  HMVAR
//
//  Created by Brandon Kelly on 9/18/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#include <samplers.hpp>
#include <steps.hpp>
#include <proposals.hpp>
#include "MaeGibbs.hpp"
#include "linmae_parameters.hpp"

using namespace HMLinMAE;

void MaeGibbs::SetupModel(int nu, vec2d y, vec3d X)
{
    nobjects = y.size();
    mfeat = X[0][0].size();
    tdof = nu;
    // default values for prior parameters on population variance of white noise variances
    nprior_ssqr = 0.1 * 0.1;
    nprior_dof = 4;

    nsamples = 0;
    
    MakeParameters_(y, X);
    ConnectParameters_();
}

// make the parameter objects for the VAR(p) model Gibbs sampler under the student t population model.
void MaeGibbs::MakeParameters_(vec2d& y, vec3d& X)
{
    CoefPopMean_ = CoefsMean(mfeat, true, "mu");
    CoefPopVar_ = CoefsVar(mfeat, false, "covar");
    NoisePopMean_ = NoiseMean(true, "ssqr");
    NoisePopVar_ = NoiseVar(nprior_dof, nprior_ssqr, false, "sigsqr var");
    for (int i=0; i<nobjects; i++) {
        std::string w_label("weights_");
        w_label.append(std::to_string(i));
        Weights_.push_back(tWeights(mfeat, tdof, true, w_label));
        
        std::string beta_label("beta_");
        beta_label.append(std::to_string(i));
        // convert vec1d for y input to arma::vec
        arma::vec this_y = arma::conv_to<arma::vec>::from(y[i]);
        // convert vec2d for X to arma::mat
        int ndata = y[i].size();
        arma::mat this_X(ndata, mfeat);
        for (int j=0; j<ndata; j++) {
            this_X.row(j) = arma::conv_to<arma::rowvec>::from(X[i][j]);
        }
        Beta_.push_back(Coefs(this_y, this_X, true, beta_label, 1.0));
        
        std::string sigsqr_label("sigsqr_");
        sigsqr_label.append(std::to_string(i));
        SigSqr_.push_back(Noise(true, sigsqr_label));
    }
}

// connect the parameter objects, so they can simulate from their conditional posteriors
void MaeGibbs::ConnectParameters_()
{
    CoefPopMean_.SetCovar(CoefPopVar_);
    CoefPopVar_.SetMu(CoefPopMean_);
    NoisePopMean_.SetNoiseVar(NoisePopVar_);
    NoisePopVar_.SetNoiseMean(NoisePopMean_);
    for (int i=0; i<nobjects; i++) {
        CoefPopMean_.AddCoef(Beta_[i]);
        CoefPopMean_.AddWeight(Weights_[i]);
        CoefPopVar_.AddCoefs(Beta_[i]);
        CoefPopVar_.AddWeights(Weights_[i]);
        NoisePopMean_.AddNoise(SigSqr_[i]);
        NoisePopVar_.AddNoise(SigSqr_[i]);
        Weights_[i].SetParameters(CoefPopMean_, CoefPopVar_, Beta_[i]);
        Beta_[i].SetParameters(Weights_[i], SigSqr_[i], CoefPopMean_, CoefPopVar_);
        SigSqr_[i].SetParameters(Beta_[i], NoisePopMean_, NoisePopVar_);
    }
}

// run the MCMC sampler
void MaeGibbs::RunMCMC(int nmcmc, int burnin, int nthin)
{
    nsamples = nmcmc;
    Sampler Model(nsamples, burnin, nthin);
    
    // proposal for Metropolis-within-gibbs are student's t
    StudentProposal tProp(4.0, 1.0);
    
    double target_rate = 0.4;
    double ivar = 0.01;
    
    // add each of the MCMC steps, need to keep this order
    Model.AddStep(new GibbsStep<double>(NoisePopMean_));
    Model.AddStep(new GibbsStep<double>(NoisePopVar_));
    for (int i=0; i<nobjects; i++) {
        Model.AddStep(new GibbsStep<double>(Weights_[i]));
    }
    Model.AddStep(new GibbsStep<arma::mat>(CoefPopVar_));
    Model.AddStep(new GibbsStep<arma::vec>(CoefPopMean_));
    for (int i=0; i<nobjects; i++) {
        arma::mat this_X = Beta_[i].GetXmat();
        arma::mat icovar = this_X.t() * this_X;
        icovar.diag() *= 1.5;
        try {
            icovar = arma::inv(arma::sympd(icovar));
        } catch (std::runtime_error& e) {
            std::cout << "Caught runtime error when trying to compute initial covariance matrix of coefficient proposal for day " << i << ", "
                << e.what() << std::endl;
            icovar.print();
            std::cout << "Just using the identity matrix..." << std::endl;
            icovar.eye();
        }
        Model.AddStep(new AdaptiveMetro(Beta_[i], tProp, icovar, 0.35, burnin));
        Model.AddStep(new UniAdaptiveMetro(SigSqr_[i], tProp, ivar, 0.4, burnin));
    }
    
    Model.Run();
}

// return a MCMC sample of the population mean of the coefficients in a vector format
vec2d MaeGibbs::GetCoefsMean()
{
    vec2d the_mus;
    for (int i=0; i<nsamples; i++) {
        vec1d this_mu = CoefPopMean_.getSample(i);
        the_mus.push_back(this_mu);
    }

    return the_mus;
}

vec1d MaeGibbs::GetNoiseMean()
{
    vec1d ssqr = NoisePopMean_.GetSamples();
    return ssqr;
}

vec2d MaeGibbs::GetCoefs(int object)
{
    vec2d the_betas;
    for (int i=0; i<nsamples; i++) {
        vec1d this_beta = Beta_[object].getSample(i);
        the_betas.push_back(this_beta);
    }
    
    return the_betas;
}

vec1d MaeGibbs::GetSigSqr(int object)
{
    return SigSqr_[object].GetSamples();
}

vec1d MaeGibbs::GetWeights(int object)
{
    vec1d this_w = Weights_[object].GetSamples();
    return this_w;
}










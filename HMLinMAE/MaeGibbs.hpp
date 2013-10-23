//
//  VarGibbs.h
//  HMVAR
//
//  Created by Brandon Kelly on 9/18/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __HMLinMAE__VarGibbs__
#define __HMLinMAE__VarGibbs__

#include <iostream>
#include "linmae_parameters.hpp"
#include <samplers.hpp>

using namespace HMLinMAE;

typedef std::vector<double> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;

namespace HMLinMAE
{
    // class for setting up gibbs sampler for VAR(p) model
    class MaeGibbs {
    public:
        int mfeat; // # of features
        int nobjects; // # of individuals in population
        int nsamples; // # of MCMC samples generated
        int nprior_dof; // degrees of freedom for scaled inverse-chi-square prior on population variance
        double nprior_ssqr; // same as above, but the scale parameters
        int tdof; // degrees of freedom for student's t model
        
        MaeGibbs() {};
        MaeGibbs(int nu, vec2d y, vec3d X) { SetupModel(nu, y, X); }
        
        void SetupModel(int nu, vec2d y, vec3d X);
        
        void RunMCMC(int nmcmc, int nburnin, int nthin=1);
        
        // grab MCMC samples
        vec2d GetCoefsMean();
        vec1d GetNoiseMean();
        vec2d GetCoefs(int day);
        vec1d GetSigSqr(int day);
        vec1d GetWeights(int day);
        
        // grab the parameter objects
        CoefsMean* GrabCoefPopMean() { return &CoefPopMean_; }
        CoefsVar* GrabCoefPopVar() { return &CoefPopVar_; }
        NoiseMean* GrabNoisePopMean() { return &NoisePopMean_; }
        NoiseVar* GrabNoisePopVar() { return &NoisePopVar_; }
        tWeights* GrabWeights(int day) { return &Weights_[day]; }
        Coefs* GrabCoef(int day) { return &Beta_[day]; }
        Noise* GrabSigSqr(int day) { return &SigSqr_[day]; }
        
    private:
        CoefsMean CoefPopMean_;
        CoefsVar CoefPopVar_;
        NoiseMean NoisePopMean_;
        NoiseVar NoisePopVar_;
        std::vector<tWeights> Weights_;
        std::vector<Coefs> Beta_;
        std::vector<Noise> SigSqr_;

        void MakeParameters_(vec2d& y, vec3d& X);
        void ConnectParameters_();
    };
} // namespace HMLinMAE


#endif /* defined(__HMLinMAE__VarGibbs__) */

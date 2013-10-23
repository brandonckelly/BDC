//
//  var_parameters.hpp
//  HMVAR
//
//  Created by Brandon Kelly on 9/13/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef __HMLinMAE__parameters__
#define __HMLinMAE__parameters__

#include <iostream>
#include <armadillo>
#include <memory>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include <parameters.hpp>

namespace HMLinMAE {
    
    // forward declaration of classes
    class Coefs;
    class Noise;
    class CoefsVar;
    class NoiseMean;
    class NoiseVar;
    class CoefsMean;
    
    class tWeights : public Parameter<double> {
    public:
        int mfeat;
        int dof;
        
        tWeights() {}
        tWeights(int m, int d, bool track, std::string label, double temperature=1.0) :
        Parameter<double>(track, label, temperature), mfeat(m), dof(d) {}
        
        double StartingValue();
        double RandomPosterior();
        void Save(double new_value) {value_ = new_value;}
        
        void SetParameters(CoefsMean& Mu, CoefsVar& Sigma, Coefs& Beta) {
            pCoefMean_ = &Mu;
            pCoefVar_ = &Sigma;
            pCoef_ = &Beta;
        }
        
    private:
        // pointers to other parameter objects
        CoefsMean* pCoefMean_;
        CoefsVar* pCoefVar_;
        Coefs* pCoef_;
    };
    
    /*
     *  NOW MAKE CLASSES FOR POPULATION LEVEL PARAMETERS
     */
    
    
    class CoefsMean : public Parameter<arma::vec> {
    public:
        int mfeat;
        
        CoefsMean() {}
        CoefsMean(int m, bool track, std::string label, double temperature=1.0) :
        Parameter<arma::vec>(track, label, temperature), mfeat(m) {
            value_.set_size(mfeat);
        }
        
        arma::vec StartingValue();
        arma::vec RandomPosterior();
        void Save(arma::vec new_value) {value_ = new_value;}
        
        // setters and getters
        void SetSize(int mfeat) {value_.set_size(mfeat);}
        
        void SetCovar(CoefsVar& Sigma) {
            pCoefsVar_ = &Sigma;
        }

        void AddCoef(Coefs& beta) {
            pCoefs_.push_back(&beta);
        }
        void AddWeight(tWeights& W) {
            pWeights_.push_back(&W);
        }
        
        std::vector<double> getSample(int iter) {
            std::vector<double> this_sample = arma::conv_to<std::vector<double> >::from(samples_[iter]);
            return this_sample;
        }
        
    private:
        // pointers to other parameters objects
        std::vector<tWeights*> pWeights_;
        CoefsVar* pCoefsVar_;
        std::vector<Coefs*> pCoefs_; // array of pointers to the VAR(p) coefficients for each individual
    };
    
    class CoefsVar : public Parameter<arma::mat> {
    public:
        int mfeat;
        int prior_dof;
        arma::mat prior_scale;
        
        CoefsVar() {}
        CoefsVar(int m, bool track, std::string label, double temperature=1.0) :
        Parameter<arma::mat>(track, label, temperature), mfeat(m) {
            prior_dof = mfeat + 2;
            prior_scale = arma::eye(mfeat, mfeat);
        }
        
        arma::mat StartingValue();
        arma::mat RandomPosterior();
        
        void SetMu(CoefsMean& Mu) {
            pCoefsMean_ = &Mu;
        }
        
        void AddWeights(tWeights& tW) {
            pWeights_.push_back(&tW);
        }
        
        void AddCoefs(Coefs& Beta) {
            pCoefs_.push_back(&Beta);
        }
        
    private:
        // pointers to other parameter objects
        CoefsMean* pCoefsMean_;
        std::vector<Coefs*> pCoefs_;
        std::vector<tWeights*> pWeights_;
    };
    
    class NoiseMean : public Parameter<double> {
    public:

        NoiseMean() {}
        NoiseMean(bool track, std::string label, double temperature=1.0) :
        Parameter<double>(track, label, temperature) {}
        
        double StartingValue();
        double LogDensityVec(double mu_ssqr);
        double RandomPosterior();
        
        void Save(double new_value) {
            value_ = new_value;
        }

        // setters and getters
        void SetNoiseVar(NoiseVar& NoiseSsqr) {
            pNoiseVar_ = &NoiseSsqr;
        }
        void AddNoise(Noise& SigSqr) {
            pNoise_.push_back(&SigSqr);
        }
        
    private:
        // pointers to other parameter objects
        CoefsMean* pCoefsMean_;
        NoiseVar* pNoiseVar_;
        std::vector<Noise*> pNoise_;
    };


    class NoiseVar : public Parameter<double> {
    public:
        NoiseVar() {}
        NoiseVar(int dof, double ssqr, bool track, std::string label, double temperature=1.0) :
        Parameter<double>(track, label, temperature), prior_dof_(dof), prior_ssqr_(ssqr) {}
        
        double StartingValue();
        double RandomPosterior();
        void Save(double new_value) {value_ = new_value;}
        
        // setters and getters
        void SetNoiseMean(NoiseMean& NoiseMu) {pNoiseMean_ = &NoiseMu;}
        void AddNoise(Noise& SigSqr) {
            pNoise_.push_back(&SigSqr);
        }
        
    private:
        int prior_dof_;
        double prior_ssqr_;
        
        NoiseMean* pNoiseMean_;
        std::vector<Noise*> pNoise_;
    };
    
    /*
     *  CLASSES FOR INDIVIDUAL LEVEL PARAMETERS
     */
    
    class Coefs : public Parameter<arma::vec> {
    public:
        int mfeat;
        int ndata;
        
        Coefs(arma::vec& y, arma::mat& X, bool track, std::string label, double temperature=1.0) :
        Parameter<arma::vec>(track, label, temperature), y_(y), X_(X)
        {
            mfeat = X.n_cols;
            ndata = X.n_rows;
        }
        
        arma::vec StartingValue();
        double LogDensity(arma::vec beta);
        void Save(arma::vec new_value) {value_ = new_value;}
        
        // setters and getters
        arma::mat& GetY() {return y_;}
        arma::mat& GetXmat() {return X_;}
        
        void SetParameters(tWeights& tW, Noise& Ssqr, CoefsMean& Mu, CoefsVar& Sigma) {
            pNoise_ = &Ssqr;
            pWeights_ = &tW;
            pCoefMean_ = &Mu;
            pCoefVar_ = &Sigma;
        }
        
        std::vector<double> getSample(int iter) {
            std::vector<double> this_sample = arma::conv_to<std::vector<double> >::from(samples_[iter]);
            return this_sample;
        }
        
    private:
        arma::vec y_;
        arma::mat X_;
        // pointers to other parameter objects
        tWeights* pWeights_;
        Noise* pNoise_;
        CoefsMean* pCoefMean_;
        CoefsVar* pCoefVar_;
    };
    
    class Noise : public Parameter<double> {
    public:
        
        Noise() {}
        Noise(bool track, std::string label, double temperature=1.0) :
        Parameter<double>(track, label, temperature) {}
        
        double StartingValue();
        double LogDensity(double sigsqr);
        
        // setters and getters
        void SetParameters(Coefs& ThisCoef, NoiseMean& NoiseMu, NoiseVar& NoiseSsqr)
        {
            pCoef_ = &ThisCoef;
            pNoiseMean_ = &NoiseMu;
            pNoiseVar_ = &NoiseSsqr;
        }
        
    private:
        // pointers to other parameter objects
        Coefs* pCoef_;
        NoiseMean* pNoiseMean_;
        NoiseVar* pNoiseVar_;
        CoefsMean* pCoefMu_;
    };
    
} // namespace HMLinMAE

#endif /* defined(__HMLinMAE__parameters__) */

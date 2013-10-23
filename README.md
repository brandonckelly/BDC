BDC
===

Code used to generate my submission for the Big Data Combine Kaggle Competition

===

Description of Model:

I used a Hierarchical model and carried out Bayesian inference using Markov Chain Monte Carlo. I modeled the value of a security two hours in the future as having a Laplacian distribution with mean given by a linear combination of the values at the current time, the value 5 minutes earlier, and the value 10 minutes earlier. The free parameters are the constant, the regression coefficients, and the width of the Laplacian distribution. These were assumed to be different for each security. I chose this model because the maximum a posteriori estimate under this model corresponds to the estimate that minimizes the mean absolute error.

In addition, I also modeled the joint distribution of the regression parameters (intercepts and slopes) as having a student's t-distribution with unknown mean and covariance matrix. In practice, I did not notice a large difference between the t model and a normal model, so I used the t model with a large value for the degrees of freedom. The distribution of the scale (variance) parameters for each security was modeled using a log-normal distribution with unknown mean and variance. I used broad priors for the group level parameters. I then used MCMC to obtain random draws from the joint posterior of the regression parameters for each security, the Laplacian scale parameters for each security, the mean and covariance matrix of the regression parameters over all securities, and the log-normal parameters for the distribution of the scale parameters. Then, for each of the MCMC samples I used the parameters to predict the value of the price two hours in the future. This gave me a set of random samples of the predicted price from its posterior probability distribution. I then computed my predictions ('best-fit' values) for the price of each security two hours in the future from the median of the predictions derived from the MCMC samples.

When fitting the data I trained my model using both the values at 4pm for the training set, and the values at 2pm for the test set. This way I also use the data from the test set, and my model is not slanted strongly towards the training set. So, in other words, I tried to predict the values at 4pm for the training set using the values at 2pm, 1:55pm, and 1:50pm, and I tried to predict the values at 2pm for the test set using the values at noon, 11:55am, and 11:50am.

Finally, I trained a gradient boosted regression tree on the residuals from the MCMC sampler predictions using the Box-Cox transformed sentiment data (the 'features'). However, the number of estimators used was very small, and this only resulting in a very small improvement; it is unclear how much this helped.

===

Installation notes:

The code is a mixture of C++ and Python. The MCMC sampler is written in C++ as a Python extension for increased speed. Everything else is written in Python, including the calls to the MCMC sampler.

In order to install the MCMC sampler, you will need the Boost and Armadillo C++ libraries installed. In particular, the Python extension is built using Boost.Python. You will also have to build my yamcmcpp library for the MCMC sampler classes, which is also available on my Github account. Once you have built these libraries, then in theory it should be sufficient to simply do

python setup.py install

However, the compilers flags are particular to my Mac OS X install, so they will probably be different for other OS. This can be edited in the setup.py file.

===

To create my submission file, perform the following steps (note that file locations, etc., would need to be changed):

1) First build the Pandas data frame

python get_data.py

2) Now run the MCMC samplers, boost the residuals, and build the submission file

python boost_hmlin_residuals.py

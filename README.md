# Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo (HMC) sampling method.

### References

The original paper, that introduced this method is described in:

   1. Simon Duane, Anthony D. Kennedy, Brian J. Pendleton and Duncan Roweth (1987).
   "Hybrid Monte Carlo". Physics Letters B. 195 (2): 216–222.

Several implementation details are given in:

   2. Radford M. Neal (1996). "Monte Carlo Implementation".
   Bayesian Learning for Neural Networks. Springer. pp. 55–98.

The generalized sampling approach is described in:

   3. Francis J. Alexander, Gregory L. Eyink and Juan M. Restrepo (2005).
   "Accelerated Monte Carlo for Optimal Estimation of Time Series",
   Journal of Statistical Physics, vol.119, pp: 1331-1345.

### Requirements

To ensure smooth execution please install the required modules with:

      $ pip install -r requirements.txt

### Examples

Some example on how to use this method can be found below:

1. [Rosenbrock](examples/example_rosenbrock.ipynb)
2. [Multivariate Normal](examples/example_multivariate_normal.ipynb)
3. [Ornstein-Uhlenbeck process](examples/example_ornstein_uhlenbeck.ipynb)

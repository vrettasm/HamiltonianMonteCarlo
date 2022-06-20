import numpy as np
from numba import njit
from copy import deepcopy
from time import perf_counter
from scipy.linalg import circulant
from scipy.optimize import check_grad
from scipy._lib._util import check_random_state


# Public functions:
__all__ = ['HMC']


class HMC(object):
    """
        1. Simon Duane, Anthony D. Kennedy, Brian J. Pendleton and Duncan Roweth (1987).
        "Hybrid Monte Carlo". Physics Letters B. 195 (2): 216–222.

        2. Radford M. Neal (1996). "Monte Carlo Implementation".
        Bayesian Learning for Neural Networks. Springer. pp. 55–98.

        3. Francis J. Alexander, Gregory L. Eyink and Juan M. Restrepo (2005).
        "Accelerated Monte Carlo for Optimal Estimation of Time Series",
        Journal of Statistical Physics, vol.119, pp: 1331-1345.
    """

    # Object variables.
    __slots__ = ("func", "grad", "_options", "_stats")

    def __init__(self, func, grad, n_samples=10_000, n_omitted=5_000,
                 kappa=100, d_tau=0.01, n_parallel=None, rng_seed=None,
                 generalized=False, grad_check=False, verbose=False):
        """
        Default constructor of an HMC sampler object.

        :param func: (callable) Function that evaluates the new proposals.
        This is the negative log probability we want to sample from.

        :param grad: (callable) Gradient of the 'func' with respect to the
        new state.

        :param n_samples: (int) Number of samples.

        :param n_omitted: (int) Number of omitted samples (burn-in) period.

        :param kappa: (int) Maximum number of leapfrog steps.

        :param d_tau: (float) Time discretization in the leapfrog integration scheme.

        :param n_parallel: (int) Number of parallel CPUs.

        :param rng_seed: (int) Random number seed.

        :param generalized: (bool) Application of the generalized HMC algorithm.

        :param grad_check: (bool) Performs gradient check before and after the sampling.

        :param verbose: (bool) Display (periodical) additional information on the rum.
        """
        # The function and its gradient must be callable objects.
        if callable(func):

            # Assign the function to the object.
            self.func = func
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Function {func} must be callable.")
        # _end_if_

        if callable(grad):

            # Assign the function to the object.
            self.grad = grad
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Function {grad} must be callable.")
        # _end_if_

        # The other options will be stored in a dictionary.
        self._options = dict()

        self._options["n_samples"] = int(n_samples)
        self._options["n_omitted"] = int(n_omitted)

        self._options["kappa"] = int(kappa)
        self._options["d_tau"] = float(d_tau)

        # Make sure the flags are boolean.
        if isinstance(grad_check, bool):

            # Assign the boolean flag.
            self._options["grad_check"] = grad_check
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Grad-Check flag must be True/[False].")
        # _end_if_

        if isinstance(generalized, bool):

            # Assign the boolean flag.
            self._options["generalized"] = generalized
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Generalized flag must be True/[False].")
        # _end_if_

        if isinstance(verbose, bool):

            # Assign the boolean flag.
            self._options["verbose"] = verbose
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Verbose flag must be True/[False].")
        # _end_if_

        # If we want parallel execution.
        if n_parallel:

            # Make sure we have at least one CPU.
            self._options["n_parallel"] = max(int(n_parallel), 1)
        # _end_if_

        # Check the seed, before assignment.
        self._options["rng_seed"] = check_random_state(rng_seed)

        # Initialize stats dictionary.
        self._stats = dict()

    # _end_def_

    @property
    def n_samples(self):
        """
        Number of samples accessor (getter).

        :return: the number of samples that we want.
        """
        return self._options["n_samples"]
    # _end_def_

    @n_samples.setter
    def n_samples(self, new_value):
        """
        Property accessor (setter).

        :param new_value: (integer) value of the samples we want to sample.
        """
        # Check for correct type.
        if isinstance(new_value, int):

            # Make sure we have only positive values.
            if new_value > 1:

                # Assign the new value.
                self._options["n_samples"] = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Number of samples should be positive: {new_value}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of samples should be integer: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def n_omitted(self):
        """
        Number of omitted samples. These are the burn-in period of the HMC sampler.

        :return: the number of omitted samples.
        """
        return self._options["n_omitted"]

    # _end_def_

    @n_omitted.setter
    def n_omitted(self, new_value):
        """
        Property accessor (setter).

        :param new_value: (integer) value of the omitted samples.
        """
        # Check for correct type.
        if isinstance(new_value, int):

            # Make sure we have only positive values.
            if new_value > 1:

                # Assign the new value.
                self._options["n_omitted"] = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Number of omitted samples should be positive: {new_value}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of omitted samples should be integer: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def kappa(self):
        """
        Number of leapfrog steps accessor (getter).

        :return: the maximum number of leapfrog steps.
        """
        return self._options["kappa"]

    # _end_def_

    @kappa.setter
    def kappa(self, new_value):
        """
        Property accessor (setter).

        :param new_value: (integer) value of the leapfrog steps.
        """
        # Check for correct type.
        if isinstance(new_value, int):

            # Make sure we have only positive values.
            if new_value > 10:

                # Assign the new value.
                self._options["kappa"] = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Number of leapfrog steps should be > 10: {new_value}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Number of leapfrog steps should be integer: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def delta_tau(self):
        """
        Leapfrog integration step.

        :return: the 'dt' in the leapfrog integration.
        """
        return self._options["d_tau"]

    # _end_def_

    @delta_tau.setter
    def delta_tau(self, new_value):
        """
        Property accessor (setter).

        :param new_value: (float) value of the leapfrog dt.
        """
        # Check for correct type.
        if isinstance(new_value, float):

            # Make sure we have only positive values.
            if new_value > 0.0:

                # Assign the new value.
                self._options["d_tau"] = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Value of leapfrog dt should be positive: {new_value}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Value of leapfrog dt should be float: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def grad_check(self):
        """
        Property accessor (getter).

        :return: True if we want to perform gradient-check, otherwise False.
        """
        return self._options["grad_check"]
    # _end_def_

    @property
    def generalized(self):
        """
        Property accessor (getter).

        :return: True if we want to apply the generalized HMC, otherwise False.
        """
        return self._options["generalized"]
    # _end_def_

    @property
    def verbose(self):
        """
        Property accessor (getter).

        :return: True if we want to display information on the run, otherwise False.
        """
        return self._options["verbose"]
    # _end_def_

    @staticmethod
    @njit(fastmath=True)
    def fast_dot(v):
        """
        Computes a 'fast' numba compiled version
        of the dot product --> v.T.dot(v).

        :param v: numpy array (dim,)

        :return: the (scalar) dot product (v, v)
        """
        return v.T.dot(v)
    # _end_def_

    def run(self, x0, *args):
        """

        :param x0:

        :param args:

        :return:
        """

        # Make sure the initial starting point of
        # the HMC search is a (copy) numpy array.
        x = np.asarray(x0, dtype=float).flatten()

        # Check numerically the gradients.
        if self._options['grad_check']:
            print("Checking gradients <BEFORE> sampling ... ")
            diff_error = check_grad(self.func, self.grad, deepcopy(x),
                                    *args, direction='all')
            print(f" Error: {diff_error}", end='\n')
        # _end_if_

        # Every time we run a new sampling process we reset the statistics.
        self._stats = {"logE": [], "Samples": [], "Accepted": [],
                       "Elapsed_Time": -1}

        # Dimensionality of the input vector.
        x_dim = x.size

        # Check for pre-conditioning.
        if self._options["generalized"]:

            # Constant.
            _alpha = float(1.0/x_dim)

            # Construct a circulant matrix
            Q = circulant(np.exp(-_alpha*np.arange(0, x_dim)))

            # Display start message.
            print(" >>> Generalized HMC sampling started ... ")
        else:

            # Identity matrix.
            Q = np.eye(x_dim)

            # Display start message.
            print(" >>> HMC sampling started ... ")
        # _end_if_

        # First time.
        t0 = perf_counter()

        # Initial function evaluation.
        fx0 = self.func(x, *args)

        # Initial gradient evaluation.
        gx0 = self.grad(x, *args)

        # Set initial values.
        E, g = fx0, gx0

        # Accepted samples counter / acceptance ratio.
        acc_counter, acc_ratio = 0, 0.0

        # Local copy of the random number generator.
        rng = np.random.default_rng(self._options["rng_seed"].get_state()[1])

        # Local copies of 'delta tau' and 'kappa' constants.
        d_tau = self._options["d_tau"]
        kappa = self._options["kappa"]

        # Local copies of random functions.
        _uniform = rng.uniform
        _standard_normal = rng.standard_normal

        # Local copy of the numba dot product.
        _dot = self.fast_dot

        # Begin Hamiltonian Monte Carlo iterations.
        for i in range(-self._options["n_omitted"], self._options["n_samples"]):

            # Initial momentum: p ~ N(0, 1).
            p = _standard_normal(x_dim)

            # Evaluate Hamiltonian.
            H = E + (0.5 * _dot(p))

            # Set the current state and gradient.
            x_new, g_new = x.copy(), g.copy()

            # Change direction at random (~50% probability).
            mu = +1.0 if 0.5 > _uniform(0.0, 1.0) else -1.0

            # Perturb the length in the leapfrog steps by 0.1 (=10%).
            epsilon = mu * d_tau * (1.0 + 0.1 * _standard_normal(1))

            # Choose leapfrog steps uniformly between [10 ... kappa].
            KAPPA = rng.integers(10, kappa, endpoint=True, dtype=int)

            # First half-step of leapfrog.
            p -= 0.5 * epsilon * Q.T.dot(g_new)
            x_new += epsilon * p

            # Full (KAPPA-1) leapfrog steps.
            for _ in range(KAPPA - 1):
                p -= epsilon * Q.T.dot(self.grad(x_new, *args))
                x_new += epsilon * Q.dot(p)
            # _end_for_

            # Gradient at 'x_new'.
            g_new = self.grad(x_new, *args)

            # Final half-step of leapfrog.
            p -= 0.5 * epsilon * Q.T.dot(g_new)

            # Compute the energy at the new point.
            E_new = self.func(x_new, *args)

            # Compute the new Hamiltonian.
            H_new = E_new + (0.5 * _dot(p))

            # Compute the difference between
            # the two Hamiltonian values.
            deltaH = H_new - H

            # Check for acc_ratio.
            if min(1, np.exp(-deltaH)) > _uniform(0.0, 1.0):

                # Update the counters.
                if i >= 0:
                    acc_counter += 1
                    acc_ratio = float(acc_counter) / (i + 1)
                # _end_if_

                # Update to the new states.
                x, g, E = x_new, g_new, E_new

            # _end_if_

            # Check if something went wrong.
            if not np.isfinite(E_new):
                raise RuntimeError(f"{self.__class__.__name__}: "
                                   f"Unexpected error happened at iteration {i}.")
            # _end_if_

            # Save current energy value.
            self._stats["logE"].append(E_new.item())

            # Update statistics:
            # These are not stored during the burn-in period (i < 0).
            if i >= 0:
                self._stats["Samples"].append(x)
                self._stats['Accepted'].append(acc_ratio)
            # _end_if_

            # Check for verbosity.
            if self._options["verbose"]:

                # Display every 100 iterations.
                if (i >= 0) and (np.mod(i, 100) == 0):
                    print(' {0}:\tE={1:.3f}\tA/R={2:.3f}'.format(i, E, acc_ratio))
                # _end_if_

            # _end_if_

        # _end_for_

        # Final time.
        tf = perf_counter()

        # Execution time (in seconds).
        time_elapsed = tf-t0

        # Display finish message.
        print(f" >>> HMC sampling finished in {time_elapsed} seconds.")

        # Store the elapsed time.
        self._stats["Elapsed_Time"] = time_elapsed

        # Check numerically the gradients.
        if self._options['grad_check']:
            print("Checking gradients <AFTER> sampling ... ")
            diff_error = check_grad(self.func, self.grad,
                                    deepcopy(x), *args,
                                    direction='all')
            print(f" Error: {diff_error}")
        # _end_if_

        # Return the dictionary with the collected stats.
        return self._stats
    # _end_def_

    def __call__(self, *args, **kwargs):
        """
        This is only a wrapper of the "run" method.
        """
        return self.run(*args, **kwargs)
    # _end_def_

    def __str__(self):
        pass
    # _end_def_

    def __repr__(self):
        pass
    # _end_def_

# _end_class_

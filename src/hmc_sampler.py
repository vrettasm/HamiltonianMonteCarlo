import numpy as np
from tqdm import tqdm
from copy import deepcopy
from time import perf_counter
from scipy.linalg import circulant
from scipy.optimize import check_grad
from multiprocessing import cpu_count
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

        :param kappa: (int) Number of leapfrog steps.

        :param d_tau: (float) Time discretization in the leapfrog integration scheme.

        :param n_parallel: (int) Number of parallel chains (one per CPU).

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

            # Get the available number of CPUs.
            num_cores = cpu_count()

            # Make sure we are in the range [1, num_cores].
            self._options["n_parallel"] = min(max(int(n_parallel), 1), num_cores)

            # Show a warning.
            print(f"{self.__class__.__name__}: WARNING: This option is not implemented yet."
                  f" The program will sample only one chain.")
        # _end_if_

        # Check the seed before assignment.
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

        :return: the number of leapfrog steps.
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
            if new_value > 0:

                # Assign the new value.
                self._options["kappa"] = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Number of leapfrog steps should be positive: {new_value}.")
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

    # Main (sampling) operation.
    def run(self, x0, *args):
        """
        Implements the HMS sampling routine.

        :param x0: Initial point to sample: (x_dim,).

        :param args: Additional func/grad parameters.

        :return: A dictionary with the collected stats.
        """

        # Make sure the initial starting point of
        # the HMC search is a (copy) numpy array.
        x = np.asarray(x0, dtype=float).flatten()

        # Local copies of the func/grad functions.
        _func = self.func
        _grad = self.grad

        # Check numerically the gradients.
        if self._options['grad_check']:
            # Display info for the user.
            print("Checking gradients <BEFORE> sampling ... ")

            # Get the grad-check error.
            diff_error = check_grad(_func, _grad, deepcopy(x), *args)

            # Display the error.
            print(f"Error <BEFORE> = {diff_error}", end='\n')
        # _end_if_

        # Every time we run a new sampling process we reset the statistics.
        self._stats = {"Energies": [], "Samples": [], "Accepted": [],
                       "Elapsed_Time": -1}

        # Dimensionality of the input vector.
        x_dim = x.size

        # Check for pre-conditioning.
        if self._options["generalized"]:

            # Constant (should be optimized).
            _alpha = float(1.0/x_dim)

            # Construct a circulant matrix
            Q = circulant(np.exp(-_alpha*np.arange(0, x_dim)))

            # Display start message.
            print(" >>> Generalized HMC ")
        else:

            # Identity matrix.
            Q = np.eye(x_dim)

            # Display start message.
            print(" >>> HMC started ")
        # _end_if_

        # Local copy of the random number generator.
        rng = np.random.default_rng(self._options["rng_seed"].get_state()[1])

        # Local copies of 'delta tau' and 'kappa' constants.
        d_tau = self._options["d_tau"]
        kappa = self._options["kappa"]

        # Local copies of random functions.
        _uniform = rng.uniform
        _standard_normal = rng.standard_normal

        # First time.
        t0 = perf_counter()

        # Initial function evaluation.
        E = _func(x, *args)

        # Initial gradient evaluation.
        g = _grad(x, *args)

        # Accepted samples counter / acceptance ratio.
        acc_counter, acc_ratio = 0, 0.0

        # Create the progress bar.
        _tqdm = tqdm(range(-self._options["n_omitted"], self._options["n_samples"]),
                     desc=" Sampling in progress ...")

        # Begin sampling iterations.
        for i in _tqdm:

            # Copy the current state and gradient.
            x_new, g_new = x.copy(), g.copy()

            # Initial momentum: p ~ N(0, 1).
            p = _standard_normal(x_dim)

            # Evaluate Hamiltonian.
            H = E + (0.5 * p.T.dot(p))

            # Change direction at random (~50% probability).
            mu = +1.0 if 0.5 > _uniform(0.0, 1.0) else -1.0

            # Perturb the length in the leapfrog steps by 0.1 (=10%).
            epsilon = mu * d_tau * (1.0 + 0.1 * _standard_normal(1))

            # First half-step of leapfrog.
            p -= 0.5 * epsilon * Q.T.dot(g_new)
            x_new += epsilon * p

            # Full (kappa-1) leapfrog steps.
            for _ in range(kappa - 1):
                p -= epsilon * Q.T.dot(_grad(x_new, *args))
                x_new += epsilon * Q.dot(p)
            # _end_for_

            # Gradient at 'x_new'.
            g_new = _grad(x_new, *args)

            # Final half-step of leapfrog.
            p -= 0.5 * epsilon * Q.T.dot(g_new)

            # Compute the energy at the new point.
            E_new = _func(x_new, *args)

            # Compute the new Hamiltonian.
            H_new = E_new + (0.5 * p.T.dot(p))

            # Compute the difference between the two Hamiltonian values.
            deltaH = H_new - H

            # Metropolis-Hastings acceptance criterion.
            # A(x, x') = min(1.0, np.exp(-deltaH)), is
            # also known as the acceptance probability.
            if _uniform(0.0, 1.0) <= min(1.0, np.exp(-deltaH)):

                # Update the counters.
                if i >= 0:
                    acc_counter += 1
                    acc_ratio = float(acc_counter) / (i + 1)
                # _end_if_

                # Update the accepted states.
                x, g, E = x_new, g_new, E_new

            # _end_if_

            # Check if something went wrong.
            if not np.isfinite(E_new):
                raise RuntimeError(f"{self.__class__.__name__}: "
                                   f"Unexpected error happened at iteration {i}.")
            # _end_if_

            # Save the energy value.
            self._stats["Energies"].append(E.item())

            # These are not stored during the burn-in period (i < 0).
            if i >= 0:
                self._stats["Samples"].append(x)
                self._stats['Accepted'].append(acc_ratio)
            # _end_if_

            # Check for verbosity.
            if self._options["verbose"]:

                # Display every 'n' iterations.
                if (i >= 0) and (np.mod(i, 500) == 0):

                    # Update the description in the progress bar.
                    _tqdm.set_description(f" Iter={i} - E={E:.3f} -"
                                          f" Acceptance={acc_ratio:.3f}")
                # _end_if_

            # _end_if_

        # _end_for_

        # Final time.
        tf = perf_counter()

        # Execution time (in seconds).
        time_elapsed = tf-t0

        # Display finish message.
        print(f" >>> HMC finished in {time_elapsed:.3f} seconds.")

        # Store the elapsed time.
        self._stats["Elapsed_Time"] = time_elapsed

        # Check numerically the gradients.
        if self._options['grad_check']:

            # Display info for the user.
            print("Checking gradients <AFTER> sampling ... ")

            # Get the grad-check error.
            diff_error = check_grad(self.func, self.grad, deepcopy(x), *args)

            # Display the information.
            print(f"Error <AFTER> = {diff_error}", end='\n')
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
        """
        Override to print a readable string presentation of the object.
        This will include its id along with its field values parameters.

        :return: a string representation of an HMC object.
        """

        # Initialize options string.
        hmc_options = ""

        # Get all the key-value pairs.
        for _key, _value in self._options.items():
            hmc_options += f"\t{_key}: {_value}\n"
        # _end_for_

        # Return the f-string.
        return f" HMC Id({id(self)}): \n" \
               f" Func(x)={self.func} \n" \
               f" Grad(x)={self.grad} \n" \
               f" Options:\n {hmc_options}"
    # _end_def_

# _end_class_

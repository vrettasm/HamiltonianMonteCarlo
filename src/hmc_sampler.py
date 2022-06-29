import numpy as np
from copy import deepcopy
from time import perf_counter
from scipy.linalg import circulant
from joblib import Parallel, delayed
from scipy.optimize import check_grad
from multiprocessing import cpu_count
from scipy._lib._util import check_random_state


# Public functions:
__all__ = ['HMC']


class HMC(object):
    """
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

        else:
            # Default is '1' Chain/CPU.
            self._options["n_parallel"] = 1
        # _end_if_

        # In case of multiple chains, disable verbose.
        if self._options["n_parallel"] > 1:

            # TO DO: Find a better solution.
            self._options["verbose"] = False
        # _end_if_

        # Check the seed before assignment.
        self._options["rng_seed"] = check_random_state(rng_seed)

        # Placeholder for stats dictionary.
        self._stats = {f"Chain-{i}": None for i in range(self._options["n_parallel"])}

    # _end_def_

    @property
    def n_chains(self):
        """
        Number of parallel chains accessor (getter).

        :return: the number of chains.
        """
        return self._options["n_parallel"]
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
    def _sample_single_chain(self, x, chain, *args):
        """
        Implements the HMS sampling routine for a single
        chain.

        :param x: Initial point to sample: (x_dim,).

        :param chain: Index/id of the (parallel) chain.

        :param args: Additional func/grad parameters.

        :return: A dictionary with the collected stats.
        """

        # Local copies of the func/grad functions.
        _func = self.func
        _grad = self.grad

        # Every time we run a new sampling process we reset all the
        # statistics of the local chain.
        chain_stats = {"Energies": [], "Accepted": [], "Samples": [],
                       "Elapsed_Time": -1}

        # Dimensionality of the input vector.
        x_dim = x.size

        # Check for pre-conditioning.
        if self._options["generalized"]:

            # Constant (should be optimized).
            _alpha = float(1.0/x_dim)

            # Construct a circulant matrix
            Q = circulant(np.exp(-_alpha*np.arange(0, x_dim)))

        else:

            # Identity matrix.
            Q = np.eye(x_dim)

        # _end_if_

        # Make sure the chain number is int.
        chain = int(chain)

        # Local chain identifier for the number generator.
        chain_id = self._options["rng_seed"].get_state()[1] + (chain << 1)

        # Local copy of the random number generator.
        # NOTE: Since all the chains share the same rng seed we add a unique
        # chain id number to the current rng seed, otherwise all chains will
        # sample the same values.
        rng = np.random.default_rng(chain_id)

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

        # Display start message.
        if self._options["verbose"]:
            print(f" >>> Chain -> {chain} started ... ")
        # _end_if_

        # Begin sampling iterations.
        for i in range(-self._options["n_omitted"],
                       +self._options["n_samples"]):

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
                raise RuntimeError(f"{self.__class__.__name__}: Chain -> {chain}"
                                   f"Unexpected error happened at iteration {i}.")
            # _end_if_

            # Save the energy value.
            chain_stats["Energies"].append(E)

            # These are not stored during the burn-in period (i < 0).
            if i >= 0:
                chain_stats["Samples"].append(x)
                chain_stats['Accepted'].append(acc_ratio)
            # _end_if_

            # Check for verbosity.
            if self._options["verbose"]:

                # Display every 'n' iterations.
                if (i >= 0) and (np.mod(i, 500) == 0):

                    # Update the description in the screen.
                    print(f" Chain -> {chain}: Iter={i} - E={E:.3f} -"
                          f" Acceptance={acc_ratio:.3f}")

                # _end_if_

            # _end_if_

        # _end_for_

        # Final time.
        tf = perf_counter()

        # Execution time (in seconds).
        time_elapsed = tf-t0

        # Display finish message.
        if self._options["verbose"]:
            print(f" >>> Chain -> {chain} finished in {time_elapsed:.3f} seconds.")
        # _end_if_

        # Store the elapsed time.
        chain_stats["Elapsed_Time"] = time_elapsed

        # Check numerically the gradients.
        if self._options['grad_check']:

            # Display info for the user.
            if self._options["verbose"]:
                print(f"Chain: {chain}, checking gradients ... ")
            # _end_if_

            # Get the grad-check error.
            diff_error = check_grad(_func, _grad, deepcopy(x), *args)

            # Display the information.
            if self._options["verbose"]:
                print(f"Error <AFTER> = {diff_error}", end='\n')
            # _end_if_

        # _end_if_

        # Return the local dictionary.
        return chain_stats
    # _end_def_

    def run(self, x0, *args):
        """
        Implements the HMS sampling routine for all
        the parallel chains.

        :param x0: Initial point to sample: (x_dim,).

        :param args: Additional func/grad parameters.

        :return: A dictionary with the collected data
        from all the parallel chains.
        """

        # Make sure the initial starting point of
        # the HMC search is a (copy) numpy array.
        x = np.asarray(x0, dtype=float).flatten()

        # Check numerically the gradients.
        if self._options['grad_check']:

            # Display info for the user.
            print("Checking gradients <BEFORE> sampling ... ")

            # Get the grad-check error.
            diff_error = check_grad(self.func, self.grad, deepcopy(x), *args)

            # Display the error.
            print(f"Error <BEFORE> = {diff_error}", end='\n\n')
        # _end_if_

        # Localize sampling function.
        _single_chain = self._sample_single_chain

        # Get the number of parallel chains.
        n_chains = self._options["n_parallel"]

        # Stores the initial (perturbed) points.
        x_init = []

        # Get the random seed.
        rng = np.random.default_rng(self._options["rng_seed"].get_state()[1])

        # Perturb the initial 'x' with N(0,1).
        for i in range(n_chains):
            x_init.append(x + rng.standard_normal(x.size))
        # _end_for_

        # Display start message.
        print(f" HMC started with {n_chains} chain(s) ... ")

        # First time.
        t0 = perf_counter()

        # Run the chains in parallel.
        results = Parallel(n_jobs=n_chains)(delayed(_single_chain)(x=x_init[i],
                                                                   chain=i, *args) for i in range(n_chains))
        # Final time.
        tf = perf_counter()

        # Display finish message.
        print(f" HMC finished in {tf - t0:.3f} seconds.")

        # Make a copy of the results to the object.
        for i, res_i in enumerate(results, start=0):
            self._stats[f"Chain-{i}"] = results[i]
        # _end_for_

        # Return the dictionary.
        return self._stats
    # _end_if_

    # Auxiliary method.
    def acf(self, lag_n=None):
        """
        Computes the sample auto-correlation function
        values of the energy, for a given lag number.

        :param lag_n: Lag value to compute the acf.

        :return: a list (of lists) with the acf values
        for all stored chains.
        """

        # A list that will hold the ACF values
        # for all the chains in the dictionary.
        acf_list = []

        # Compute the ACF values for every stored chain.
        for i, chain in enumerate(self._stats.values()):

            # Make sure the samples are an array.
            x = np.asarray(chain["Energies"])

            # Remove singleton dimensions.
            x = np.squeeze(x)

            # Check the number of input dimensions.
            if x.ndim != 1:
                raise ValueError(f"{self.__class__.__name__}: Chain -> {i}: "
                                 f"Wrong number of dimensions -> {x.ndim}.")
            # _end_if_

            # Number of samples.
            n_obs = int(x.size)

            # Sanity check.
            if n_obs == 0:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Chain -> {i}: The sample list is empty.")
            # _end_if_

            # Check for input.
            if lag_n is None:

                # Use a default value.
                lag_n = min(int(np.ceil(10 * np.log10(n_obs))), n_obs - 1)
            # _end_if_

            # Check the bounds.
            if -n_obs < lag_n < n_obs:

                # Make sure the lag is positive.
                lag_n = np.abs(lag_n)
            else:

                # Out of bounds error.
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Value of 'lag_n' is out of bounds -> {lag_n}.")
            # _end_if_

            # Get the sample mean.
            x_mu = x.mean()

            # Sample auto-covariance at lag '0'.
            acf_0 = np.cov((x - x_mu))

            # Denominator for the loop.
            kappa = float(n_obs) * acf_0

            # Add a new list (initialize with '1').
            acf_list.append([1.0])

            # Compute the sample acf values for every lag step [1: lag+1].
            for j in range(1, lag_n + 1):
                acf_list[i].append(np.sum((x[j:] - x_mu) * (x[:-j] - x_mu)) / kappa)
            # _end_for_

        # _end_for_

        # Return as list.
        return acf_list
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

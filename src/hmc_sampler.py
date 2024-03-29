import numpy as np
from tqdm import tqdm
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

    def __init__(self, func, grad, n_samples=10_000, n_omitted=1_000,
                 kappa=100, d_tau=0.01, n_chains=None, rng_seed=None,
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

        :param n_chains: (int) Number of parallel chains (one per CPU).

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

        if n_chains:

            # Make sure the number of chains is positive int.
            self._options["n_chains"] = int(np.abs(n_chains))

        else:
            # Default is '1' Chain/CPU.
            self._options["n_chains"] = 1
        # _end_if_

        # Check the seed before assignment.
        self._options["rng_seed"] = check_random_state(rng_seed)

        # Placeholder for stats dictionary.
        self._stats = None

    # _end_def_

    @property
    def n_chains(self):
        """
        Number of parallel chains accessor (getter).

        :return: the number of chains.
        """
        return self._options["n_chains"]
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
    @staticmethod
    def _sample_single_chain(x, _func, _grad, chain, options, *args):
        """
        Implements the HMS sampling routine for a single chain. It
        is declared as static to avoid any race conditions when we
        call it in parallel mode from run().

        :param x: Initial point to sample: (x_dim,).

        :param _func: This is the function that computes the -log(p).

        :param _grad: This is the gradient: d(_func)/dx.

        :param chain: Number (int) of the (parallel) chain.

        :param options: Dictionary with additional options.

        :param args: Additional _func/_grad parameters.

        :return: A dictionary with the collected stats.
        """

        # Every time we run a new sampling process we reset all the
        # statistics of the local chain.
        chain_stats = {"Energies": [], "Accepted": [], "Samples": [],
                       "Elapsed_Time": -1}

        # Dimensionality of the input vector.
        x_dim = x.size

        # Check for pre-conditioning.
        if options["generalized"]:

            # Constant (should be optimized).
            _alpha = float(1.0/x_dim)

            # Construct a circulant matrix
            Q = circulant(np.exp(-_alpha*np.arange(0, x_dim)))

        else:

            # Here we should have the Identity Matrix.
            # But a scalar "1.0" is much faster in the
            # matrix multiplications!!
            Q = np.array(1.0)

        # _end_if_

        # Make sure the chain number is int.
        chain = int(np.abs(chain))

        # Local chain identifier for the number generator.
        chain_id = options["rng_seed"].get_state()[1] + (chain << 1)

        # Local copy of the random number generator.
        # NOTE: Since all the chains share the same rng seed we add a unique
        # chain id number to the current rng seed, otherwise all chains will
        # sample the same values.
        rng = np.random.default_rng(chain_id)

        # Local copies of 'delta tau' and 'kappa' constants.
        d_tau = options["d_tau"]
        kappa = options["kappa"]

        # Total number of samples (including the omitted).
        TOTAL = options["n_omitted"] + options["n_samples"]

        # Pre-sampled uniform values.
        uniform_step = rng.uniform(low=0.0, high=1.0, size=TOTAL)
        uniform_MHAC = rng.uniform(low=0.0, high=1.0, size=TOTAL)

        # Pre-sampled standard normal values.
        std_normal_perturb = rng.standard_normal(size=TOTAL)

        # Local copy of standard normal dist.
        _standard_normal = rng.standard_normal

        # Accepted samples counter / acceptance ratio.
        acc_counter, acc_ratio = int(0), float(0.0)

        # Create a range object.
        chain_range = range(-options["n_omitted"], options["n_samples"])

        # If we have only one chain use tqdm.
        if options["n_chains"] == 1:

            # Create a local tqdm object.
            chain_range = tqdm(chain_range, desc=f" Chain -> {chain} in progress ... ")
        else:

            # Display start message only.
            print(f" >>> Chain -> {chain} started ... ", flush=True)
        # _end_if_

        # Create a boolean flag that allows (or not) the display of extra
        # information during sampling.
        allow_display = options["verbose"] and (options["n_chains"] == 1)

        # First time.
        t0 = perf_counter()

        # Initial function evaluation.
        E = _func(x, *args)

        # Initial gradient evaluation.
        g = _grad(x, *args)

        # Start the sampling.
        for j, i in enumerate(chain_range, start=0):

            # Fresh copy of the current state
            # and its gradient.
            x_new, g_new = x.copy(), g.copy()

            # Reset momentum: p ~ N(0, 1).
            p = _standard_normal(x_dim)

            # Evaluate the Hamiltonian.
            H = E + (0.5 * p.T.dot(p))

            # Change direction at random (~50% probability).
            mu = -1.0 if uniform_step[j] < 0.5 else +1.0

            # Perturb the length of the leapfrog steps by 0.1 (~ 10%).
            epsilon = mu * d_tau * (1.0 + 0.1 * std_normal_perturb[j])

            # First "half" leapfrog step.
            p -= 0.5 * epsilon * Q.T.dot(g_new)
            x_new += epsilon * p

            # Full (kappa-1) leapfrog steps.
            for _ in range(kappa - 1):
                p -= epsilon * Q.T.dot(_grad(x_new, *args))
                x_new += epsilon * Q.dot(p)
            # _end_for_

            # Gradient at 'x_new'.
            g_new = _grad(x_new, *args)

            # Final "half" leapfrog step.
            p -= 0.5 * epsilon * Q.T.dot(g_new)

            # Compute the energy at the new point.
            E_new = _func(x_new, *args)

            # Compute the new Hamiltonian.
            H_new = E_new + (0.5 * p.T.dot(p))

            # Compute the difference between the two Hamiltonian values.
            deltaH = H_new - H

            # Metropolis-Hastings acceptance criterion.
            # A(x', x) = min(1.0, np.exp(-deltaH)), is
            # also known as the acceptance probability.
            if uniform_MHAC[j] < min(1.0, np.exp(-deltaH)):

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
                raise RuntimeError(f" Chain -> {chain}:"
                                   f" Unexpected error happened at iteration {i}.")
            # _end_if_

            # Save the current energy value.
            chain_stats["Energies"].append(E)

            # These are not stored during the burn-in period (i < 0).
            if i >= 0:
                chain_stats["Samples"].append(x)
                chain_stats["Accepted"].append(acc_ratio)
            # _end_if_

            # Check for display.
            if allow_display:

                # Update every 'n' iterations.
                if (i >= 0) and (np.mod(i, 500) == 0):

                    # Update the description in the tqdm.
                    chain_range.set_description(f" Chain -> {chain}:"
                                                f" Iter={i} | E={E:.3f} |"
                                                f" Acceptance={acc_ratio:.3f}")
                # _end_if_

            # _end_if_

        # _end_for_

        # Execution time (in seconds).
        time_elapsed = perf_counter() - t0

        # Display finish message.
        print(f" >>> Chain -> {chain}"
              f" finished in {time_elapsed:.3f} seconds.", flush=True)

        # Store the elapsed time.
        chain_stats["Elapsed_Time"] = time_elapsed

        # Check numerically the gradients.
        if options['grad_check']:

            # Get the grad-check error.
            diff_error = check_grad(_func, _grad, x.copy(), *args)

            # Display the error information.
            print(f"Chain -> {chain}: Grad-Check "
                  f"error <AFTER> sampling = {diff_error:.3E}", flush=True)
        # _end_if_

        # Return the local dictionary.
        return chain, chain_stats
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
        x = np.asfarray(x0).flatten()

        # Get the number of parallel chains.
        n_chains = self._options["n_chains"]

        # Make sure we clear the previous data.
        self._stats = {f"Chain-{i}": None for i in range(n_chains)}

        # Local copies of the methods.
        _func = self.func
        _grad = self.grad
        _single_chain = self._sample_single_chain

        # Check numerically the gradients.
        if self._options['grad_check']:

            # Get the grad-check error.
            diff_error = check_grad(_func, _grad, x.copy(), *args)

            # Display the error.
            print(f"Grad-Check error <BEFORE> sampling = {diff_error:.3E}\n")

        # _end_if_

        # Get the random seed.
        rng = np.random.default_rng(self._options["rng_seed"].get_state()[1])

        # List with the perturbed initial positions.
        x_init = []

        # Create the perturbed 'x' initial points.
        for _ in range(n_chains):
        
            # Perturb the initial 'x' with N(0, 0.1).
            x_init.append(x + 0.1*rng.standard_normal(x.size))
            
        # _end_for_
        
        # Get the number of CPUs. Leave one CPU out.
        n_cpus = max(cpu_count()-1, 1)
        
        # Check if the number of CPUS exceeds the number of chains.
        if n_cpus > n_chains:

            # In this case downgrade the number of cpus,
            # because we can't use more the 1 CPU/Chain.
            n_cpus = n_chains
        # _end_if_

        # Display start message.
        print(f"HMC started with {n_chains} chain(s) ... ")

        # First time.
        t0 = perf_counter()

        # Run the multiple chains in parallel.
        results = Parallel(n_jobs=n_cpus)(
            delayed(_single_chain)(x_init[i], _func, _grad, i,
                                   deepcopy(self._options), *args) for i in range(n_chains)
        )

        # Final time.
        tf = perf_counter()

        # Display finish message.
        print(f"HMC finished in {tf - t0:.3f} seconds.")

        # Extract all the result.
        for result_i in results:

            # Get the chain number.
            i = result_i[0]

            # Store each chain data separately.
            self._stats[f"Chain-{i}"] = result_i[1]
        # _end_for_

        # Return the dictionary.
        return self._stats
    # _end_if_

    # Auxiliary method.
    def acf(self, lag_n=None):
        """
        Computes the sample auto-correlation function values of the energy,
        for a given lag number.

        :param lag_n: Lag value to compute the acf.

        :return: a list (of lists) with the acf values for all stored chains.
        """

        # Sanity check.
        if self._stats is None:
            raise NotImplementedError(f"{self.__class__.__name__}: Stats dictionary is not implemented.")
        # _end_if_

        # A list that will hold the ACF values
        # for all the chains in the dictionary.
        acf_list = []

        # Compute the ACF values for every stored chain.
        for i, chain in enumerate(self._stats.values()):

            # Make sure the samples are an array.
            x = np.asfarray(chain["Energies"])

            # Remove singleton dimensions.
            x = x.squeeze()

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

import unittest
import numpy as np
from src.hmc_sampler import HMC


class TestHMC(unittest.TestCase):

    @staticmethod
    def rose_func(v, a=1.15, b=0.5):
        x, y = v
        return (a - x) ** 2 + b * (y - x ** 2) ** 2
    # _end_def_

    @staticmethod
    def rose_grad(v, a=1.15, b=0.5):
        x, y = v
        return np.array([2.0 * (x - a) - 4.0 * b * x * (y - x ** 2),
                         2.0 * b * (y - x ** 2)])
    # _end_def_

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestHMC - START -")

        # Create a HMC sampler.
        cls.hmc_sample = HMC(cls.rose_func, cls.rose_grad, n_samples=10_000,
                             n_omitted=1_000, kappa=100, d_tau=0.01, n_chains=1)
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestHMC - FINISH -")
    # _end_def_

    def test_init(self) -> None:
        """
        Test the __init__ method for setting the parameters correctly.

        :return: None.
        """

        # Make sure the parameters are passed correctly.
        self.assertEqual(self.hmc_sample.n_samples, 10000)
        self.assertEqual(self.hmc_sample.n_omitted, 1000)
        self.assertEqual(self.hmc_sample.delta_tau, 0.01)
        self.assertEqual(self.hmc_sample.n_chains, 1)
        self.assertEqual(self.hmc_sample.kappa, 100)

        # Make sure the default parameters are set correctly.
        self.assertFalse(self.hmc_sample.generalized,
                         msg="Generalized is enabled")

        self.assertFalse(self.hmc_sample.grad_check,
                         msg="Grad-check is enabled")

        self.assertFalse(self.hmc_sample.verbose,
                         msg="Verbosity is enabled")
    # _end_def_

    def test_type_errors(self) -> None:
        """
        Test the accessor methods for type errors.

        :return None
        """

        # Type of samples should be integer.
        with self.assertRaises(TypeError):
            self.hmc_sample.n_samples = 1.0
        # _end_with_

        # Type of omitted samples should be integer.
        with self.assertRaises(TypeError):
            self.hmc_sample.n_omitted = 1.0
        # _end_with_

        # Type of leap-frog steps should be integer.
        with self.assertRaises(TypeError):
            self.hmc_sample.kappa = 1.0
        # _end_with_

        # Type of leap-frog time step should be a float.
        with self.assertRaises(TypeError):
            self.hmc_sample.delta_tau = 100
        # _end_with_
    # _end_def_

    def test_value_errors(self) -> None:
        """
        Test the accessor methods for value errors.

        :return: None.
        """

        # Value of samples should always be positive.
        with self.assertRaises(ValueError):
            self.hmc_sample.n_samples = -1
        # _end_with_

        # Value of omitted samples should always be positive.
        with self.assertRaises(ValueError):
            self.hmc_sample.n_omitted = -1
        # _end_with_

        # Value of leap-frog steps should always be positive.
        with self.assertRaises(ValueError):
            self.hmc_sample.kappa = -1
        # _end_with_

        # Value of leap-frog time step should always be positive.
        with self.assertRaises(ValueError):
            self.hmc_sample.delta_tau = -0.01
        # _end_with_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()

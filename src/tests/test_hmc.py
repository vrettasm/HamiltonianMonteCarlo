import unittest
import numpy as np
from src.hmc_sampler import HMC


class TestHMC(unittest.TestCase):

    @classmethod
    def rose_func(cls, v, a=1.15, b=0.5):
        x, y = v
        return (a - x) ** 2 + b * (y - x ** 2) ** 2
    # _end_def_

    @classmethod
    def rose_grad(cls, v, a=1.15, b=0.5):
        x, y = v
        return np.array([2.0 * (x - a) - 4.0 * b * x * (y - x ** 2),
                         2.0 * b * (y - x ** 2)])
    # _end_def_

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestHMC - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestHMC - FINISH -")
    # _end_def_

    def test_init(self) -> None:
        """
        Test the __init__ method.

        :return: None.
        """

        # Create a HMC sampler.
        hmc_sample = HMC(self.rose_func, self.rose_grad)

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()

import numpy as np

__all__ = ["_LN10", "_RNG", "_RNG2"]

_LN10 = np.log(10.)
_RNG = np.random.default_rng(42) # for all important random numbers
_RNG2 = np.random.default_rng(17) # for random numbers used just in plotting


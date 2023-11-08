import numpy as np

__all__ = ["_LN10", "_RNG", "_RNG2", "_LN_NOISE"]

_LN10 = np.log(10.)
_RNG = np.random.default_rng(42) # for all important random numbers
_RNG2 = np.random.default_rng(17) # for random numbers used just in plotting
_LN_NOISE = -8. # used in Aq step....what is this for?? 
#This was -4 and the algorithm was not taking any steps. Does take steps if I turn this down

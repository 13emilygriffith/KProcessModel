
import numpy as np
import jax.numpy as jnp

from .data import FitParams
from .regularize import Regs
from .optimize import A_step, q_step, inflate_ivars


__all__ = ["initialize_2"]

def initialize_2(AbundData, FixedParams): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	assert FixedParams.K <= 2

	regs = Regs(AbundData, FixedParams)
	fitParams = FitParams(AbundData, FixedParams)

	fitParams.lnqs = jnp.where(regs.Q.fixed, regs.Q.lnq0s, fitParams.lnqs)

	I = [el in [FixedParams.CC_elem, FixedParams.Ia_elem] for el in AbundData.elements]
	fitParams.lnAs, _ = A_step(AbundData, FixedParams, fitParams.lnqs, fitParams.lnAs)
	print("initialize_2():", np.median(fitParams.lnAs[1:] - fitParams.lnAs[0], axis=1))

	fitParams.lnqs, _ = q_step(AbundData, FixedParams, fitParams.lnqs, fitParams.lnAs)
	fitParams.lnAs, _ = A_step(AbundData, FixedParams, fitParams.lnqs, fitParams.lnAs)
	print("initialize_2():", np.median(fitParams.lnAs[1:] - fitParams.lnAs[0], axis=1))

	AbundData.allivars = inflate_ivars(AbundData, FixedParams, fitParams, Q=7)
	return AbundData, fitParams
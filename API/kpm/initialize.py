
import numpy as np
import jax.numpy as jnp
import os
import pickle

from .data import fit_params
from .regularize import regularizations
from .optimize import A_step, q_step, inflate_ivars, Aq_step


__all__ = ["initialize_2", "initialize_As", "find_As"]

def initialize_2(data, fixed): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	assert fixed.K <= 2

	regs = regularizations(data, fixed)
	fit = fit_params(data, fixed)

	fit.lnq_pars = jnp.where(regs.Q.fixed_q, regs.Q.lnq_par0s, fit.lnq_pars)

	I = [el in [fixed.CC_elem, fixed.Ia_elem] for el in data.elements]
	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=I)
	print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))

	fit.lnq_pars, _ = q_step(data, fixed, fit.lnq_pars, fit.lnAs)
	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I)
	print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))

	data.allivars = inflate_ivars(data, fixed, fit, Q=7)
	return data, fit

def initialize_As(data, fixed, fit_old): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	fit = find_As(data, fixed, fit_old.lnq_pars)

	data.allivars = inflate_ivars(data, fixed, fit, Q=5)
	return data, fit


def find_As(data, fixed, lnq_pars): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	assert fixed.K <= 2

	fit = fit_params(data, fixed)
	fit.lnq_pars = lnq_pars

	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I)

	return fit


def run_kpm(data, fixed, fit, file_path, name='kpm_allstars', N_rounds=3, N_itters=16, overwrite=False):
	"""
	Maybe this should move to a different file

	Add notes
	N_itters = number of itterations of Aq step to run per round
	N_rounds = number of rounds of itterations. After each itteration we recalculate the ivars and save

	"""

	# add error handeling so that N_rounds > 0 and N_itters > 0
	# set some maximum limit on the number of itterations

	# initialize fit
	# should only initialize if first file doesn't exist
	# Need to update so if early file is missing but later isn't it reruns whole thing

	#data, fit = initialize_2(data, fixed)

	pik_suffix = file_path+'/'+name+'_K'+str(fixed.K)+'_qccFe'+str(fixed.q_CC_Fe)+'_J'+str(fixed.J)

	# run round of itterations
	for r in range(N_rounds):
		pik_name = pik_suffix + '_' + str(r) + '.out'
		if (os.path.isfile(pik_name)==True) and (overwrite==False):
			print('File %s exists, loading data' % (pik_name))
			with open(pik_name, "rb") as f:
				fit, fixed = pickle.load(f)
				# NEED TO CHECK IF FIXED IS THE SAME AS ORIGINAL
				# NEED TO CHECK IF EMPTY

		elif (os.path.isfile(pik_name)==False) or (overwrite==True):
			if (os.path.isfile(pik_name)==True) and (overwrite==True):
				print('File %s exists, but overwriting' % (pik_name))

			for i in range(N_itters):
				fit = Aq_step(data, fixed, fit)
			
			save = [fit, fixed]
			with open(pik_name, "wb") as f:
				pickle.dump(save, f)

			data.allivars = inflate_ivars(data, fixed, fit)

	data.allivars = data.allivars_orig
	return data, fixed, fit
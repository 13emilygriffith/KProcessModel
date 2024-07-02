
import numpy as np
import jax.numpy as jnp
import os
import pickle

from .data import fit_params
from .regularize import regularizations
from .optimize import A_step, q_step, inflate_ivars, Aq_step
from ._globals import _LN10


__all__ = ["initialize", "initialize_As", "find_As", "initialize_from_2"]

def initialize(data, fixed, verbose=False): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- very brittle
	- relies on global variables
	"""

	regs = regularizations(data, fixed)
	fit = fit_params(data, fixed)

	fit.lnq_pars = jnp.where(regs.Q.fixed_q, regs.Q.lnq_par0s, fit.lnq_pars)

	I = [el in fixed.proc_elems for el in data.elements]
	print(data.elements[I])

	D_temp = fixed.dilution
	fixed.dilution = False # just for initalization

	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=I, verbose=verbose)
	fit.lnAs = fit.lnAs.at[-1,:].set(np.zeros(data.N))
	if verbose: print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))

	fit.lnq_pars, _ = q_step(data, fixed, fit.lnq_pars, fit.lnAs, verbose=verbose)

	# fit.lnAs = fit.lnAs.at[:-1,:].set(fit.lnAs[:-1,:]+0.1)
	# fit.lnAs = fit.lnAs.at[-1,:].set(np.ones(data.N)*0.1/_LN10) #trying to give an initial value
	fit.lnAs = fit.lnAs.at[-1,:].set(np.zeros(data.N))
	fixed.dilution = D_temp
	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I, verbose=verbose)
	if verbose: print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))

	data.allivars = inflate_ivars(data, fixed, fit, Q=7)
	return data, fit

def initialize_from_2(data, fixed, fit_2, verbose=False):
	fit = fit_params(data, fixed)
	fit.lnq_pars[:2,:,:] = fit_2.lnq_pars
	fit.lnAs[:2,:] = fit_2.lnAs[:2,:]

	regs = regularizations(data, fixed)
	fit.lnq_pars = jnp.where(regs.Q.fixed_q, regs.Q.lnq_par0s, fit.lnq_pars)

	I = [el in fixed.proc_elems for el in data.elements]

	D_temp = fixed.dilution
	fixed.dilution = False # just for initalization

	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=I, verbose=verbose)
	fit.lnAs = fit.lnAs.at[-1,:].set(np.zeros(data.N))
	if verbose: print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))

	fit.lnq_pars, _ = q_step(data, fixed, fit.lnq_pars, fit.lnAs, verbose=verbose)

	fit.lnAs = fit.lnAs.at[-1,:].set(np.zeros(data.N))
	fixed.dilution = D_temp
	fit.lnAs, _ = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I, verbose=verbose)
	if verbose: print("initialize_2():", np.median(fit.lnAs[1:] - fit.lnAs[0], axis=1))
	# fit.lnAs = fit.lnAs.at[2,:].set(np.zeros(data.N))
	# fixed.dilution = D_temp

	data.allivars = inflate_ivars(data, fixed, fit, Q=7)
	return data, fit
	

def initialize_As(data, fixed, fit_old, verbose=False): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	fit = find_As(data, fixed, fit_old.lnq_pars, verbose=verbose)

	data.allivars = inflate_ivars(data, fixed, fit, Q=5)
	return data, fit


def find_As(data, fixed, lnq_pars, verbose=False): 
	#, K, q_CC_Fe, dq_CC_Fe_dZ, elements, knot_xs, alldata, allivars_orig):
	"""
	## Bugs:
	- DOESN'T WORK for K > 2 ??
	- very brittle
	- relies on global variables
	"""
	#assert fixed.K <= 2

	fit = fit_params(data, fixed)
	fit.lnq_pars = lnq_pars

	if fixed.D == 1:
		fit.lnAs[-1,:] = np.ones(data.N)*0.00

	fixed.dilution = False
	new_lnAs, dc2 = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I, verbose=verbose)
	print('find_As: delta chi2 ', np.nansum(dc2),
		'number of bad values: ', np.sum(np.logical_not(np.isfinite(dc2))),
		'number of worse stars:', np.sum(dc2<-0.01))
	fixed.dilution = True
	fit.lnAs = new_lnAs
	for itter in range(10):
		new_lnAs, dc2 = A_step(data, fixed, fit.lnq_pars, fit.lnAs, I=fixed.I, verbose=verbose)
		print('find_As: delta chi2 ', np.nansum(dc2),
			'number of bad values: ', np.sum(np.logical_not(np.isfinite(dc2))),
			'number of worse stars:', np.sum(dc2<-0.01))
		fit.lnAs = new_lnAs
	return fit


def run_kpm(data, fixed, fit, file_path, name='kpm_allstars', N_rounds=3, N_itters=16, overwrite=False, verbose=False):
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

	if fixed.K>1:
		pik_suffix = file_path+'/'+name+'_K'+str(fixed.K)+'_qccFe'+str(fixed.q_fixed[1,0])+'_J'+str(fixed.J)
	else: 
		pik_suffix = file_path+'/'+name+'_K'+str(fixed.K)+'_J'+str(fixed.J)

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
				fit = Aq_step(data, fixed, fit, verbose=verbose)
				if fixed.D==0: 
					fit.lnAs = fit.lnAs.at[-1,:].set(np.zeros(data.N))
			
			save = [fit, fixed]
			with open(pik_name, "wb") as f:
				pickle.dump(save, f)

			data.allivars = inflate_ivars(data, fixed, fit)

	data.allivars = data.allivars_orig
	return data, fixed, fit
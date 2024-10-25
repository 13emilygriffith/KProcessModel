import numpy as np 
import jax.numpy as jnp
from jax import vmap

from ._globals import _RNG
from .regularize import regularizations
from .general import all_stars_KPM
from .general import internal_get_lnqs
from .one_star import one_star_A_step, one_element_q_step, one_element_chi, one_star_chi

__all__ = ["inflate_ivars", "A_step", "q_step", "Aq_step", "run_kpm"]


def inflate_ivars(data, fixed, fit, Q=5):
	"""
	Inflate `ivars` to discourage outliers from skewing model fits

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `fit`: KPM `fit_params` class
	- `Q`: int strength of ivar inflation

	Outputs
	-------
	- shape `(M, )` new array of ivars

	Comments
	--------
	- This sucks

	"""
	synth = all_stars_KPM(fixed, fit)
	diff = data.alldata - synth
	chi = diff * jnp.sqrt(data.allivars_orig)

	return (data.allivars_orig * Q**2) / (Q**2 + diff**2 * data.allivars_orig)

def A_step(data, fixed, lnq_pars, lnAs, I=None, verbose=False):
	"""
	Optimize the `lnAs` for all stars

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `lnq_pars`: shape `(K, J, M)` natural-logarithmic processes
	- `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
	- `I`: shape `(M)` boolean of element ids to include in the fit
	- `verbose`: boolean print fit information

	Outputs
	-------
	shape `(K, N)` best-fit natural-logarithmic amplitudes

	Comments
	--------
	- Ridiculous post-processing of outputs, with MAGIC numbers.
	"""
   
	regs = regularizations(data, fixed) 
	if I==None: I = [True]*data.M

	new_lnAs, dc2 = vmap(one_star_A_step, in_axes=(1, 0, 0, None, None, 1),
					out_axes=(1, 0))(internal_get_lnqs(lnq_pars[:,:,I], fixed.L, fixed.xs, lnAs, fixed.Delta), 
					data.alldata[:,I], data.sqrt_allivars[:,I], 
					regs.A.sqrt_Lambda_A, fixed.Delta, lnAs)
	new_lnAs = jnp.where(dc2 < 0, lnAs, new_lnAs)
	if not jnp.all(jnp.isfinite(new_lnAs)):
		if verbose: print("A-step(): fixing bad elements:", jnp.sum(jnp.logical_not(jnp.isfinite(new_lnAs))))
		new_lnAs = jnp.where(jnp.isfinite(new_lnAs), new_lnAs, lnAs)
	if np.any(new_lnAs > 2.0): # MAGIC HACK
		if verbose: print("A-step(): fixing large elements:", np.sum(new_lnAs > 2.0), np.max(new_lnAs))
		new_lnAs = jnp.where(new_lnAs > 2.0, 2.0, new_lnAs)
	if np.any(new_lnAs < -9.0): # MAGIC HACK
		if verbose: print("A-step(): fixing small elements:", np.sum(new_lnAs < -9.0), np.min(new_lnAs))
		new_lnAs = jnp.where(new_lnAs < -9.0, -9.0, new_lnAs)
	return new_lnAs, dc2


def q_step(data, fixed, lnq_pars, lnAs, verbose=False):
	"""
	Optimize the `lnq_pars` for all stars

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `lnq_pars`: shape `(K, J, M)` natural-logarithmic processes
	- `lnAs`: shape `(K, N)` natural-logarithmic amplitudes

	Outputs
	-------
	shape `(K, J, M)` best-fit natural-logarithmic processes

	Comments
	--------
	- Ridiculous post-processing of outputs.
	"""
	
	regs = regularizations(data, fixed)

	new_lnq_pars, dc2 = vmap(one_element_q_step, in_axes=(None, 1, 1, None, None, 2, 2, 2, None, 2),
					out_axes=(2, 0))(lnAs, data.alldata, data.sqrt_allivars, 
					fixed.L, fixed.xs, jnp.sqrt(regs.Q.Lambdas), jnp.array(regs.Q.q0s),
					jnp.array(regs.Q.fixed_q), fixed.Delta, lnq_pars)
	new_lnq_pars = jnp.where(dc2 < 0, lnq_pars, new_lnq_pars)
	if not np.all(jnp.isfinite(new_lnq_pars)):
		if verbose: print("q-step(): fixing bad elements:", np.sum(jnp.logical_not(jnp.isfinite(new_lnq_pars))))
		new_lnq_pars = jnp.where(jnp.isfinite(new_lnq_pars), new_lnq_pars, fit.lnq_pars)
	if np.any(new_lnq_pars > 1.0): # MAGIC HACK
		if verbose: print("q-step(): fixing large elements:", np.sum(new_lnq_pars > 1.0), np.max(new_lnq_pars))
		new_lnq_pars = jnp.where(new_lnq_pars > 1.0, 1.0, new_lnq_pars)
	if np.any(new_lnq_pars < -9.0): # MAGIC HACK
		if verbose: print("q-step(): fixing small elements:", np.sum(new_lnq_pars < -9.0), np.min(new_lnq_pars))
		new_lnq_pars = jnp.where(new_lnq_pars < -9.0, -9.0, new_lnq_pars)
	return new_lnq_pars, dc2


def objective_q(data, fixed, lnq_pars, lnAs):
	"""
	This is NOT the objective, but it stands in for now!!

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `lnq_pars`: shape `(K, J, M)` natural-logarithmic processes
	- `lnAs`: shape `(K, N)` natural-logarithmic amplitudes

	Outputs
	-------
	- chi value for q step

	"""
	regs = regularizations(data, fixed)
	chi = vmap(one_element_chi, in_axes=(2, None, 1, 1, None, None, 2, 2, None),
			   out_axes=(0))(lnq_pars, lnAs, data.alldata, data.sqrt_allivars, fixed.L, fixed.xs,
							 jnp.sqrt(regs.Q.Lambdas), regs.Q.q0s, fixed.Delta)
	return np.sum(chi * chi) + np.sum((regs.A.sqrt_Lambda_A[:, None] * jnp.exp(lnAs[-1:])) ** 2)

def objective_A(data, fixed, lnq_pars, lnAs):
	"""
	This is NOT the objective, but it stands in for now!!

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `lnq_pars`: shape `(K, J, M)` natural-logarithmic processes
	- `lnAs`: shape `(K, N)` natural-logarithmic amplitudes

	Outputs
	-------
	- chi value for A step

	"""
	regs = regularizations(data, fixed)
	chi = vmap(one_star_chi, in_axes=(1, 1, 0, 0, None, None),
			   out_axes=(0))(lnAs, internal_get_lnqs(lnq_pars, fixed.L, fixed.xs, lnAs, fixed.Delta),
							 data.alldata, data.sqrt_allivars, regs.A.sqrt_Lambda_A, fixed.Delta)
	return np.sum(chi ** 2) + np.sum(regs.Q.Lambdas * (jnp.exp(lnq_pars) - regs.Q.q0s) ** 2)

def Aq_step(data, fixed, fit, verbose=False):
	"""
	Calls both the q step and A step iteratively, updating objective when fit improves

	Inputs
	------
	- `data`: KPM `abund_data` class
	- `fixed`: KPM `fixed_params` class
	- `fit`: KPM `fit_params` class

	Outputs
	-------
	- updated `fir_params` class

	Comments
	--------
	- This contains multiple hacks, especially the noisification hack.
	- Maybe some of the hacks should be pushed back into the A-step and
	  the q-step?
	- Relies on terrible global variables.
	"""
	prefix = "Aq-step():"

	# fix old_lnAs
	old_objective = objective_A(data, fixed, fit.lnq_pars, fit.lnAs)

	old_lnAs = jnp.where(jnp.isnan(fit.lnAs), 1., fit.lnAs)
	old_lnq_pars = fit.lnq_pars
	
	# add noise -- this is a hack to escape possible local minima.
	# Note the use of logaddexp, so things are non-intuitive here.
	A_noise = fixed.ln_noise + np.log(_RNG.uniform(size=old_lnAs.shape))
	init_lnAs = jnp.logaddexp(old_lnAs, A_noise)

	# run q step
	objective1 = objective_q(data, fixed, old_lnq_pars, init_lnAs)
	new_lnq_pars, _ = q_step(data, fixed, old_lnq_pars, init_lnAs, verbose=verbose)
	objective2 = objective_q(data, fixed, new_lnq_pars, init_lnAs)
	if objective2 > objective1:
		if verbose: print(prefix, "q-step WARNING: objective function got worse:", objective1, objective2)
		new_lnq_pars = old_lnq_pars.copy()
		objective2 = objective1

	# run A step
	objective3 = objective_A(data, fixed, new_lnq_pars, init_lnAs)
	new_lnAs, _ = A_step(data, fixed, new_lnq_pars, init_lnAs, I=fixed.I,  verbose=verbose)
	objective4 = objective_A(data, fixed, new_lnq_pars, new_lnAs)
	if objective4 > objective3:
		if verbose: print(prefix, "A-step WARNING: objective function got worse:", objective3, objective4)
		new_lnAs = init_lnAs.copy()
		objective4 = objective3

	# check objective
	if verbose: print(old_objective, objective1, objective2, objective3, objective4)
	if objective4 < old_objective:
		if verbose: print(prefix, "we took a step!", fixed.ln_noise, objective4, old_objective - objective4)
		# If we took a step, then we can be more aggressive with the noise we are adding (see above).
		fixed.ln_noise = fixed.ln_noise + 0.1 # Magic
		fit.lnq_pars = new_lnq_pars
		fit.lnAs = new_lnAs

	else:
		if verbose: print(prefix, "we didn't take a step :(", fixed.ln_noise, old_objective, old_objective - objective4)
		# If we didn't take a step, maybe it's because we added too much noise (see above)?
		fixed.ln_noise = fixed.ln_noise - 1.0
		fit.lnq_pars = old_lnq_pars.copy()
		fit.lnAs = old_lnAs.copy()

	return fit

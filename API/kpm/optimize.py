import numpy as np 
import jax.numpy as jnp
from jax import vmap

from ._globals import _RNG, _LN_NOISE
from .regularize import regularizations
from .general import all_stars_KPM
from .general import internal_get_lnqs
from .one_star import one_star_A_step, one_element_q_step, one_element_chi, one_star_chi

__all__ = ["inflate_ivars", "A_step", "q_step", "Aq_step", "run_kpm"]


def inflate_ivars(data, fixed, fit, Q=5):
	"""
	## inputs

	## outputs
	shape `(M, )` new array of ivars

	## comments
	- This sucks

	"""
	synth = all_stars_KPM(fixed, fit)
	diff = data.alldata - synth
	chi = diff * jnp.sqrt(data.allivars_orig)

	return (data.allivars_orig * Q**2) / (Q**2 + diff**2 * data.allivars_orig)

def A_step(data, fixed, lnq_pars, lnAs, I=None):
    """
    ## inputs
    ## outputs
    shape `(K, N)` best-fit natural-logarithmic amplitudes

    ## bugs
    - Ridiculous post-processing of outputs, with MAGIC numbers.
    """
   
    regs = regularizations(data, fixed) 
    if I==None: I = [True]*data.M

    new_lnAs, dc2 = vmap(one_star_A_step, in_axes=(1, 0, 0, None, 1),
                    out_axes=(1, 0))(internal_get_lnqs(lnq_pars[:,:,I], fixed.L, fixed.xs), 
                    data.alldata[:,I], data.sqrt_allivars[:,I], 
                    regs.A.sqrt_Lambda_A, lnAs)
    new_lnAs = jnp.where(dc2 < 0, lnAs, new_lnAs)
    if not jnp.all(jnp.isfinite(new_lnAs)):
        print("A-step(): fixing bad elements:", jnp.sum(jnp.logical_not(jnp.isfinite(new_lnAs))))
        new_lnAs = jnp.where(jnp.isfinite(new_lnAs), new_lnAs, lnAs)
    if np.any(new_lnAs > 2.0): # MAGIC HACK
        print("A-step(): fixing large elements:", np.sum(new_lnAs > 2.0), np.max(new_lnAs))
        new_lnAs = jnp.where(new_lnAs > 2.0, 2.0, new_lnAs)
    if np.any(new_lnAs < -9.0): # MAGIC HACK
        print("A-step(): fixing small elements:", np.sum(new_lnAs < -9.0), np.min(new_lnAs))
        new_lnAs = jnp.where(new_lnAs < -9.0, -9.0, new_lnAs)
    return new_lnAs, dc2

def q_step(data, fixed, lnq_pars, lnAs):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `alldata`: shape `(N, M)` log_10 abundance measurements
    - `sqrt_allivars`: shape `(N, M)` inverse errors on alldata
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - `old_lnqs`: shape `(K, Nbin, M)` initialization for optimizations

    ## outputs
    shape `(K, Nbin, M)` best-fit natural-logarithmic processes

    ## bugs
    - Ridiculous post-processing of outputs.
    """
    
    regs = regularizations(data, fixed)

    new_lnq_pars, dc2 = vmap(one_element_q_step, in_axes=(None, 1, 1, None, None, 2, 2, 2, 2),
                    out_axes=(2, 0))(lnAs, data.alldata, data.sqrt_allivars, 
                    fixed.L, fixed.xs, jnp.sqrt(regs.Q.Lambdas), jnp.array(regs.Q.q0s),
                    jnp.array(regs.Q.fixed_q), lnq_pars)
    new_lnq_pars = jnp.where(dc2 < 0, lnq_pars, new_lnq_pars)
    if not np.all(jnp.isfinite(new_lnq_pars)):
        print("q-step(): fixing bad elements:", np.sum(jnp.logical_not(jnp.isfinite(new_lnq_pars))))
        new_lnq_pars = jnp.where(jnp.isfinite(new_lnq_pars), new_lnq_pars, fit.lnq_pars)
    if np.any(new_lnq_pars > 1.0): # MAGIC HACK
        print("q-step(): fixing large elements:", np.sum(new_lnq_pars > 1.0), np.max(new_lnq_pars))
        new_lnq_pars = jnp.where(new_lnq_pars > 1.0, 1.0, new_lnq_pars)
    if np.any(new_lnq_pars < -9.0): # MAGIC HACK
        print("q-step(): fixing small elements:", np.sum(new_lnq_pars < -9.0), np.min(new_lnq_pars))
        new_lnq_pars = jnp.where(new_lnq_pars < -9.0, -9.0, new_lnq_pars)
    return new_lnq_pars, dc2


def objective_q(data, fixed, lnq_pars, lnAs):
    """
    This is NOT the objective, but it stands in for now!!
    """
    regs = regularizations(data, fixed)
    chi = vmap(one_element_chi, in_axes=(2, None, 1, 1, None, None, 2, 2),
               out_axes=(0))(lnq_pars, lnAs, data.alldata, data.sqrt_allivars, fixed.L, fixed.xs,
                             jnp.sqrt(regs.Q.Lambdas), regs.Q.q0s)
    return np.sum(chi * chi) + np.sum((regs.A.sqrt_Lambda_A[:, None] * jnp.exp(lnAs)) ** 2)

def objective_A(data, fixed, lnq_pars, lnAs):
    """
    This is NOT the objective, but it stands in for now!!
    """
    regs = regularizations(data, fixed)
    chi = vmap(one_star_chi, in_axes=(1, 1, 0, 0, None),
               out_axes=(0))(lnAs, internal_get_lnqs(lnq_pars, fixed.L, fixed.xs),
                             data.alldata, data.sqrt_allivars, regs.A.sqrt_Lambda_A)
    return np.sum(chi ** 2) + np.sum(regs.Q.Lambdas * (jnp.exp(lnq_pars) - regs.Q.q0s) ** 2)

def Aq_step(data, fixed, fit):
    """
    ## Bugs:
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
    A_noise = _LN_NOISE + np.log(_RNG.uniform(size=old_lnAs.shape))
    init_lnAs = jnp.logaddexp(old_lnAs, A_noise)
    q_noise = _LN_NOISE + np.log(_RNG.uniform(size=old_lnq_pars.shape))
    q_noise[:, :, data.elements == "Mg"] = -np.inf # HACK 
    q_noise[:, :, data.elements == "Fe"] = -np.inf # HACK
    init_lnq_pars = jnp.logaddexp(old_lnq_pars, q_noise)

    # run q step
    objective1 = objective_q(data, fixed, old_lnq_pars, init_lnAs)
    new_lnq_pars, _ = q_step(data, fixed, old_lnq_pars, init_lnAs)
    objective2 = objective_q(data, fixed, new_lnq_pars, init_lnAs)
    if objective2 > objective1:
        print(prefix, "q-step WARNING: objective function got worse:", objective1, objective2)
        new_lnq_pars = old_lnq_pars.copy()
        objective2 = objective1

    # run A step
    objective3 = objective_A(data, fixed, new_lnq_pars, init_lnAs)
    new_lnAs, _ = A_step(data, fixed, new_lnq_pars, init_lnAs, I=fixed.I)
    objective4 = objective_A(data, fixed, new_lnq_pars, new_lnAs)
    if objective4 > objective3:
        print(prefix, "A-step WARNING: objective function got worse:", objective3, objective4)
        new_lnAs = init_lnAs.copy()
        objective4 = objective3

    # check objective
    print(old_objective, objective1, objective2, objective3, objective4)
    if objective4 < old_objective:
        print(prefix, "we took a step!", _LN_NOISE, objective4, old_objective - objective4)
	# If we took a step, then we can be more aggressive with the noise we are adding (see above).
	_LN_NOISE = np.around(_LN_NOISE + 0.1, 1) # Gross global variable!
        fit.lnq_pars = new_lnq_pars
        fit.lnAs = new_lnAs

    else:
        print(prefix, "we didn't take a step :(", _LN_NOISE, old_objective, old_objective - objective4)
        # If we didn't take a step, maybe it's because we added too much noise (see above)?
	_LN_NOISE = np.around(_LN_NOISE - 1.0, 1)
        fit.lnq_pars = old_lnq_pars.copy()
        fit.lnAs = old_lnAs.copy()

    return fit

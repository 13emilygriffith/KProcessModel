import numpy as np 
import jax.numpy as jnp
from jax import vmap

from ._globals import _RNG, _LN_NOISE
from .regularize import regularizations
from .general import all_stars_KPM
from .general import internal_get_lnqs as get_lnqs
from .one_star import one_star_A_step, one_element_q_step, one_element_chi, one_star_chi

__all__ = ["inflate_ivars", "A_step", "q_step", "Aq_step", "run_kpm"]


def inflate_ivars(data, fixed, fit, Q=5):
	"""
	## inputs
	- `data`: shape `(M, )` log_10 abundance measurements
	- `ivars`: shape `(M, )` inverse errors on the data
	- `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
	- `lnqs`: shape `(K, Nknot, M)` natural-logarithmic processes
	- `knot_xs`: shape `(Nknot, )` metallicity bin centers
	- `xs`: shape `(N, )` abundance data (used to interpolate the `lnqs`)
	- Q softening parameter, smaller is more agressive 

	## outputs
	shape `(M, )` new array of ivars

	## comments
	- This sucks

	"""
	synth = all_stars_KPM(fixed, fit)
	diff = data.alldata - synth
	chi = diff * jnp.sqrt(data.allivars_orig)

	return (data.allivars_orig * Q**2) / (Q**2 + diff**2 * data.allivars_orig)

def A_step(data, fixed, lnqs, lnAs, I=None):
    """
    ## inputs
    - `lnqs`: shape `(K, Nknot, M)` natural-logarithmic processes
    - `data`: shape `(N, M)` log_10 abundance measurements
    - `sqrt_ivars`: shape `(N, M)` inverse variances on alldata
    - `knot_xs`: shape `(Nknot, )` metallicity knot locations
    - `xs`: shape `(N, )` metallicities (to use with the knots)
    - `old_lnAs`: previous `lnAs`; used for initialization of the optimizer

    ## outputs
    shape `(K, N)` best-fit natural-logarithmic amplitudes

    ## bugs
    - Ridiculous post-processing of outputs, with MAGIC numbers.
    """
   
    regs = regularizations(data, fixed) 
    if I==None: I = [True]*data.M

    new_lnAs, dc2 = vmap(one_star_A_step, in_axes=(1, 0, 0, None, 1),
                    out_axes=(1, 0))(get_lnqs(fixed.K, lnqs[:,:,I], fixed.knot_xs, fixed.xs), 
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

def q_step(data, fixed, lnqs, lnAs):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `alldata`: shape `(N, M)` log_10 abundance measurements
    - `sqrt_allivars`: shape `(N, M)` inverse errors on alldata
    - `knot_xs`: shape `(Nknot, )` metallicity bin centers
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - `old_lnqs`: shape `(K, Nbin, M)` initialization for optimizations

    ## outputs
    shape `(K, Nbin, M)` best-fit natural-logarithmic processes

    ## bugs
    - Ridiculous post-processing of outputs.
    """
    
    regs = regularizations(data, fixed)

    new_lnqs, dc2 = vmap(one_element_q_step, in_axes=(None, 1, 1, None, None, 2, 2, 2, 2),
                    out_axes=(2, 0))(lnAs, data.alldata, data.sqrt_allivars, 
                    fixed.knot_xs, fixed.xs, jnp.sqrt(regs.Q.Lambdas), jnp.array(regs.Q.q0s),
                    jnp.array(regs.Q.fixed_q), lnqs)
    new_lnqs = jnp.where(dc2 < 0, lnqs, new_lnqs)
    if not np.all(jnp.isfinite(new_lnqs)):
        print("q-step(): fixing bad elements:", np.sum(jnp.logical_not(jnp.isfinite(new_lnqs))))
        new_lnqs = jnp.where(jnp.isfinite(new_lnqs), new_lnqs, fit.lnqs)
    if np.any(new_lnqs > 1.0): # MAGIC HACK
        print("q-step(): fixing large elements:", np.sum(new_lnqs > 1.0), np.max(new_lnqs))
        new_lnqs = jnp.where(new_lnqs > 1.0, 1.0, new_lnqs)
    if np.any(new_lnqs < -9.0): # MAGIC HACK
        print("q-step(): fixing small elements:", np.sum(new_lnqs < -9.0), np.min(new_lnqs))
        new_lnqs = jnp.where(new_lnqs < -9.0, -9.0, new_lnqs)
    return new_lnqs, dc2


def objective_q(data, fixed, lnqs, lnAs):
    """
    This is NOT the objective, but it stands in for now!!
    """
    regs = regularizations(data, fixed)
    chi = vmap(one_element_chi, in_axes=(2, None, 1, 1, None, None, 2, 2),
               out_axes=(0))(lnqs, lnAs, data.alldata, data.sqrt_allivars, fixed.knot_xs, fixed.xs,
                             jnp.sqrt(regs.Q.Lambdas), regs.Q.q0s)
    return np.sum(chi * chi) + np.sum((regs.A.sqrt_Lambda_A[:, None] * jnp.exp(lnAs)) ** 2)

def objective_A(data, fixed, lnqs, lnAs):
    """
    This is NOT the objective, but it stands in for now!!
    """
    regs = regularizations(data, fixed)
    chi = vmap(one_star_chi, in_axes=(1, 1, 0, 0, None),
               out_axes=(0))(lnAs, get_lnqs(fixed.K, lnqs, fixed.knot_xs, fixed.xs),
                             data.alldata, data.sqrt_allivars, regs.A.sqrt_Lambda_A)
    return np.sum(chi ** 2) + np.sum(regs.Q.Lambdas * (jnp.exp(lnqs) - regs.Q.q0s) ** 2)

def Aq_step(data, fixed, fit):
    """
    ## Bugs:
    - This contains multiple hacks.
    - Maybe some of the hacks should be pushed back into the A-step and
      the q-step?
    """
    prefix = "Aq-step():"

    # fix old_lnAs
    old_lnAs = jnp.where(jnp.isnan(fit.lnAs), 1., fit.lnAs)
    old_lnqs = fit.lnqs
    old_objective = objective_A(data, fixed, old_lnqs, old_lnAs)

    # add noise
    A_noise = _LN_NOISE + np.log(_RNG.uniform(size=old_lnAs.shape))
    init_lnAs = jnp.logaddexp(old_lnAs, A_noise)
    q_noise = _LN_NOISE + np.log(_RNG.uniform(size=old_lnqs.shape))
    q_noise[:, :, data.elements == "Mg"] = -np.inf # HACK 
    q_noise[:, :, data.elements == "Fe"] = -np.inf # HACK
    init_lnqs = jnp.logaddexp(old_lnqs, q_noise)

    # run q step
    objective1 = objective_q(data, fixed, old_lnqs, init_lnAs)
    new_lnqs, _ = q_step(data, fixed, old_lnqs, init_lnAs)
    objective2 = objective_q(data, fixed, new_lnqs, init_lnAs)
    if objective2 > objective1:
        print(prefix, "q-step WARNING: objective function got worse:", objective1, objective2)
        new_lnqs = old_lnqs.copy()
        objective2 = objective1

    # run A step
    objective3 = objective_A(data, fixed, new_lnqs, init_lnAs)
    new_lnAs, _ = A_step(data, fixed, new_lnqs, init_lnAs)
    objective4 = objective_A(data, fixed, new_lnqs, new_lnAs)
    if objective4 > objective3:
        print(prefix, "A-step WARNING: objective function got worse:", objective3, objective4)
        new_lnAs = init_lnAs.copy()
        objective4 = objective3

    # check objective
    print(old_objective, objective1, objective2, objective3, objective4)
    if objective4 < old_objective:
        print(prefix, "we took a step!", _LN_NOISE, objective4, old_objective - objective4)
        #return new_lnAs, new_lnqs, np.around(_LN_NOISE + 0.1, 1)	why return noise?
        fit.lnqs = new_lnqs
        fit.lnAs = new_lnAs

    else:
        print(prefix, "we didn't take a step :(", _LN_NOISE, old_objective, old_objective - objective4)
        #return old_lnAs.copy(), old_lnqs.copy(), np.around(ln_noise - 1.0, 1)	 why return noise?
        # Can we just return original class?
        fit.lnqs = old_lnqs.copy()
        fit.lnAs = old_lnAs.copy()

    return fit








from ._globals import _LN10

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import vmap
import numpy as np

def get_lnqs(fixed, fit):
    """
    linear interpolation on vmap?
    """
    return jnp.concatenate([vmap(jnp.interp, in_axes=(None, None, 1),
                                 out_axes=(1))(fixed.xs, fixed.knot_xs[k], fit.lnqs[k])[None, :, :]
                            for k in range(fixed.K)], axis=0)

## Not needed 
# def get_processes(K):
#     processes_all = np.array(['CC', 'Ia', 'third', 'fourth']) # process names
#     processes = processes_all[:K]
#     return processes

def all_stars_KPM(fixed, fit):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, Nknot, M)` natural-logarithmic processes
    - `knot_xs`: shape `(Nknot, )` metallicity bin centers
    - `xs`: shape `(N, )` abundance data (used to interpolate the `lnqs`)

    ## outputs
    shape `(M, )` log_10 abundances

    ## comments
    - Note the `ln10`.
    """
    return logsumexp(fit.lnAs[:, :, None]
                     + get_lnqs(fixed, fit), axis=0) / _LN10

def internal_get_lnqs(K, lnqs, knot_xs, xs):
    """
    linear interpolation on vmap?
    """
    return jnp.concatenate([vmap(jnp.interp, in_axes=(None, None, 1),
                                 out_axes=(1))(xs, knot_xs[k], lnqs[k])[None, :, :]
                            for k in range(K)], axis=0)
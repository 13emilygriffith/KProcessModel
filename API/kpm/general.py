from ._globals import _LN10

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import vmap
import numpy as np

def get_lnqs(fixed, fit):
    """
    see `internal_get_lnqs` for actual information.
    """
    return internal_get_lnqs(fit.lnq_pars, fixed.L, fixed.xs)

## Not needed 
# def get_processes(K):
#     processes_all = np.array(['CC', 'Ia', 'third', 'fourth']) # process names
#     processes = processes_all[:K]
#     return processes

def all_stars_KPM(fixed, fit):
    """
    ## inputs -- actually these are things in `fixed` and `fit`
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `lnq_pars`: shape `(K, 2*J+1, M)` amplitudes
    - `xlim`: shape `(2, )` metallicity range
    - `xs`: shape `(N, )` abundance data (used to get the `lnqs`)

    ## outputs
    shape `(M, )` log_10 abundances

    ## comments
    - Note the `ln10`.
    """
    return logsumexp(fit.lnAs[:, :, None]
                     + get_lnqs(fixed, fit), axis=0) / _LN10

def fourier_sum(amps, argument):
    foo = amps[0]
    for j in range(1, (len(amps) - 1) // 2 + 1):
        foo += amps[2*j - 1] * jnp.cos(j * argument) \
             + amps[2*j]     * jnp.sin(j * argument)
    return foo

fourier_sum_orama = vmap(vmap(fourier_sum, in_axes=(0, None), out_axes=1), \
    in_axes=(0, None), out_axes=0)

def internal_get_lnqs(lnq_pars, L, xs):
    """
    sums of sines and cosines
    """
    tmp = jnp.swapaxes(lnq_pars, 1, 2)
    return fourier_sum_orama(tmp, xs / L)

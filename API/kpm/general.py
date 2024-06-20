from ._globals import _LN10

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import vmap
import numpy as np

def get_lnqs(fixed, fit):
    """
    see `internal_get_lnqs` for actual information.
    """
    return internal_get_lnqs(fit.lnq_pars, fixed.L, fixed.xs, fit.lnAs, fixed.D)

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
    return (logsumexp(fit.lnAs[:-1, :, None]
                     + get_lnqs(fixed, fit), axis=0) / _LN10) - (fixed.D * jnp.abs(fit.lnAs[-1,:,None]))

def fourier_sum(amps, argument):
    foo = amps[0] * jnp.ones_like(argument)
    for j in range(1, (len(amps) - 1) // 2 + 1):
        #print('fourier_sum', j)
        foo += amps[2*j - 1] * jnp.cos(j * argument) \
             + amps[2*j]     * jnp.sin(j * argument)
    return foo

fourier_sum_orama = vmap(vmap(fourier_sum, in_axes=(0, None), out_axes=1), \
    in_axes=(0, None), out_axes=0)

def internal_get_lnqs(lnq_pars, L, xs, lnAs, D):
    """
    sums of sines and cosines
    """
    xs_dilute = xs + (D * jnp.abs(lnAs[-1,:]))
    tmp = jnp.swapaxes(lnq_pars, 1, 2)
    return fourier_sum_orama(tmp, xs_dilute / L)

def get_lnqs_for_xs(lnq_pars, L, xs):
    """
    sums of sines and cosines
    """
    tmp = jnp.swapaxes(lnq_pars, 1, 2)
    return fourier_sum_orama(tmp, xs / L)

def all_stars_fk(fixed, fit, k):
    """
    ## inputs
    - `fixed` class
    - `fit` class
    - `k`: int

    ## outputs
    shape `(M, N)` fk

    """
    lnqs = get_lnqs(fixed, fit)
    denom = np.sum(np.exp(fit.lnAs[:-1,:,None])*np.exp(lnqs[:,:,:]),axis=0)
    num = np.exp(fit.lnAs[k,:,None])*np.exp(lnqs[k,:,:])

    return num / denom

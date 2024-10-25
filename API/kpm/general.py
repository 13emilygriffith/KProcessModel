from ._globals import _LN10

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import vmap
import numpy as np

def get_lnqs(fixed, fit):
    """
    Calculate lnq values from lnq_pars

    Inputs
    ------
    - `fixed`: KPM `fixed_params` class
    - `fit`: KPM `fit_params` class

    Outputs
    -------
    - shape `(K, N, M)` `lnqs` for all metallicities in `fixed.xs`
    """
    return internal_get_lnqs(fit.lnq_pars, fixed.L, fixed.xs, fit.lnAs, fixed.Delta)

def all_stars_KPM(fixed, fit):
    """
    Calculate all star abundances from fit parameters

    Inputs
    ------
    - `fixed`: KPM fixed_params class
    - `fit`: KPM fit_params class


    Outputs
    -------
    - shape `(M, N)` log_10 predicted abundances

    Comments
    --------
    - Note the `ln10`
    """
    return (logsumexp(fit.lnAs[:, :, None]
                     + get_lnqs(fixed, fit), axis=0) / _LN10) - (fixed.Delta)

def fourier_sum(amps, argument):
    # Used in internal_get_lnqs
    foo = amps[0] * jnp.ones_like(argument)
    for j in range(1, (len(amps) - 1) // 2 + 1):
        #print('fourier_sum', j)
        foo += amps[2*j - 1] * jnp.cos(j * argument) \
             + amps[2*j]     * jnp.sin(j * argument)
    return foo

# Used in internal_get_lnqs
fourier_sum_orama = vmap(vmap(fourier_sum, in_axes=(0, None), out_axes=1), \
    in_axes=(0, None), out_axes=0)

def internal_get_lnqs(lnq_pars, L, xs, lnAs, Delta):
    """
    Calculate lnq values from lnq_pars

    Inputs
    ------
    - `lnq_pars`: shape `(K,)` natural-logarithmic processes
    - `L`: float length of metallicity space
    - `xs`: array of metalicity values
    - `lnAs`: `lnAs`: shape `(K, )` natural-logarithmic amplitudes
    - `Delta`: float dillution coefficient

    Outputs
    -------
    - shape `(K, N, M)` `lnqs` for all metallicities in `xs`

    Comments
    -------_
    - can remove `lnAs` from input
    """

    xs_dilute = xs + Delta
    tmp = jnp.swapaxes(lnq_pars, 1, 2)
    return fourier_sum_orama(tmp, xs_dilute / L)

def get_lnqs_for_xs(lnq_pars, L, xs):
    """
    Calcualte lnq values along arbitrary xs

    Inputs
    ------
    - `lnq_pars`: shape `(K,)` natural-logarithmic processes
    - `L`: float length of metallicity space
    - `xs`: array of metalicity values

    Outputs
    -------
    shape `(K, len(xs), M)` `lnqs` for all metallicities in `xs`

    Comments
    --------
    - should add dilution here
    """
    tmp = jnp.swapaxes(lnq_pars, 1, 2)
    return fourier_sum_orama(tmp, xs / L)

def all_stars_fk(fixed, fit, k):
    """
    Calculate fractional contribution from the kth processes

    Inputs
    ------
    - `fixed`: KPM fixed_params class
    - `fit`: KPM `fit_params` class
    - `k`: int process number

    Outputs
    -------
    - shape `(M, N)` fk

    """
    lnqs = get_lnqs(fixed, fit)
    denom = np.sum(np.exp(fit.lnAs[:,:,None])*np.exp(lnqs[:,:,:]),axis=0)
    num = np.exp(fit.lnAs[k,:,None])*np.exp(lnqs[k,:,:])

    return num / denom

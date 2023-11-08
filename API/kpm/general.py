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
    foo = amps[0] * jnp.ones_like(argument)
    for j in range(1, (len(amps) - 1) // 2 + 1):
        #print('fourier_sum', j)
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

def all_stars_fcc(fixed, fit):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, Nbin, M)` natural-logarithmic processes
    - `proc`: str (CC, IA, AGB, or fourth)

    ## outputs
    shape `(M, )` fcc

    """
    lnqs = get_lnqs(fixed, fit)
    Acc_qcc = np.exp(fit.lnAs[0,:,None])*np.exp(lnqs[0,:,:])
    AIa_qIa = np.exp(fit.lnAs[1,:,None])*np.exp(lnqs[1,:,:])
    denom = Acc_qcc + AIa_qIa
    # if fixed.K==4:
    #   Aagb_qagb = np.exp(lnAs[2,:,None])*np.exp(lnqs[2,:,:])
    #   A4_q4 = np.exp(lnAs[3,:,None])*np.exp(lnqs[3,:,:])
    #   denom = Acc_qcc + AIa_qIa + Aagb_qagb + A4_q4

    i = np.where(fixed.processes=='CC')[0]
    num = np.exp(fit.lnAs[i,:,None])*np.exp(lnqs[i,:,:])

    return num / denom

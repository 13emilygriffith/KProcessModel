import numpy as np 
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jaxopt

from ._globals import _LN10
from .general import internal_get_lnqs

def one_star_KPM(lnAs, lnqs, Delta):
    """
    ## inputs
    - `lnAs`: shape `(K+1)` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, M)` natural-logarithmic processes
    - `D`: binary (1 = dilution on)

    ## outputs
    shape `(M, )` log_10 abundances

    ## comments
    - Note the `ln10`. 
    """
    return (logsumexp(lnAs[:, None] + lnqs, axis=0) / _LN10) - (Delta)

def one_star_chi(lnAs, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, Delta):
    """
    ## inputs
    - `lnAs`: shape `(K, )` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, M)` natural-logarithmic processes
    - `alldata`: shape `(M, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(M, )` inverse errors on the data
    - `sqrt_Lambda`: shape `(K, )` regularization strength on As

    ## outputs
    chi for this one star
    """

    return jnp.concatenate([sqrt_ivars * (alldata - one_star_KPM(lnAs, lnqs, Delta)),
                            sqrt_Lambda_A * jnp.exp(lnAs[:])])

def one_star_A_step(lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, Delta, init):
    """
    ## inputs
    - `lnqs`: shape `(K, M)` natural-logarithmic processes 
    - `alldata`: shape `(M, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(M, )` inverse errors on the data
    - `sqrt_Lambda`: shape `(K, )` regularization
    - `init`: shape `(K,)` initial guess for the A vector

    ## outputs
    shape `(K,)` best-fit natural-logarithmic amplitudes

    ## bugs
    - Doesn't check the output of the optimizer AT ALL.
    - Check out the crazy `maxiter` input!
    """
    solver = jaxopt.GaussNewton(residual_fun=one_star_chi, maxiter=4)
    lnAs_init = init.copy()
    chi2_init = np.sum(one_star_chi(lnAs_init, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, Delta) ** 2)
    res = solver.run(lnAs_init, lnqs=lnqs, alldata=alldata, sqrt_ivars=sqrt_ivars,
                     sqrt_Lambda_A=sqrt_Lambda_A, Delta=Delta)
    chi2_res = np.sum(one_star_chi(res.params, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, Delta) ** 2)
    return res.params, chi2_init - chi2_res

def one_element_KPM(lnqs, lnAs, Delta):
    """
    ## inputs
    - `lnqs`: shape `(K)` natural-logarithmic process elements
    - `lnAs`: shape `(K+1, N)` natural-logarithmic amplitudes
    - `D`: binary (1 = dilution on)

    ## outputs
    shape `(N, )` log_10 abundances

    ## comments
    - Note the `ln10`.
    """

    return (logsumexp(lnqs + lnAs[:,:], axis=0) / _LN10) - Delta

def one_element_chi(lnq_pars, lnAs, alldata, sqrt_ivars, L, xs, sqrt_Lambdas, q0s, Delta):
    """
    ## inputs
    - `lnqs`: shape `(K, J)` natural-logarithmic process vectors
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `alldata`: shape `(N, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(N, )` inverse variances on the data
    - `L`: xrange
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - `sqrt_Lambdas`: shape `(K, Nbin)` list of regularization amplitudes
    - `q0s`: shape `(K, J)` 

    ## outputs
    chi for this one star (weighted residual)
    """
    lnqs = internal_get_lnqs(lnq_pars[:, :, None], L, xs, lnAs, Delta)[:, :, 0]
    return jnp.concatenate([sqrt_ivars * (alldata - one_element_KPM(lnqs, lnAs, Delta)),
                            jnp.ravel(sqrt_Lambdas * (jnp.exp(lnq_pars) - q0s))])

def one_element_q_step(lnAs, alldata, sqrt_ivars, L, xs, sqrt_Lambdas, q0s,
                       fixed, Delta, init):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `alldata`: shape `(N, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(N, )` inverse errors on the data
    - `L`: xrange
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - ... 

    ## outputs
    shape `(K, J)` best-fit natural-logarithmic process elements

    ## bugs
    - Uses the `fixed` input incredibly stupidly, because Hogg SUX.
    - Doesn't check the output of the optimizer AT ALL.
    - Check out the crazy `maxiter` input!
    """
    solver = jaxopt.GaussNewton(residual_fun=one_element_chi, maxiter=4)
    lnq_pars_init = init.copy()
    chi2_init = np.sum(one_element_chi(lnq_pars_init, lnAs, alldata, sqrt_ivars, 
                       L, xs, sqrt_Lambdas, q0s, Delta) ** 2)
    res = solver.run(lnq_pars_init, lnAs=lnAs, alldata=alldata, sqrt_ivars=sqrt_ivars,
                     L=L, xs=xs,
                     sqrt_Lambdas=sqrt_Lambdas, q0s=q0s, Delta=Delta)
    chi2_res = np.sum(one_element_chi(res.params, lnAs, alldata, sqrt_ivars, 
                      L, xs, sqrt_Lambdas, q0s, Delta) ** 2)
    return jnp.where(fixed, lnq_pars_init, res.params), chi2_init - chi2_res
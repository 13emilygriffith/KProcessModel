import numpy as np 
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jaxopt

from ._globals import _LN10
from .general import internal_get_lnqs as get_lnqs

def one_star_KPM(lnAs, lnqs):
    """
    ## inputs
    - `lnAs`: shape `(K,)` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, M)` natural-logarithmic processes

    ## outputs
    shape `(M, )` log_10 abundances

    ## comments
    - Note the `ln10`.
    """
    return logsumexp(lnAs[:, None] + lnqs, axis=0) / _LN10

def one_star_chi(lnAs, lnqs, data, sqrt_ivars, sqrt_Lambda):
    """
    ## inputs
    - `lnAs`: shape `(K, )` natural-logarithmic amplitudes
    - `lnqs`: shape `(K, M)` natural-logarithmic processes
    - `data`: shape `(M, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(M, )` inverse errors on the data
    - `sqrt_Lambda`: shape `(K, )` regularization strength on As

    ## outputs
    chi for this one star
    """
    return jnp.concatenate([sqrt_ivars * (data - one_star_KPM(lnAs, lnqs)),
                            sqrt_Lambda * jnp.exp(lnAs)])

def one_star_A_step(lnqs, data, sqrt_ivars, sqrt_Lambda, init):
    """
    ## inputs
    - `lnqs`: shape `(K, M)` natural-logarithmic processes
    - `data`: shape `(M, )` log_10 abundance measurements
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
    chi2_init = np.sum(one_star_chi(lnAs_init, lnqs, data, sqrt_ivars, sqrt_Lambda) ** 2)
    res = solver.run(lnAs_init, lnqs=lnqs, data=data, sqrt_ivars=sqrt_ivars,
                     sqrt_Lambda=sqrt_Lambda)
    chi2_res = np.sum(one_star_chi(res.params, lnqs, data, sqrt_ivars, sqrt_Lambda) ** 2)
    return res.params, chi2_init - chi2_res

def one_element_KPM(lnqs, lnAs):
    """
    ## inputs
    - `lnqs`: shape `(K, N)` natural-logarithmic process elements
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes

    ## outputs
    shape `(N, )` log_10 abundances

    ## comments
    - Note the `ln10`.
    """
    return logsumexp(lnqs + lnAs, axis=0) / _LN10

def one_element_chi(lnqs, lnAs, data, sqrt_ivars, knot_xs, xs, sqrt_Lambdas, q0s):
    """
    ## inputs
    - `lnqs`: shape `(K, Nknot)` natural-logarithmic process vectors
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `data`: shape `(N, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(N, )` inverse variances on the data
    - `knot_xs`: shape `(Nknot, )` metallicity bin "centers"
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - `sqrt_Lambdas`: shape `(K, Nbin)` list of regularization amplitudes
    - `q0s`: shape `(K, Nknot)` 

    ## outputs
    chi for this one star (weighted residual)
    """
    K, Nknot = knot_xs.shape
    interp_lnqs = get_lnqs(K, lnqs[:, :, None], knot_xs, xs)[:, :, 0]
    return jnp.concatenate([sqrt_ivars * (data - one_element_KPM(interp_lnqs, lnAs)),
                            jnp.ravel(sqrt_Lambdas * (jnp.exp(lnqs) - q0s))])

def one_element_q_step(lnAs, data, sqrt_ivars, knot_xs, xs, sqrt_Lambdas, q0s,
                       fixed, init):
    """
    ## inputs
    - `lnAs`: shape `(K, N)` natural-logarithmic amplitudes
    - `data`: shape `(N, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(N, )` inverse errors on the data
    - `knot_xs`: shape `(Nknot, )` metallicity bin centers
    - `xs` : shape `(N, )` metallicities to use with `metallicities`
    - ... 

    ## outputs
    shape `(K, Nknot)` best-fit natural-logarithmic process elements

    ## bugs
    - Uses the `fixed` input incredibly stupidly, because Hogg SUX.
    - Doesn't check the output of the optimizer AT ALL.
    - Check out the crazy `maxiter` input!
    """
    solver = jaxopt.GaussNewton(residual_fun=one_element_chi, maxiter=4)
    lnqs_init = init.copy()
    chi2_init = np.sum(one_element_chi(lnqs_init, lnAs, data, sqrt_ivars, 
                       knot_xs, xs, sqrt_Lambdas, q0s) ** 2)
    res = solver.run(lnqs_init, lnAs=lnAs, data=data, sqrt_ivars=sqrt_ivars,
                     knot_xs=knot_xs, xs=xs,
                     sqrt_Lambdas=sqrt_Lambdas, q0s=q0s)
    chi2_res = np.sum(one_element_chi(res.params, lnAs, data, sqrt_ivars, 
                      knot_xs, xs, sqrt_Lambdas, q0s) ** 2)
    return jnp.where(fixed, lnqs_init, res.params), chi2_init - chi2_res
import numpy as np 
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jaxopt

from ._globals import _LN10
from .general import internal_get_lnqs

def one_star_KPM(lnAs, lnqs, D):
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
    return (logsumexp(lnAs[:-1, None] + lnqs, axis=0) / _LN10) - (D * lnAs[-1,None])

def one_star_chi(lnAs, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, D, sqrt_Lambda_D):
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

    return jnp.concatenate([sqrt_ivars * (alldata - one_star_KPM(lnAs, lnqs, D)),
                            sqrt_Lambda_A * jnp.exp(lnAs[:-1]), 
                            jnp.array(sqrt_Lambda_D * D * lnAs[-1]).reshape(1)])

    # return jnp.concatenate([sqrt_ivars * (alldata - one_star_KPM(lnAs, lnqs, D)),
    #                         sqrt_Lambda * jnp.exp(lnAs[:-1])]) # pre D code

def one_star_A_step(lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, D, sqrt_Lambda_D, init):
    """
    ## inputs
    - `lnqs`: shape `(K, M)` natural-logarithmic processes 
    - `alldata`: shape `(M, )` log_10 abundance measurements
    - `sqrt_ivars`: shape `(M, )` inverse errors on the data
    - `sqrt_Lambda`: shape `(K, )` regularization
    - `D`:
    - `init`: shape `(K,)` initial guess for the A vector

    ## outputs
    shape `(K,)` best-fit natural-logarithmic amplitudes

    ## bugs
    - Doesn't check the output of the optimizer AT ALL.
    - Check out the crazy `maxiter` input!
    """
    solver = jaxopt.GaussNewton(residual_fun=one_star_chi, maxiter=4)
    lnAs_init = init.copy()
    chi2_init = np.sum(one_star_chi(lnAs_init, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, D, sqrt_Lambda_D) ** 2)
    res = solver.run(lnAs_init, lnqs=lnqs, alldata=alldata, sqrt_ivars=sqrt_ivars,
                     sqrt_Lambda_A=sqrt_Lambda_A, D=D, sqrt_Lambda_D=sqrt_Lambda_D)
    chi2_res = np.sum(one_star_chi(res.params, lnqs, alldata, sqrt_ivars, sqrt_Lambda_A, D, sqrt_Lambda_D) ** 2)
    return res.params, chi2_init - chi2_res

def one_element_KPM(lnqs, lnAs, D):
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

    return (logsumexp(lnqs + lnAs[:-1,:], axis=0) / _LN10) - (D * lnAs[-1,:])

def one_element_chi(lnq_pars, lnAs, alldata, sqrt_ivars, L, xs, sqrt_Lambdas, q0s, D):
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
    lnqs = internal_get_lnqs(lnq_pars[:, :, None], L, xs, lnAs, D)[:, :, 0]
    return jnp.concatenate([sqrt_ivars * (alldata - one_element_KPM(lnqs, lnAs, D)),
                            jnp.ravel(sqrt_Lambdas * (jnp.exp(lnq_pars) - q0s))])

def one_element_q_step(lnAs, alldata, sqrt_ivars, L, xs, sqrt_Lambdas, q0s,
                       fixed, D, init):
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
                       L, xs, sqrt_Lambdas, q0s, D) ** 2)
    res = solver.run(lnq_pars_init, lnAs=lnAs, alldata=alldata, sqrt_ivars=sqrt_ivars,
                     L=L, xs=xs,
                     sqrt_Lambdas=sqrt_Lambdas, q0s=q0s, D=D)
    chi2_res = np.sum(one_element_chi(res.params, lnAs, alldata, sqrt_ivars, 
                      L, xs, sqrt_Lambdas, q0s, D) ** 2)
    return jnp.where(fixed, lnq_pars_init, res.params), chi2_init - chi2_res
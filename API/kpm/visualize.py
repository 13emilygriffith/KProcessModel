
import numpy as np
import jax.numpy as jnp
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .general import internal_get_lnqs, all_stars_KPM, all_stars_fk
from ._globals import _RNG2

__all__ = ["plot_qs", "plot_As", "plot_model_abundances", "plot_fk", "plot_chi2"]


def plot_qs(data, fixed, fit):
    """
    # Bugs:
    - Assumes a rigid structure for the processes?
    - Need to set size and number of subplots based on number of elements
    """
    MgH = np.linspace(fixed.xlim[0]-0.2, fixed.xlim[1]+0.2, 300) # plotting xs
    new_qs = np.exp(internal_get_lnqs(fit.lnq_pars, fixed.L, MgH, np.zeros((fixed.K, len(MgH))), fixed.Delta)) # interp to plotting xs

    plt.figure(figsize=(15, 2.5*data.M//4+1))
    for i in range(data.M):
        plt.subplot(data.M//4+1,4,i+1)

        for k in range(fixed.K):
            q = new_qs[k,:,i]
            plt.plot(MgH, q, '-', alpha=0.9, label='q_'+str(k))
        plt.gca().set_prop_cycle(None)
       

        plt.xlabel('[Mg/H]')
        plt.xlim(np.min(fixed.xs), np.max(fixed.xs))
        plt.ylabel('q '+data.elements[i])
        plt.ylim(-0.15, 1.5)

        if i==0:
            plt.legend(ncol=1, fontsize=10)
        #plt.ylim(-0.1,1.1)
    plt.tight_layout()

def plot_As(fit):
	plt.figure(figsize=(3,3))

	Acc = np.exp(fit.lnAs[0,:])
	AIa = np.exp(fit.lnAs[1,:])

	plt.hist2d(Acc, AIa/Acc, bins=100, range=[[0,3],[-0.05,1.5]],
	           norm=LogNorm(), cmap='PuOr_r', rasterized=True)
	plt.xlabel(r'$A^{\rm CC}_{i}$', fontsize=12)
	plt.ylabel(r'$A^{\rm Ia}_{i}/A^{\rm CC}_{i}$', fontsize=12)


def plot_model_abundances(data, fixed, fit, noise=False):
    """
    ## bugs:
    - Relies on lots of global variables.
    """
    MgHmin = np.min(fixed.xs) - 0.1

    synthdata = all_stars_KPM(fixed, fit)
    synthnoise = 0.
    noisestr = ""
    if noise:
        synthnoise = _RNG2.normal(size=synthdata.shape) / data.sqrt_allivars
        noisestr = " + noise"
    fig, axes = plt.subplots(len(data.elements) - 1, 3, figsize=(12,3 * (len(data.elements) - 1)))

    for j in range(len(data.elements) - 1):
        ax = axes[j, 0]
        mask = np.where(data.allivars[:,j+1]!=0)[0]
        ax.hist2d(data.alldata[:,0][mask], data.alldata[:,j+1][mask] - data.alldata[:,0][mask],
                  cmap='magma', bins=100, range=[[MgHmin,0.6],[-0.7,0.4]], norm=LogNorm())
        ax.set_xlabel('[Mg/H]')
        ax.set_ylabel('[{}/Mg]'.format(data.elements[j+1]))
        ax.set_ylim(-0.7,0.4)
        if j == 0:
            ax.set_title('observed')

        ax = axes[j, 1]
        sata = synthdata + synthnoise
        ax.hist2d(sata[:,0], sata[:,j+1] - sata[:,0],
                  cmap='magma', bins=100, range=[[MgHmin,0.6],[-0.7,0.4]], norm=LogNorm())
        ax.set_xlabel('[Mg/H]')
        ax.set_ylabel('[{}/Mg]'.format(data.elements[j+1]))
        ax.set_ylim(-0.7,0.4)
        if j == 0:
            ax.set_title('predicted' + noisestr)

        ax = axes[j, 2]
        ax.hist2d(data.sqrt_allivars[:, 0][mask] * (data.alldata[:, 0][mask] - synthdata[:, 0][mask]),
                  data.sqrt_allivars[:, j+1][mask] * (data.alldata[:, j+1][mask] - synthdata[:, j+1][mask]),
                cmap='magma', bins=100, range=[[-10, 10], [-10, 10]], norm=LogNorm())
        ax.set_xlabel('[Mg/H] chi')
        ax.set_ylabel('[{}/H] chi'.format(data.elements[j+1]))
        if j == 0:
            ax.set_title('dimensionless residual')

    plt.tight_layout()

def plot_fk(data, fixed, fit):

    plt.figure(figsize=(2.5*(fixed.K), 2.5*data.M))
    
    for k in range(fixed.K):
        fk = all_stars_fk(fixed, fit ,k)
        for i in range(data.M):
            plt.subplot(data.M,fixed.K,(fixed.K*i)+k+1)
            plt.hist2d(fixed.xs, fk[:,i], norm=LogNorm(), bins=100, range=[[-2,0.4],[-0.1,1.1]],
                cmap='magma')

            plt.xlabel('[Mg/H]')
            plt.ylabel('f_ '+str(k+1)+' '+data.elements[i])
            plt.ylim(-0.05,1.05)
            if (i==0): plt.title('k='+str(k))
    plt.tight_layout()
    plt.show()


def plot_chi2(data, fixed_list, fit_list, color_list, label_list):

    f, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(10,3))

    for fit, fix, color, label in zip(fit_list,fixed_list, color_list, label_list):

        synthdata = all_stars_KPM(fix, fit)

        chi2_stars = np.sum(((data.alldata - synthdata)**2) * data.allivars, axis=1)
        chi2_elems = np.sum(((data.alldata - synthdata)**2) * data.allivars, axis=0)

        count, bins_count = np.histogram(chi2_stars, bins=1000)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        ax0.plot(cdf, np.log10(bins_count[1:]), label=label, lw=1, color=color)

        ax1.plot(fix.elements, chi2_elems,
            'o-', color=color, lw=1, label=label)

    ax0.set_xlabel(r'N$_{*}$/N$_{\rm total}$', fontsize=12)
    ax0.set_ylabel(r'log $\chi^2$', fontsize=12)
    ax0.set_ylim(0.75,2.5)

    ax1.set_ylabel(r'$\chi^2$ per element', fontsize=12)
    plt.legend(loc='best', fontsize=11)

    plt.tight_layout()




    
    






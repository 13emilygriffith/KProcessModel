
import numpy as np
import jax.numpy as jnp
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .general import internal_get_lnqs, all_stars_KPM
from ._globals import _RNG2

__all__ = ["plot_qs", "plot_As", "plot_model_abundances"]


def plot_qs(data, fixed, fit):
    """
    # Bugs:
    - Assumes a rigid structure for the processes?
    - Need to set size and number of subplots based on number of elements
    """
    MgH = np.linspace(np.min(fixed.knot_xs[0]), np.max(fixed.knot_xs[0]), 300) # plotting xs
    new_qs = np.exp(internal_get_lnqs(fixed.K, fit.lnqs, fixed.knot_xs, MgH)) # interp to plotting xs
    # w22_MgH = w22_metallicities
    # w22_qs = np.exp(w22_lnqs)

    plt.figure(figsize=(10,10))
    for i in range(data.M):
        plt.subplot(4,4,i+1)
        new_qcc = new_qs[0,:,i]
        new_qIa = new_qs[1,:,i]
        # w22_qcc = w22_qs[0,:,i]
        # w22_qIa = w22_qs[1,:,i]

        # plt.plot(w22_MgH, w22_qcc, 'b-', lw=4, alpha=0.25, label='qcc W22')
        # plt.plot(w22_MgH, w22_qIa, 'r-', lw=4, alpha=0.25, label='qIa W22')

        plt.plot(MgH, new_qcc, 'b-', alpha=0.9, label='qcc')
        plt.plot(MgH, new_qIa, 'r-', alpha=0.9, label='qIa')

        if fixed.K==4:
          new_qagb = new_qs[2,:,i]
          new_q4 = new_qs[3,:,i]

          plt.plot(MgH, new_qagb, 'm-', alpha=0.9, label='qagb')
          plt.plot(MgH, new_q4, 'c-', alpha=0.9, label='q4')

        plt.xlabel('[Mg/H]')
        plt.xlim(np.min(fixed.knot_xs), 0.55) #np.max(knot_xs))
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
        ax.hist2d(data.alldata[:,0], data.alldata[:,j+1] - data.alldata[:,0],
                  cmap='magma', bins=100, range=[[MgHmin,0.6],[-0.5,0.2]], norm=LogNorm())
        ax.set_xlabel('[Mg/H]')
        ax.set_ylabel('[{}/Mg]'.format(data.elements[j+1]))
        ax.set_ylim(-0.5,0.2)
        if j == 0:
            ax.set_title('observed')

        ax = axes[j, 1]
        sata = synthdata + synthnoise
        ax.hist2d(sata[:,0], sata[:,j+1] - sata[:,0],
                  cmap='magma', bins=100, range=[[MgHmin,0.6],[-0.5,0.2]], norm=LogNorm())
        ax.set_xlabel('[Mg/H]')
        ax.set_ylabel('[{}/Mg]'.format(data.elements[j+1]))
        ax.set_ylim(-0.5,0.2)
        if j == 0:
            ax.set_title('predicted' + noisestr)

        ax = axes[j, 2]
        ax.hist2d(data.sqrt_allivars[:, 0] * (data.alldata[:, 0] - synthdata[:, 0]),
                  data.sqrt_allivars[:, j+1] * (data.alldata[:, j+1] - synthdata[:, j+1]),
                cmap='magma', bins=100, range=[[-10, 10], [-10, 10]], norm=LogNorm())
        ax.set_xlabel('[Mg/H] chi')
        ax.set_ylabel('[{}/H] chi'.format(data.elements[j+1]))
        if j == 0:
            ax.set_title('dimensionless residual')

    plt.tight_layout()
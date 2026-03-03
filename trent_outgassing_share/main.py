from matplotlib import pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import os
THISFILE = os.path.dirname(os.path.abspath(__file__))

from models import AdiabatClimateEquilibrium, EvoAtmosphereRobust

def plot(P, T, mix, ylim, filename, P_ref=1e3):
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1,1,figsize=[5,4])

    ind = np.argmin(np.abs(P - P_ref))
    species_layer = []
    mix_layer = np.empty(len(mix))
    for i,sp in enumerate(mix):
        species_layer.append(sp)
        mix_layer[i] = mix[sp][ind]
    inds = np.argsort(mix_layer)[::-1]
    
    for i in inds[:10]:
        sp = species_layer[i]
        ax.plot(mix[sp], P/1e6, lw=2, label=sp)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-10,1.2)
    ax.set_ylim(*ylim)
    ax.legend(ncol=2,bbox_to_anchor=(1.02, 1.02), loc='upper left')

    ax1 = ax.twiny()
    ax1.plot(T, P/1e6, c='k', lw=2, ls='--', label='Temp.')
    ax1.set_xlabel('Temperature (K)')
    ax1.legend(ncol=1,bbox_to_anchor=(1.02, .2), loc='upper left')

    plt.savefig(filename, dpi=150, bbox_inches='tight')

def run(P_surf, mix, verbose=False):

    c = CLIMATE_MODEL

    # Solve
    solve_history = c.solve(P_surf, mix, verbose=verbose) # enable all printing

    # Return P, T, mix
    P, T, mix = c.return_atmosphere()

    return P, T, mix

def example():

    P_surf = 1.0e6 # dynes/cm^2
    # Some composition
    mix = {
        'CO2': 0.5,
        'H2O': 0.2,
        'SO2': 0.3,
    }
    P, T, mix = run(P_surf, mix, verbose=True)

    plot(P, T, mix, ylim=(P[0]/1e6,P[-1]/1e6), filename='figures/test.pdf', P_ref=1e3)

# Initialize global
CLIMATE_MODEL = AdiabatClimateEquilibrium(
    species_file=os.path.join(THISFILE, 'input/species_climate.yaml'),
    settings_file=os.path.join(THISFILE, 'input/settings_climate.yaml'),
    flux_file=os.path.join(THISFILE, 'input/gj176_scaled_to_l9859b.txt'),
)

if __name__ == "__main__":
    _ = threadpool_limits(limits=4) # set number of threads
    example()

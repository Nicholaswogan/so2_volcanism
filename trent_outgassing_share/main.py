from matplotlib import pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from models import AdiabatClimateEquilibrium, EvoAtmosphereRobust

def plot(P, T, mix, ylim, filename, species):
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1,1,figsize=[5,4])

    for i,sp in enumerate(species):
        ls = '-'
        if i > 9:
            ls = ':'
        ax.plot(mix[sp], P/1e6, lw=2, label=species[i], ls=ls)
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

# def plot_history

def main():

    # Columns of each gas in moles/cm^2
    N_i = {
        'CO2': 100.0,
        'H2O': 1.0,
        'SO2': 10.0,
    }

    # Initialize
    c = AdiabatClimateEquilibrium(
        species_file='input/species_climate.yaml',
        settings_file='input/settings_climate.yaml',
        flux_file='input/gj176_scaled_to_l9859d.txt',
        data_dir=None
    )
    c.verbose = True # enable all printing

    # Solve
    solve_history = c.solve(N_i, verbose=True) # enable all printing

    # Plot end state
    P, T, mix = c.return_atmosphere()
    species_plot = c.species_names
    plot(P, T, mix, ylim=(c.P_surf/1e6,1e-7), filename='figures/PTX_climate.png', species=species_plot)

    # Now run Photochemistry
    pc = EvoAtmosphereRobust(
        mechanism_file='input/zahnle_HOCS.yaml',
        settings_file='input/settings.yaml',
        flux_file='input/gj176_scaled_to_l9859d.txt'
    )
    pc.var.verbose = 1 # enable all printing
    pc.rdat.verbose = True

    # Initialize to climate state
    Kzz = np.ones_like(P)*1e6
    pc.initialize_to_PT(P, T, Kzz, mix)

    # Solve
    pc.find_steady_state_robust()

    # Plot
    P1, T1, mix1 = pc.return_atmosphere()
    plot(P1, T1, mix1, ylim=(c.P_surf/1e6,1e-7), filename='figures/PTX_photochem.png', species=species_plot)

if __name__ == "__main__":
    _ = threadpool_limits(limits=4) # set number of threads
    main()

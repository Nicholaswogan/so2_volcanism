import numpy as np
from tempfile import NamedTemporaryFile
from copy import deepcopy
import numba as nb
from numba import types
from scipy import integrate
from scipy import constants as const
from scipy import interpolate
import yaml

from fixedpoint import RobustFixedPointSolver

from photochem import EvoAtmosphere, PhotoException
from photochem.clima import AdiabatClimate, ClimaException
from photochem.equilibrate import ChemEquiAnalysis

### Modified version of Climate model ###

class AdiabatClimateEquilibrium(AdiabatClimate):
    """Couples surface thermochemical equilibrium with a 1-D climate solve.

    The class wraps :class:`photochem.clima.AdiabatClimate` and adds a
    fixed-point closure in which:

    1. Surface composition is computed from thermochemical equilibrium at the
       current surface pressure-temperature state.
    2. The climate model is solved for radiative-convective equilibrium (RCE)
       using that surface composition.
    3. Surface temperature is iterated until these two steps are mutually
       consistent.
    """

    def __init__(self, species_file, settings_file, flux_file, data_dir=None):
        """Initialize the climate-equilibrium model.

        Parameters
        ----------
        species_file : str
            Path to climate species YAML file.
        settings_file : str
            Path to climate settings YAML file.
        flux_file : str
            Path to stellar flux file.
        data_dir : str, optional
            Optional path to photochem/clima data directory.
        """

        super().__init__(
            species_file, 
            settings_file, 
            flux_file,
            data_dir=data_dir
        )

        # Change defaults
        self.P_top = 1.0 # dynes/cm^2
        self.use_make_column_P_guess = False
        self.verbose = False

        # Save an equilibrium solver
        self.eqsolver = ChemEquiAnalysis(species_file)

        # Do some extra work to get species masses (g/mol)
        with open(species_file,'r') as f:
            species_dict = yaml.load(f, Loader=yaml.Loader)
        # Get atoms_masses
        atoms_masses = {}
        for val in species_dict['atoms']:
            atoms_masses[val['name']] = val['mass']
        # Get species masses (g/mol)
        self.species_composition = {}
        for sp in species_dict['species']:
            mass = 0.0
            for atom in sp['composition']:
                mass += sp['composition'][atom]*atoms_masses[atom]
            self.species_composition[sp['name']] = sp['composition']

    def equilibrate_columns(self, T_surf, N_i):
        """Compute surface partial pressures from bulk columns at fixed ``T_surf``.

        Parameters
        ----------
        T_surf : float
            Trial surface temperature in K.
        N_i : dict
            Mapping from species name to column inventory (mol/cm^2).

        Returns
        -------
        ndarray
            Surface partial pressures in dynes/cm^2 ordered as
            ``self.species_names``.

        Notes
        -----
        This routine (i) computes a consistent surface pressure from
        :meth:`make_column`, (ii) derives elemental abundances from ``N_i``,
        and (iii) solves thermochemical equilibrium at ``(P_surf, T_surf)``.
        """
        # T_surf is surface T in K
        # N_i is dict of each atmospheric species reservoir in moles/cm^2

        # Convert to an array 
        N_i_arr = np.ones(len(self.species_names))*1e-15
        for sp,val in N_i.items():
            ind = self.species_names.index(sp)
            N_i_arr[ind] = val
        N_i_arr = np.clip(N_i_arr, a_min=0.0, a_max=np.inf)

        # Use AdiabatClimate's own column solver to get the physically consistent
        # surface pressure for the current species set and T_surf.
        T_trop_prev = self.T_trop
        self.T_trop = T_surf - T_surf*1e-5
        try:
            self.make_column(T_surf, N_i_arr)
        finally:
            self.T_trop = T_trop_prev
        P_surf = self.P_surf # dynes/cm^2

        # Compute atomic composition
        atoms_res = np.ones(len(self.eqsolver.atoms_names))*1e-10
        for i,sp in enumerate(self.species_names):
            if N_i_arr[i] <= 0.0:
                continue
            if sp not in self.species_composition:
                continue
            for atom,stoich in self.species_composition[sp].items():
                if atom in self.eqsolver.atoms_names:
                    j = self.eqsolver.atoms_names.index(atom)
                    atoms_res[j] += N_i_arr[i]*stoich

        # Set atomic composition
        if np.sum(atoms_res) <= 0.0:
            raise ValueError('N_i does not contain atoms used by the equilibrium solver.')
        molfracs_atoms_sun = np.ones_like(self.eqsolver.molfracs_atoms_sun)*1e-10
        for i,atom in enumerate(self.eqsolver.atoms_names):
            molfracs_atoms_sun[i] = atoms_res[i] # atomic composition
        molfracs_atoms_sun /= np.sum(molfracs_atoms_sun) 
        self.eqsolver.molfracs_atoms_sun = molfracs_atoms_sun

        # Solve for equilibrium
        converged = self.eqsolver.solve_metallicity(P_surf, T_surf, metallicity=1.0)

        # Get mixing ratios
        f_i_surf = np.zeros(len(self.species_names))
        for j,sp in enumerate(self.eqsolver.gas_names):
            ind = self.species_names.index(sp)
            f_i_surf[ind] = self.eqsolver.molfracs_species_gas[j]
        f_i_surf = f_i_surf/np.sum(f_i_surf)

        # Partial pressures
        P_i_surf = f_i_surf*P_surf

        if not converged:
            P_i_surf *= np.nan

        return P_i_surf

    def RCE_robust(self, P_i, T_guess_mid=None, T_perturbs=None):
        """Attempt an RCE solve using a sequence of robust temperature guesses.

        Parameters
        ----------
        P_i : ndarray
            Surface partial pressures (dynes/cm^2), ordered as
            ``self.species_names``.
        T_guess_mid : float, optional
            Central value for surface-temperature perturbation guesses. If
            ``None``, uses ``1.5 * self.rad.equilibrium_temperature(0.0)``.
        T_perturbs : ndarray, optional
            Additive perturbations (K) applied to ``T_guess_mid``.

        Returns
        -------
        bool
            ``True`` if any attempted RCE solve converges, else ``False``.
        """

        if T_guess_mid is None:
            T_guess_mid = self.rad.equilibrium_temperature(0.0)*1.5
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 50.0, -50.0, 100.0, -100.0, 150.0, 800.0, 600.0, 400.0, 300.0, 200.0])

        # Try a bunch of isothermal atmospheres.
        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            T_guess = np.ones(self.T.shape[0])*T_surf_guess
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess)
                if converged:
                    break
            except ClimaException:
                converged = False

        return converged

    def _g(self, T_surf, N_i):
        """Evaluate the fixed-point map for the coupled climate-chemistry solve.

        Parameters
        ----------
        T_surf : float
            Trial surface temperature in K.
        N_i : dict
            Mapping from species name to column inventory (mol/cm^2).

        Returns
        -------
        float
            Updated surface temperature from the climate model when RCE
            converges; ``np.nan`` otherwise.
        """
        P_i = self.equilibrate_columns(T_surf, N_i)
        converged = self.RCE_robust(P_i)
        if not converged:
            return np.nan
        return self.T_surf
    
    def solve(self, N_i, *, tol=1.0, **kwargs):
        """Solve for a self-consistent surface temperature.

        Parameters
        ----------
        N_i : dict
            Mapping from species name to column inventory (mol/cm^2).
        tol : float, optional
            Residual tolerance for the fixed-point solver (K for scalar solve).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`fixedpoint.RobustFixedPointSolver`.

        Returns
        -------
        SolveResult
            Output object from :class:`fixedpoint.RobustFixedPointSolver`.
        """

        def g(x):
            return np.array([self._g(x[0], N_i)])

        solver = RobustFixedPointSolver(
            g=g,
            x0=np.array([self.rad.equilibrium_temperature(0.0)*1.5]),
            tol=tol,
            **kwargs
        )
        result = solver.solve()

        return result
    
    def return_atmosphere(self):
        """Return the current climate state in array/dict form.

        Returns
        -------
        tuple
            ``(P, T, mix)`` where:

            - ``P`` is pressure in dynes/cm^2 including the surface point.
            - ``T`` is temperature in K including the surface point.
            - ``mix`` is a dict mapping species name to mixing-ratio profile.
        """
        f_i = np.concatenate((self.f_i_surf.reshape((1,len(self.f_i_surf))),self.f_i))
        P = np.append(self.P_surf, self.P)
        T = np.append(self.T_surf, self.T)

        mix = {}
        for i,sp in enumerate(self.species_names):
            mix[sp] = f_i[:,i]

        return P, T, mix

### Modified version of Photochemical model ###

class RobustData():
    
    def __init__(self):

        # Parameters for determining steady state
        self.atols = [1e-23, 1e-22, 1e-20, 1e-18]
        self.min_mix_reset = -1e-13
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 3 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.freq_update_atol = 10_000
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = True # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.max_time = None
        self.robust_stepper_initialized = None
        # Surface pressures
        self.Pi = None

class EvoAtmosphereRobust(EvoAtmosphere):
    """Photochemical model wrapper with robust initialization and stepping.

    This class extends :class:`photochem.EvoAtmosphere` with helpers to:

    - initialize from climate-model ``P-T-Kzz`` and composition profiles,
    - apply and restore surface-pressure boundary conditions,
    - integrate with reinitialization safeguards and adaptive tolerances,
    - save/restore full model state during robust solves.
    """

    def __init__(self, mechanism_file, settings_file, flux_file, data_dir=None):
        """Initialize the robust photochemical model.

        Parameters
        ----------
        mechanism_file : str
            Path to reaction-mechanism YAML file.
        settings_file : str
            Path to photochemical settings YAML file.
        flux_file : str
            Path to stellar flux file.
        data_dir : str, optional
            Optional photochem data directory.
        """

        with NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(ATMOSPHERE_INIT)
            f.flush()
            super().__init__(
                mechanism_file, 
                settings_file, 
                flux_file,
                f.name,
                data_dir
            )

        self.rdat = RobustData()

        # Values in photochem to adjust
        self.var.verbose = 0
        self.var.upwind_molec_diff = True
        self.var.autodiff = True
        self.var.atol = 1.0e-23
        self.var.equilibrium_time = 1e15
        self.var.conv_longdy = 1e-3

        # Model state
        self.max_time_state = None

        for i in range(len(self.var.cond_params)):
            self.var.cond_params[i].smooth_factor = 1
            self.var.cond_params[i].k_evap = 0

    def set_surface_pressures(self, Pi):
        """Set lower boundary pressures for selected species.

        Parameters
        ----------
        Pi : dict
            Mapping ``{species_name: partial_pressure_dyn_cm2}``.
        """
        
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

    def initialize_to_PT(self, P, T, Kzz, mix):
        """Initialize model state from target pressure-temperature-composition data.

        Parameters
        ----------
        P : ndarray
            Pressure profile in dynes/cm^2 (surface to top).
        T : ndarray
            Temperature profile in K on ``P``.
        Kzz : ndarray
            Eddy diffusion profile in cm^2/s on ``P``.
        mix : dict
            Mapping of species name to mixing-ratio profile on ``P``.
        """

        P, T, mix = deepcopy(P), deepcopy(T), deepcopy(mix)

        rdat = self.rdat

        # Ensure X sums to 1
        ftot = np.zeros(P.shape[0])
        for key in mix:
            ftot += mix[key]
        for key in mix:
            mix[key] = mix[key]/ftot

        # Compute mubar at all heights
        mu = {}
        for i,sp in enumerate(self.dat.species_names[:-2]):
            mu[sp] = self.dat.species_mass[i]
        mubar = np.zeros(P.shape[0])
        for key in mix:
            mubar += mix[key]*mu[key]

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P, T, mubar, self.dat.planet_radius, self.dat.planet_mass, rdat.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
        if P1.shape[0] != Kzz.shape[0]:
            Kzz1 = np.append(Kzz,Kzz[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz.copy()
            mix1 = mix

        rdat.log10P_interp = np.log10(P1.copy()[::-1])
        rdat.T_interp = T1.copy()[::-1]
        rdat.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        
        # extrapolate to 1e6 bars
        T_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.T_interp, bounds_error=False, fill_value='extrapolate')(12)
        edd_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.log10edd_interp, bounds_error=False, fill_value='extrapolate')(12)
        rdat.log10P_interp = np.append(rdat.log10P_interp, 12)
        rdat.T_interp = np.append(rdat.T_interp, T_tmp)
        rdat.log10edd_interp = np.append(rdat.log10edd_interp, edd_tmp)

        rdat.P_desired = P1.copy()
        rdat.T_desired = T1.copy()
        rdat.Kzz_desired = Kzz1.copy()

        # Calculate the photochemical grid
        ind_t = np.argmin(np.abs(P1 - rdat.TOA_pressure_avg))
        z_top = z1[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1, np.log10(P1))
        T_p = np.interp(z_p, z1, T1)
        Kzz_p = 10.0**np.interp(z_p, z1, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Update photochemical model grid
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        self.prep_atmosphere(self.wrk.usol)

    def initialize_to_PT_bcs(self, P, T, Kzz, mix, Pi):
        """Initialize from ``P-T-Kzz-mix`` and set surface-pressure BCs.

        Parameters
        ----------
        P : ndarray
            Pressure profile in dynes/cm^2.
        T : ndarray
            Temperature profile in K.
        Kzz : ndarray
            Eddy diffusion profile in cm^2/s.
        mix : dict
            Species mixing-ratio profiles.
        Pi : dict
            Surface partial-pressure boundary conditions in dynes/cm^2.
        """
        self.rdat.Pi = Pi
        self.set_surface_pressures(Pi)
        self.initialize_to_PT(P, T, Kzz, mix)

    def set_particle_radii(self, radii):
        """Set particle radii profiles for selected species.

        Parameters
        ----------
        radii : dict
            Mapping ``{species_name: radius_profile_cm}``.
        """
        particle_radius = self.var.particle_radius
        for key in radii:
            ind = self.dat.species_names.index(key)
            particle_radius[ind,:] = radii[key]
        self.var.particle_radius = particle_radius
        self.update_vertical_grid(TOA_alt=self.var.top_atmos)

    def initialize_robust_stepper(self, usol):
        """Initialize the robust integrator state.

        Parameters
        ----------
        usol : ndarray
            Number-density state array.
        """
        rdat = self.rdat  
        rdat.total_step_counter = 0
        rdat.nerrors = 0
        rdat.max_time = 0
        self.max_time_state = None
        self.initialize_stepper(usol)
        rdat.robust_stepper_initialized = True

    def robust_step(self):
        """Take one safeguarded integration step.

        Returns
        -------
        tuple
            ``(give_up, reached_steady_state)``.
        """

        rdat = self.rdat

        if not rdat.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                rdat.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Reset integrator if we get large magnitude negative numbers
            if not self.healthy_atmosphere():
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Update the max time achieved
            if self.wrk.tn > rdat.max_time:
                rdat.max_time = self.wrk.tn
                self.max_time_state = self.model_state_to_dict() # save the model state

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > rdat.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < rdat.max_dT_tol and max_dlog10edd < rdat.max_dlog10edd_tol and rdat.TOA_pressure_avg/3 < TOA_pressure < rdat.TOA_pressure_avg*3

            if condition1 and condition2:
                if rdat.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (rdat.total_step_counter % rdat.freq_update_atol):
                ind = int(rdat.total_step_counter/rdat.freq_update_atol)
                ind1 = ind - len(rdat.atols)*int(ind/len(rdat.atols))
                self.var.atol = rdat.atols[ind1]
                if rdat.verbose:
                    print('new atol = %.1e'%(self.var.atol))
                self.initialize_stepper(self.wrk.usol)
                break

            if not (self.wrk.nsteps % rdat.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid, if possible.
                try:
                    self.set_press_temp_edd(rdat.P_desired,rdat.T_desired,rdat.Kzz_desired,hydro_pressure=True)
                except PhotoException:
                    pass
                try:
                    self.update_vertical_grid(TOA_pressure=rdat.TOA_pressure_avg)
                except PhotoException:
                    pass
                self.initialize_stepper(self.wrk.usol)

            if rdat.total_step_counter > rdat.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % rdat.freq_print) and rdat.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Integrate until steady state or a stop condition is reached.

        Returns
        -------
        bool
            ``True`` if steady-state convergence is reached, else ``False``.
        """

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def healthy_atmosphere(self):
        """Check for unphysical negative mixing-ratio excursions.

        Returns
        -------
        bool
            ``True`` if atmosphere history satisfies the configured lower bound.
        """
        return np.min(self.wrk.mix_history[:,:,0]) > self.rdat.min_mix_reset
    
    def find_steady_state_robust(self):
        """Try multiple tolerance settings to recover steady-state convergence.

        Returns
        -------
        bool
            ``True`` if any robust attempt converges, else ``False``.
        """

        # Change some rdat settings
        self.rdat.freq_update_atol = 100_000
        self.rdat.max_total_step = 10_000

        # First just try to get to steady-state with standard atol
        self.var.atol = 1.0e-23
        converged = self.find_steady_state()
        if converged:
            return converged

        # Convergence did not happen. Save the max time state.
        max_time = self.rdat.max_time
        max_time_state = deepcopy(self.max_time_state)

        # Lets try a couple different atols.
        for atol in [1.0e-18, 1.0e-15]:
            # Lets initialize to max time state
            self.initialize_from_dict(max_time_state)
            # Do some smaller number of steps
            self.rdat.max_total_step = 5_000
            self.var.atol = atol # set the atol
            converged = self.find_steady_state() # Integrate
            if converged:
                # If converged then lets return
                return converged

            # No convergence. We re-save max time state
            if self.rdat.max_time > max_time:
                max_time = self.rdat.max_time
                max_time_state = deepcopy(self.max_time_state)

        # No convergence, we reinitialize to max time state and return
        self.initialize_from_dict(max_time_state)

        return converged
    
    def return_atmosphere(self):
        """Return current photochemical atmosphere fields.

        Returns
        -------
        tuple
            ``(P, T, mix)`` where ``P`` is pressure (dynes/cm^2), ``T`` is
            temperature (K), and ``mix`` is a species->mixing-ratio dict.
        """

        T = self.var.temperature
        P = self.wrk.pressure_hydro
        mix = self.mole_fraction_dict()
        mix.pop('alt')
        mix.pop('pressure')
        mix.pop('density')
        mix.pop('temp')

        return P, T, mix

    def model_state_to_dict(self):
        """Serialize model state needed for restart.

        Returns
        -------
        dict
            Restart dictionary compatible with :meth:`initialize_from_dict`.
        """

        if self.rdat.log10P_interp is None:
            raise Exception('This routine can only be called after `initialize_to_PT_bcs`')

        out = {}
        out['rdat'] = deepcopy(self.rdat.__dict__)
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['particle_radius'] = self.var.particle_radius

        # Other settings
        out['equilibrium_time'] = self.var.equilibrium_time
        out['verbose'] = self.var.verbose
        out['atol'] = self.var.atol
        out['autodiff'] = self.var.autodiff

        return out

    def initialize_from_dict(self, out):
        """Restore model state from :meth:`model_state_to_dict` output.

        Parameters
        ----------
        out : dict
            Restart dictionary created by :meth:`model_state_to_dict`.
        """

        for key, value in out['rdat'].items():
            setattr(self.rdat, key, value)

        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']
        self.var.particle_radius = out['particle_radius']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])

        # Other settings
        self.var.equilibrium_time = out['equilibrium_time']
        self.var.verbose = out['verbose']
        self.var.atol = out['atol']
        self.var.autodiff = out['autodiff']
        
        # Now set boundary conditions
        Pi = self.rdat.Pi
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

        self.prep_atmosphere(self.wrk.usol)

@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:] # type: ignore
    T : types.double[:] # type: ignore
    mubar : types.double[:] # type: ignore

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    if P_top < P[-1]:
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Integrate to TOA
    out = integrate.solve_ivp(hydrostatic_equation, [P_[0], P_[-1]], np.array([0.0]), t_eval=P_, args=args, rtol=1e-6)
    assert out.success

    # Stitch together
    z_ = out.y[0]

    return P_, T_, mubar_, z_

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""

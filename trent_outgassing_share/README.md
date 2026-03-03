# For Trent

## L98-59 b example

Set up a new conda environment.

```sh
conda env create -f environment.yaml
conda activate outgassing
```

Run the example, and check out the results in the `figures/` directory. You must first run `python input_files.py` to get required input files.

```sh
python input_files.py
python main.py
```

## Description of what is happening in `main.py`

`main.py` does a two-stage calculation:

1. `AdiabatClimateEquilibrium.solve(N_i)` finds a climate state for fixed volatile column inventories `N_i` (mol/cm^2) where the surface composition is chemically equilibrated at the same surface `P-T` state used by the climate solve (i.e., climate and surface chemical equilibrium are solved self-consistently).
2. That P-T-composition profile is then used to initialize `EvoAtmosphereRobust`, which is integrated to photochemical steady state.

Physical approach and approximations:

- Bulk inventory control:
  The user specifies bulk surface/atmospheric volatile columns (e.g., CO2, H2O, SO2). These are treated as constraints on total available material, not as fixed vertical profiles.

- Surface-only thermochemical equilibrium closure:
  For each trial surface temperature, the code computes an equilibrium gas composition only at the surface pressure/temperature using `ChemEquiAnalysis`, based on elemental abundances implied by `N_i`.
  It does **not** enforce thermochemical equilibrium independently at every altitude.

- Vertical transport in photochemistry:
  After climate convergence, the photochemical model is initialized with the climate P-T profile and a prescribed constant eddy diffusion (`Kzz = 1e6 cm^2/s` in this example), then integrated to steady state.

- One-way climate-to-photochemistry coupling:
  During the photochemical integration, the temperature profile is held fixed to the climate solution; the model does not iterate photochemical composition back into the climate solver.
  This is usually a decent approximation when photochemical products are trace or when they do not strongly alter the radiative budget compared to the major climate-setting gases.

- Scope:
  This is a practical reduced-order coupling: thermochemical equilibrium constrains the deep/surface boundary state, while disequilibrium chemistry and transport shape the vertical atmospheric composition above.

## Using Codex/Claude

If you use a coding tool like Codex or Claude, then download all the Photochem source with the commands bellow, and tell the bot to read the code when trying to make changes.

```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/Nicholaswogan/photochem/archive/refs/tags/v0.8.2.zip \
  https://github.com/Nicholaswogan/clima/archive/refs/tags/v0.7.4.zip \
  https://github.com/Nicholaswogan/Equilibrate/archive/refs/tags/v0.2.1.zip
do
  f="$(basename "$u")"
  wget -O "$f" "$u"
  unzip -q "$f"
  rm -f "$f"
done
cd ..
```

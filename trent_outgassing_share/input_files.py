from photochem.utils import settings_file_for_climate, species_file_for_climate, stars
from photochem.utils._format import yaml, FormatSettings_main, MyDumper
from photochem.utils import zahnle_rx_and_thermo_files
import planets
from astropy import constants

def main():

    planet_mass = float(constants.M_earth.cgs.value)*planets.L9859b.mass
    planet_radius = float(constants.R_earth.cgs.value)*planets.L9859b.radius
    surface_albedo = 0.1

    species_file_for_climate(
        filename='trent_outgassing_share/input/species_climate.yaml',
        species=['CH4','CO','CO2','H2O','O2','O3','OCS','SO2','H2','H2S','S2','SO','CS','CS2'],
        condensates=['H2O','CO2']
    )
    
    settings_file_for_climate(
        filename='trent_outgassing_share/input/settings_climate.yaml',
        planet_mass=planet_mass,
        planet_radius=planet_radius,
        surface_albedo=surface_albedo
    )

    stars.muscles_spectrum(
        star_name='GJ176',
        outputfile='trent_outgassing_share/input/gj176_scaled_to_l9859b.txt',
        Teq=planets.L9859b.Teq
    )

    zahnle_rx_and_thermo_files(
        atoms_names=['H', 'O', 'C', 'S'],
        rxns_filename='trent_outgassing_share/input/zahnle_HOCS.yaml',
        thermo_filename=None,
        remove_reaction_particles=True
    )

    settings_file = {
        'atmosphere-grid': {
            'bottom': 0.0, 
            'top': 'atmospherefile', 
            'number-of-layers': 100
        },
        'planet': {
            'planet-mass': planet_mass,
            'planet-radius': planet_radius,
            'surface-albedo': surface_albedo,
            'solar-zenith-angle': 60.0,
            'hydrogen-escape': {'type': 'none'},
            'water': {'fix-water-in-troposphere': False, 'gas-rainout': False, 'water-condensation': False}
        },
        'boundary-conditions': [{
            'name': 'H2',
            'lower-boundary': {'type': 'vdep', 'vdep': 0.0},
            'upper-boundary': {'type': 'veff', 'veff': 0.0}
        }]
    }
    settings_file = FormatSettings_main(settings_file)
    with open('trent_outgassing_share/input/settings.yaml','w') as f:
        yaml.dump(settings_file,f,Dumper=MyDumper,sort_keys=False,width=70)

if __name__ == '__main__':
    main()
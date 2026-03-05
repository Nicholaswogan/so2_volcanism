from photochem.utils import stars
import numpy as np

class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    eclipse_duration: float # in seconds
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, eclipse_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.eclipse_duration = eclipse_duration
        self.a = a
        self.stellar_flux = stellar_flux

# Cadieux et al. (2025), unless otherwise noted.

L9859b = Planet(
    radius=0.837,
    mass=0.46,
    Teq=620.0,
    transit_duration=1.01*60*60,
    eclipse_duration=1.01*60*60, # Assumed same as transit.
    a=0.0223,
    stellar_flux=stars.equilibrium_temperature_inverse(620.0, 0.0)
)

L9859 = Star(
    radius=0.3155,
    Teff=3415.0,
    metal=-0.46,
    kmag=7.1, # From Exo.Mast
    logg=4.91,
    planets={'b': L9859b}
)
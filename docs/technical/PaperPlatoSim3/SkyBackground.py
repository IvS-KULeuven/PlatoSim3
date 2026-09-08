import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from platosim.validation import switchOffAllEffects
from platosim.simulation import Simulation

# First make a grid in Equatorial coordinates
N = 120
x = np.linspace(0, 360, N)
y = np.linspace(-90, 90, N)
x, y = np.meshgrid(x, y)
RA  = x.flatten() * u.degree
Dec = y.flatten() * u.degree

c = SkyCoord(ra=RA, dec=Dec, frame='icrs')
ra_rad = c.ra.wrap_at(180 * u.deg).radian
dec_rad = c.dec.radian

plt.figure(figsize=(8,4.2))
plt.subplot(111, projection="aitoff")
plt.title("Aitoff grid")
plt.grid(True)
plt.plot(ra_rad, dec_rad, 'o', markersize=1)
plt.subplots_adjust(top=0.95,bottom=0.0)
plt.show()

# Compute the sky background for each grid point (as pointing) using PlatoSim
N = len(RA.value)
sky = np.zeros(N)

for i, ra, dec in zip(range(N), RA.value, Dec.value):
    
    print(i)
    sim = Simulation("output_skybackground", outputDir=os.getcwd())
    sim.useFastCamera()
    switchOffAllEffects(sim)
    
    # One full-frame exposure
    sim["ObservingParameters/NumExposures"] = 1
    sim["ObservingParameters/RApointing"]   = ra
    sim["ObservingParameters/DecPointing"]  = dec
    sim["Sky/SkyBackground"]   = -1
    sim["SubField/NumRows"]    = 1
    sim["SubField/NumColumns"] = 1
    
    # Define catalogue
    if i == 0:
        row = np.array([100])
        col = np.array([100])
        mag = np.array([30])
        ID  = [0]
        starcatFile = os.getcwd() + "/starcat.txt"
    sim.createStarCatalogFileFromPixelCoordinates(row, col, mag, ID, starcatFile)
    
    # Set subfield to same location as platform pointing
    sim.setSubfieldAroundCoordinates(ra, dec, 1, 1, normal=True)
    
    # Make sure no sources are located in the sub-field
#     sim["ObservingParameters/DecPointing"] = -sim["ObservingParameters/DecPointing"]

    # Simulate each sky background
    simfile = sim.run(removeOutputFile = True)
    sky[i] = simfile.getImage(0)

np.savetxt(os.getcwd() + "/sky2.txt", np.transpose([RA.value, Dec.value, sky]))

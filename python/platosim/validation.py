#!/usr/bin/env python3

"""
Auxiliary scripts to facilitate the validation and verification of PlatoSim:
    - switch off all effects for the PlatoSim simulations
    - conversion between equatorial and galactic coordinates
    - absolute stellar aberration
"""

# Built-in
from math import sin, cos, radians, pow

# PlatoSim standard
import numpy as np
from scipy import optimize
from astropy.coordinates import SkyCoord



def switchOffAllEffects(sim):

    """
    PURPOSE: Switch off all effects for the given simulation:
                - variability;
                - cosmics;
                - jitter;
                - thermo-elastic drift;
                - aberration correction (absolute + differential);
                - field distortion;
                - charge diffusion;
                - jitter smoothing;
                - flatfield;
                - dark signal;
                - brighter-fatter effect (BFE);
                - photon noise;
                - readout noise;
                - charge transfer inefficiency (CTI);
                - open-shutter smearing;
                - quantum efficiency;
                - overall relative transmissivity;
                - polarisation;
                - particulate contamination;
                - molecular contamination;
                - convolution with the PSF;
                - full-well saturation (blooming);
                - digital saturation;
                - quantisation.
                - non-linear gain
                - straylight
                - bad pixels

    INPUT:
        - sim: Simulation for which to switch off all effects.

    OUTPUT:
        - Simulation with all effects switched off.
    """

    # Straylight
    sim["Sky/StrayLight/IncludeStrayLight"] = "no"

    # Sky parameters

    sim["Sky/IncludeVariableSources"] = "no"
    sim["Sky/IncludeCosmicsInSubField"] = "no"
    sim["Sky/IncludeCosmicsInSmearingMap"] = "no"
    sim["Sky/IncludeCosmicsInBiasMap"] = "no"

    # Platform parameters

    sim["Platform/UseJitter"] = "no"

    # Telescope parameters

    sim["Telescope/UseDrift"] = "no"

    # Camera parameters

    sim["Camera/IncludeAberrationCorrection"] = "no"
    sim["Camera/IncludeFieldDistortion"] = "no"
    sim["Camera/IncludePointLikeGhosts"] = "no"
    sim["Camera/IncludeExtendedGhosts"]  = "no"

    # PSF parameters

    sim["PSF/MappedFromFile/IncludeChargeDiffusion"] = "no"
    sim["PSF/AnalyticNonGaussian/IncludeChargeDiffusion"] = "no"
    sim["PSF/MappedFromFile/IncludeJitterSmoothing"] = "no"

    # CCD parameters

    sim["CCD/IncludeFlatfield"] = "no"
    sim["CCD/IncludeDarkSignal"] = "no"
    sim["CCD/IncludeBFE"] = "no"
    sim["CCD/IncludePhotonNoise"] = "no"
    sim["CCD/IncludeReadoutNoise"] = "no"
    sim["CCD/IncludeCTIeffects"] = "no"
    sim["CCD/IncludeOpenShutterSmearing"] = "no"
    sim["CCD/IncludeQuantumEfficiency"] = "no"
    sim["CCD/IncludeRelativeTransmissivity"] = "no"
    sim["CCD/IncludePolarization"] = "no"
    sim["CCD/IncludeParticulateContamination"] = "no"
    sim["CCD/IncludeMolecularContamination"] = "no"
    sim["CCD/IncludeConvolution"] = "no"
    sim["CCD/IncludeFullWellSaturation"] = "no"
    sim["CCD/IncludeDigitalSaturation"] = "no"
    sim["CCD/IncludeQuantisation"] = "no"
    sim["CCD/IncludeGainNonlinearity"] = "no"
    sim["CCD/BadPixelMap/includeBadPixelMap"] = "no"

    return sim





def equatorial2galactic(ra, dec):

    """
    PURPOSE: Convert the given equatorial coordinates to galactic coordinates.

    INPUT:
        - ra: Right ascension [degrees]
        - dec: Declination [degrees]

    OUTPUT:
        - galactic longitude [radians]
        - galactic latitude [radians]
    """

    coordinates = SkyCoord(ra, dec, frame = 'icrs', unit = 'deg').transform_to('geocentrictrueecliptic')

    return (coordinates.lon.radian, coordinates.lat.radian)





def galactic2equatorial(lon, lat):

    """
    PURPOSE: Convert the given galactic coordinates to equatorial coordinates.

    INPUT:
        - lon: Galactic longitude [radians]
        - dec: Galactic latitude [radians]

    OUTPUT:
        - right ascension [degrees]
        - declination [degrees]
    """

    coordinates = SkyCoord(lon, lat, frame = 'geocentrictrueecliptic', unit = 'rad').transform_to('icrs')
    
    return (coordinates.ra.deg, coordinates.dec.deg)





def aberration(lon, lat, lonSpacecraft):

    """
    PURPOSE: Aberrate the given ecliptic coordinates for the given ecliptic longitude 
             of the spacecraft in its (circular) orbit around the Sun.
    
    INPUT:
        - lon: Ecliptic longitude [radians]
        - lat: Ecliptic latitude [radians]
        - lonSpacecraft: Ecliptic longitude of the spacecraft in its (circular) orbit
                         around the Sun [radians]
    
    OUTPUT:
        - aberrated ecliptic longitude [radians]
        - aberrated ecliptic latitude [radians]
    """


    amplitude = 30 / 300000

    deltaLon   = -amplitude * sin(lon - lonSpacecraft) / cos(lat)
    deltaLat   = -amplitude * cos(lon - lonSpacecraft) * sin(lat)

    return (lon + deltaLon, lat + deltaLat)




def gaussian1D(amplitude, center, sigma):

    """
    PURPOSE: Return a 1D gaussian funrction with the given parameters.

    INPUT:
        - amplitue: Height / amplitude of the gaussian
        - center: Coordinate of the centre of the gaussian
        - sigma: Width of the gaussian

    OUTPUT:
        - 1D gaussian function
    """

    return lambda x: amplitude * np.exp(-(((center - x) / sigma)**2))

def oneOverF2(factor):

    return lambda f: factor / f**2





def gaussian2D(amplitude, centerRow, centerColumn, sigmaRow, sigmaColumn):

    """
    PURPOSE: Return a 2D gaussian function with the given parameters.

    INPUT:
        - amplitude: Height/amplitude of the gaussian
        - centerRow: Row coordinate of the centroid of the gaussian
        - centerColumn: Column coordinate of the centroid of the gaussian
        - sigmaRow: Width of the gaussian in the row direction
        - sigmaColumn: Width of the gaussian in the column direction

    OUTPUT:
        - 2D gaussian function
    """

    return lambda row, column: amplitude * np.exp(-(((centerRow - row) / sigmaRow)**2 + ((centerColumn - column) / sigmaColumn)**2) / 2)




def fitGaussian1D(data, amplitude, center, sigma, subtractConstant = True):

    if subtractConstant:

        data -= np.median(data)
    
    initialParameters = [amplitude, center, sigma]
    errorfunction = lambda p: np.ravel(gaussian1D(*p)(*np.indices(data.shape)) - data)

    parameters = optimize.leastsq(errorfunction, initialParameters)[0]
    return parameters




def fitGaussian2D(data, amplitude, centerRow, centerColumn, sigmaRow, sigmaColumn, subtractConstant = True):

    """
    PURPOSE: Fit a 2D gaussian function with the given initial parameters to the given data.

    INPUT:
        - data: Data to which to fit a 2D gaussian function
        - amplitude: Initial estimate for the amplitude of the gaussian
        - centerRow: Initial estimate for the row coordinate of the centroid of the gaussian
        - centerColumn: Initial estimate for the column coordinate of the centroid of the gaussian
        - sigmaRow: Initial estimate for the width of the gaussian in the row direction
        - sigmaColumn: Initial estimate for the width of the gaussian in the column direction
        - subtractConstant: Boolean indicating whether a constant should be subtract.  This constant
                            is the median value of the given data.
    
    OUTPUT:
        - parameter: Optimised parameters for the 2D gaussian function
    """

    if subtractConstant:

        data -= np.median(data)
    
    initialParameters = [amplitude, centerRow, centerColumn, sigmaRow, sigmaColumn]
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) - data)

    parameters = optimize.leastsq(errorfunction, initialParameters)[0]
    return parameters





def fitOneOverF2(data, factor):

    initialParameters = [factor]
    errorfunction = lambda p: np.ravel(oneOverF(*p)(*np.indices(data.shape)) - data)

    parameters = optimize.leastsq(errorfunction, initialParameters)[0]
    return parameters

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is an integrated part of PlatoSim's toolkit PLATOnium. 
Given a star and a planet this script creates a synthetic stellar
and exoplanet variability model that can be used directly as input
for PlatoSim. A bolometric PLATO passband correction is applied to 
each photometric amplitude signal using synthetic high-resolution
PHOENIX spectra (AFGKM stars) or medium-resolution ATLAS9 spectra
(OBA stars). 
"""

notes="""
-------------------------
    Solar-like stars    :
-------------------------

Parsing the argument "--star" or "--star_params" will generate a
star template (for F5-K7 dwarf and subgiants) that includes:
  - Granulation noise              (VarSim: De Ridder+2009)
  - Convection driven oscillations (VarSim: De Ridder+2009)
  - Solar spot modulated activity  (pyspot: Aigrain+2015)
  - Solar flare events             (VarSim: Daveport+2014)

We provide a few benchmark stars ("--star <object>"):
  - GJ1214     (M  V)
  - WASP-43    (K7 V)
  - CoRoT-1    (G0 V)
  - Sun        (G0 V)
  - Kepler-21  (F6 IV)
  - WASP-33    (A5 V)

Generate template from parameters ("--star_params <kwargs*>"):
  - M    : Stellar mass          [Msun]
  - R    : Stellar radius        [Rsun]
  - Teff : Effective temperature [K]
  - logg : log Surface gravity   [dex]
  - Z    : Metallicity [Fe/H]    [dex]

Usage example:
  $ varsim --star Sun --quarter 1-8 -o </path/to/file> -p
  $ varsim --star_params 1 1 5800 4.5 0 -o </path/to/file>

-------------------------
    Exoplanet models    :
-------------------------

Parsing the argument "--planet" or "--planet_params" will generate a
template of a transiting planet. In combined set of models are:
  - Transits                (BATMAN: Kriedberg+2015)
  - Occultations            (SPIDERMAN: Louden & Kriedberg 2018)
  - Limb Darkening          (LDTk: Parviainen & Aigrain 2015)
  - Doppler boosting        (VarSim: PyAstronomy)
  - Ellipsoidal variations  (VarSim: PyAstronomy)

We provide a few benchmark planets ("--planet <object>"):
  - Earth        (solar system)
  - Neptune      (solar system)
  - Jupiter      (solar system)
  - hot-Earth    (solar system, short-period)
  - hot-Neptune  (solar system, short-period)
  - hot-Jupiter  (solar system, short-period)
  - WASP-43b     (hot-Jupiter, with phase-curve)
  - CoRoT-1b     (hot-Jupiter, with phase-curve)
  - Kepler-21b   (super-Earth, short-period)
  - WASP-33b     (hot-Jupiter, with phase-curve)

Generate transits from parameters ("--planet_params <kwargs*>"):
  - t0 : Time of emphemeris     [days]
  - P  : Orbital period         [days]
  - i  : Inclination            [deg]
  - w  : Argument of periastron [deg]
  - Rp : Planet radius          [R_Earth]
  - Mp : Planet mass            [M_Earth]

Generate occultation from parameters ("--phase_curve <kwargs*>"):
(model i from Zhang and Showman 2017)
  - xi : Ratio of radiative-to-advective timescale
  - Tn : Temperature of the planet's nightside [K]
  - dT : Day-night temperature contrast        [K]

Usage example:
  $ varsim --star CoRoT-1 --planet CoRoT-1b --quarter 1-8 -o </path/to/file> -p
  $ varsim --star_params 1 1 5800 4.5 0 --planet_params 1 10 0 90 0 1 1 -o </path/to/file>

-------------------------
    Pulsating stars     :
-------------------------

Besides producing variable templates for solar-like stars, it is
also possible to create a template for pulsating stars that are
more massive and more evolved than the Sun:
  - beta Cephei    (bCep)  [ToyModel, HeyAerts2024, mocka]
  - Slowly puls B  (SPB)   [ToyModel, Pedersen2021, mocka]
  - delta Scuti    (dSct)  [ToyModel, Bowman2018,   mocka]
  - gamma Doradus  (gDor)  [ToyModel, Gang2020,     mocka]
  - roAp star      (roAp)  [ToyModel]
  - RR Lyrae       (RRLyr) [          Bodi2023,     mocka]
  - Cepheid        (Ceph)  [          Bodi2023,     mocka]

Names within the round brackets are benchmark stars ("--star <Object>").
Names within the square brackets are the mode model ("--puls <Model>"):
  - ToyModel    : Simple model drawing modes from distributions
  - "Reference" : Draw a random star from a the stellar benchmark sample
  - mocka       : Analytic model built from the benchmark sample 

Usage examples:
  $ varsim --star gDor --puls Gang2020 --quarter 1-8 -o </path/to/file> -p
"""

# Built-in
import os
import sys
import glob
import random
import argparse
import datetime
import warnings
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as ss
from scipy.ndimage import median_filter
from scipy.interpolate import make_interp_spline
from astropy.io import fits
from astropy import constants as c
from astropy import units as u

# PLATOnium standard
import natsort
import batman
from ldtk import LDPSetCreator, TabulatedFilter

# PlatoSim functions
import platosim.plot      as pt
import platosim.utilities as ut
#import platosim.smbhb     as smbhb
from platosim.utilities import errorcode
from platosim.spectrum  import Spectrum
from platosim.varsource import (Pulsator,
                                StellarFlares,
                                StellarSpots,
                                SolarLikeOscillator,
                                SurfaceModulations,
                                EclipsingBinary,
                                SMBHB,
                                PlanetMRforecast,
                                DopplerBeaming,
                                EllipsoidalDistortion)

#==============================================================#
#                         BEGIN CLASS                          #
#==============================================================#

class VarSim(object):
    """Class to generate noise-less light curves.
    """
    def __init__(self, args):
        
        # CONSTANTS
        
        self.Teff_sun = 5777.  # [K]
                
        # I/O SETTINGS

        # Check if notes are requested
        if args.notes:
            print(notes); exit()

        # Parameters in {True, False, None}
        self.plot   = args.plot
        self.seed   = args.seed
        self.ofile  = args.ofile
        self.starID = None
        
        # Star mode
        self.star        = args.star
        self.star_params = args.star_params

        # Planet mode
        self.planet = args.planet
        self.planet_params = args.planet_params
        #self.phase_curve = args.phase_curve TODO
        
        # Binary mode
        self.binary = args.binary

        # SMBHBB
        self.smbhb = args.smbhb
        self.smbhb_params = args.smbhb_params
        
        # Limb darkening model
        self.ldms = ['linear', 'quadratic', 'squareroot', 'power2']
        if args.ldm is None:
            self.ldm = 'power2'
        elif args.ldm not in self.ldms:
            errorcode('error', f'Limb Darkening model "{args.ldm}" is not available!' +
                      f'\nUse either: {self.ldms}')
        else:
            self.ldm = args.ldm
            
        # Use Kallinger2014 by default
        if args.gran == None: 
            args.gran = 'Kallinger2014'
        elif not args.gran in ['Kallinger2014']:
            args.gran = False

        # Use Corsaro2013 by default
        # NOTE allow parsing puls models
        if args.puls == None: 
            args.puls = 'Corsaro2013'
        elif args.puls == 'no':
            args.puls = False
        
        # Use Aigrain2015 by default
        if args.spot == None:
            args.spot = 'Aigrain2015'
        elif not args.spot in ['Aigrain2015']:
            args.spot = False

        # Use Doorsselaere2017 by default
        if args.flare == None:
            if args.spot:
                args.flare = 'Doorsselaere2017'
            else:
                args.flare = 'ToyModel'
        elif not args.flare in ['Doorsselaere2017', 'ToyModel']:
            args.flare = False

        # Project modes
        self.kul20 = args.kul20
        self.mocka = args.mocka
        self.mocka_solar = True
        
        # Prepare pandas series for parameters
        self.df = pd.Series()
        
        # Random number BitGenerator (PCG64)
        if not self.seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed)

        # Verbosity (a.k.a log level) -> Identical to PlatoSim usage
        if args.verbose == 0:
            self.verbose = 0            
            warnings.filterwarnings("ignore")
        elif args.verbose is None:
            self.verbose = 2
        else:
            self.verbose = args.verbose

        # Print software name
        if self.verbose > 1:
            errorcode('software', '\nVariable Source Simulator\n')
            
        # Data path for VarSim
        self.idir = str(Path(os.getenv("PLATO_PROJECT_HOME")) / 'inputfiles/data_varsim')

        # Output file
        if self.ofile is not None:
            self.ofile = Path(self.ofile).resolve()
            
        # Add latex font if catalogue is saved
        from platosim.matplotlibrc import setup; setup()
        # if self.ofile is None:
        #     from platosim.matplotlibrc import setup; setup()
        # else:
        #     from platosim.matplotlibrc import latex; latex()

        # Data (download) for usage of varsim
        if not Path(self.idir + '/passband_plato.txt').is_file():
            errorcode('message', 'Inuaguration: Welcome to the VarSim!')
            print(f'Downloading a few prerequisite files')
            ut.downloadFromFTP('passband_plato.txt',       self.idir)
            ut.downloadFromFTP('passband_cheops.txt',      self.idir)
            ut.downloadFromFTP('passband_tess.txt',        self.idir)
            ut.downloadFromFTP('passband_kepler.txt',      self.idir)
            ut.downloadFromFTP('varsim_meunier19a_t1.txt', self.idir)
            ut.downloadFromFTP('varsim_mainFitsBiSON.txt', self.idir)
            
        # OBS PARAMETERS

        # Cadence [d]
        if args.cadence:
            cadence = args.candece / 86400.
        else:
            cadence = 25 / 86400.

        # Parsing "quarter" overwrites "time"
        # NOTE Default is Q1 (t=0 and 91 days duration)
        if args.quarter:
            Q = ut.convertQuarterRange(args.quarter)
            if len(Q) == 1:
                numQuarters = 1
            elif len(Q) > 1:
                numQuarters = Q[1] - Q[0] + 1
            else:
                errorcode('error', 'Wrong input format of "quarter"!')        
            timeStart = ut.quarter() * (Q[0]-1)
            timeDur   = ut.quarter() * numQuarters

        elif args.time:
            timeStart = 0
            timeDur   = args.time
        else:
            timeStart = 0
            timeDur   = ut.quarter()

        # Time points (ensure even number of time points)
        time = np.arange(0, timeDur, cadence) + timeStart
        if len(time) % 2 != 0:
            time = np.arange(0, timeDur + cadence, cadence) + timeStart

        # Store parameters
        self.time      = time * u.d
        self.cadence   = cadence * u.d
        self.timeDur   = timeDur * u.d
        self.timeStart = timeStart * u.d

        # Load the instrument passband
        if args.inst:
            self.intrument = args.inst
        else:
            self.instrument = 'plato'
        passband = pd.read_csv(self.idir + f'/passband_{self.instrument}.txt', comment='#')
        self.wvl_tele = passband.wavelength.to_numpy() * u.nm  # Wavelengths [nm]
        self.tra_tele = passband.absolute.to_numpy()           # Absolute transmission
        
        # Define pandas data frame to store all signals
        self.lc = pd.DataFrame(data=self.time.to('s').value, columns=['time'])
        
        if self.verbose > 1:
            print(f'Simulating time series for : ' +
                  f'{len(self.time)} x {self.cadence.to("s")} ({self.timeDur.value:.1f} days)')
            print(f'Simulating {self.instrument} bandpass  : ' +
                  f'{self.wvl_tele[0]} - {self.wvl_tele[-1]}')

            
    #--------------------------------------------------------------#
    #                     BENCHMARK STARS/PLANETS                  #
    #--------------------------------------------------------------#
    
    def load_star(self, source):
        """Function to load benchmark stars used by varsim.

        NOTE in the following the astropy-units are required, however,
        the the choice of units (e.g. seconds vs. hours) are optional. 

        Parameters
        ----------
        source : str
            Stellar effective temperature [astropy.units]

        Returns
        -------
        Stellar parameters {M, R, Teff, logg, Z}
        """

        # Exoplanet host stars
        
        if source == 'GJ1214':
            spec = 'M'
            lum  = 'V'
            M    = 0.15  * u.M_sun
            R    = 0.216 * u.R_sun
            Teff = 3026 * u.K
            logg = 4.5
            Z    = 0.0

        if source == 'WASP-43':
            # http://exoplanet.eu/catalog/wasp-33_b/
            spec = 'K7'
            lum  = 'V'
            M    = 0.717 * u.M_sun
            R    = 0.667 * u.R_sun
            Teff = 4520 * u.K
            logg = 4.5
            Z    = 0.0

        if source == 'CoRoT-1':
            spec = 'G0'
            lum  = 'V'
            M    = 0.95 * u.M_sun
            R    = 1.11 * u.R_sun
            Teff = 6298 * u.K
            logg = 4.5
            Z    = 0.0

        if source == "Sun":
            spec = 'G0'
            lum  = 'V'
            R    = 1. * u.R_sun
            M    = 1. * u.M_sun
            Teff = 5777. * u.K
            logg = 4.5
            Z    = 0.0

        if source == 'Kepler-21':
            spec = 'F6'
            lum  = 'V'            
            M    = 1.41 * u.M_sun
            R    = 1.90 * u.R_sun
            Teff = 6305 * u.K
            logg = 4.5
            Z    = 0.0
            
        if source == 'WASP-33':
            spec = 'A5'
            lum  = 'V'
            M    = 1.59 * u.M_sun
            R    = 1.77 * u.R_sun
            Teff = 7430 * u.K
            logg = 4.5
            Z    = 0.0

        # Pulsating stars (MOCKA defaults)

        if source == 'bCep':
            M = 8.0 * u.M_sun
            R = 10.0 * u.R_sun
            Teff = 26000 * u.K
            logg = 3.8
            Z    = -0.5
        
        if source == 'SPB':
            M = 2.5 * u.M_sun
            R = 5.0 * u.R_sun
            Teff = 12000 * u.K
            logg = 4.0
            Z    = 0.0
        
        if source == 'dSct':
            M = 1.26 * u.M_sun
            R = 1.20 * u.R_sun
            Teff = 6071 * u.K
            logg = 4.0
            Z    = 0.0

        if source == 'gDor':
            M = 1.26 * u.M_sun
            R = 1.20 * u.R_sun
            Teff = 6071 * u.K
            logg = 4.0
            Z    = 0.0
            
        if source == 'roAp':
            M = 1.26 * u.M_sun
            R = 1.20 * u.R_sun
            Teff = 6071 * u.K
            logg = 4.0
            Z    = 0.0
            
        if source == 'RRLyr':
            M = 1.0 * u.M_sun
            R = 1.0 * u.R_sun
            Teff = 5500 * u.K
            logg = 3.5
            Z    = 0.0

        if source == 'Ceph':
            M = 2.0 * u.M_sun
            R = 2.0 * u.R_sun
            Teff = 7500 * u.K
            logg = 4.0
            Z    = 0.0
            
        if source == 'LPV':
            M = 2.0 * u.M_sun
            R = 2.0 * u.R_sun
            Teff = 7500 * u.K
            logg = 4.0
            Z    = 0.0

        return M, R, Teff, logg, Z


    def load_exoplanet(self, source):
        """Module to load benchmark exoplanets to be used by varsim.

        NOTE in the following the astropy-units are required, however,
        the the choice of units (e.g. seconds vs. hours) are optional. 

        Parameters
        ----------
        t0 : float
            Time of inferior conjunction (i.e. central time of first transit) [unit required]
        P : float
           Orbital period [astropy.units]
        a : float
            Semi-major axis [unit required]
        i : float
            Orbital inclination (90-0) [astropy.units]
        e : float
            Eccentricity (0-1)
        w : float
            Longitude of periastron (0-360) [astropy.units]
        rp : float
            Planet radius [astropy.units]
        fp : float
            Planet-to-star flux ratio

        Optional SPIDERMAN
        ------------------
        xi : float
            Ratio of radiative to advective timescale
        Tn : float
            Temperature of nightside [astropy.units]
        dT : float
           Day-night temperature contrast [astropy.units]
        """

        # Confirmed planets from literature
        
        if source == 'WASP-43b':  # K7 V
            # http://exoplanet.eu/catalog/wasp-43_b/
            params = {'t0': 1 * u.d,
                      'P' : 0.81347753 * u.d,
                      'e' : 0.0035,
                      'i' : 82.33 * u.deg,
                      'w' : 328 * u.deg,
                      'rp': 1.036 * u.R_jup,
                      'mp': 2.052 * u.M_jup,
                      'xi': 0.0,
                      'Tn': 1000 * u.K,
                      'dT': 1000 * u.K}
        
        if source == 'CoRoT-1b':
            # http://exoplanet.eu/catalog/corot-1_b/
            params = {'t0': 1 * u.d,
                      'P' : 1.5089557 * u.d,
                      'e' : 0.0,
                      'i' : 83.96 * u.deg,
                      'w' : 90.0 * u.deg,
                      'rp': 1.49 * u.R_jup,
                      'mp': 1.03 * u.M_jup,
                      'xi': 0.1,
                      'Tn': 1757 * u.K,
                      'dT': (3144 - 1757) * u.K}

        if source == 'Kepler-21b':  # F6 IV
            # http://exoplanet.eu/catalog/kepler-21_b/
            params = {'t0': 1 * u.d,
                      'P' : 2.78578 * u.d,
                      'e' : 0.02,
                      'i' : 83.96 * u.deg,
                      'w' : -15. * u.deg,
                      'rp': 1.636 * u.R_earth,
                      'mp': 5.079 * u.M_earth,
                      'xi': 0.,
                      'Tn': 300. * u.K,
                      'dT': 50. * u.K}
            
        if source == 'WASP-33b':  # A5 V
            # http://exoplanet.eu/catalog/wasp-33_b/
            params = {'t0': 1 * u.d,
                      'P' : 1.21986967 * u.d,
                      'e' : 0.0,
                      'i' : 87.7 * u.deg,
                      'w' : 130.0 * u.deg,
                      'rp': 1.603 * u.R_jup,
                      'mp': 2.8 * u.M_jup,
                      'xi': 0.1,
                      'Tn': 1757 * u.K,
                      'dT': (3144 - 1757) * u.K}

        # Theoretical "hot" (low P) planets:
        
        if source == 'hot-Earth':
            # 
            params = {'t0': 1 * u.d,
                      'P' : 10 * u.d,
                      'e' : 0.,
                      'i' : 90. * u.deg,
                      'w' : 0. * u.deg,
                      'rp': 1. * u.R_earth,
                      'mp': 1. * u.M_earth,
                      'xi': 0.,
                      'Tn': 500. * u.K,
                      'dT': 0. * u.K}

        if source == 'hot-Neptune':
            # 
            params = {'t0': 1 * u.d,
                      'P' : 10 * u.d,
                      'e' : 0.0167,
                      'i' : 90.0 * u.deg,
                      'w' : 0. * u.deg,
                      'rp': 3.9 * u.R_earth,
                      'mp': 17.15 * u.M_earth,
                      'xi': 0.,
                      'Tn': 300. * u.K,
                      'dT': 0. * u.K}

        if source == 'hot-Jupiter':
            # 
            params = {'t0': 1 * u.d,
                      'P' : 10 * u.d,
                      'e' : 0.,
                      'i' : 88. * u.deg,
                      'w' : 0.0 * u.deg,
                      'rp': 1.0 * u.R_jup,
                      'mp': 1.0 * u.M_jup,
                      'xi': 0.0,
                      'Tn': 300.0 * u.K,
                      'dT': 0.0 * u.K}
            
        #--------------------------------------------
        
        if source == 'Earth':
            # 
            params = {'t0': 10 * u.d,
                      'P' : 365.25 * u.d,
                      'e' : 0.0167,
                      'i' : 90.0 * u.deg,
                      'w' : 0. * u.deg,
                      'rp': 1. * u.R_earth,
                      'mp': 1. * u.M_earth,
                      'xi': 0.,
                      'Tn': 300. * u.K,
                      'dT': 50. * u.K}

        if source == 'Neptune':
            # 
            params = {'t0': 10 * u.d,
                      'P' : 365.25 * u.d,
                      'e' : 0.0167,
                      'i' : 90.0 * u.deg,
                      'w' : 0. * u.deg,
                      'rp': 3.9 * u.R_earth,
                      'mp': 17.15 * u.M_earth,
                      'xi': 0.,
                      'Tn': 300. * u.K,
                      'dT': 50. * u.K}

        if source == 'Jupiter':
            # Parameters are drawn from astropy
            params = {'t0': 10 * u.d,
                      'P' : 100 * u.d,
                      'i' : 90 * u.deg,
                      'e' : 0,
                      'w' : 90 * u.deg,
                      'rp': 1.0 * u.R_jup,
                      'mp': 1.0 * u.M_jup,
                      'xi': 0.0,
                      'Tn': 300 * u.K,
                      'dT': 300 * u.K}
            
        return params
        

    def load_exomoon(self, source):
        """Module to load benchmark exopmoon to be used by varsim.

        NOTE in the following the astropy-units are required, however,
        the the choice of units (e.g. seconds vs. hours) are optional. 

        Parameters
        ----------
        R : float
            Moon radius [astropy.units]
        Mr : float
            Planet-to-moon mass ratio [0, 1]
        P : float
           Orbital period [astropy.units]
        tau : float
            [0, 1] [astropy.units]
        W : float
            [0, 180] [astropy.units]
        i : float
            Orbital inclination [0, 90] [astropy.units]
        e : float
            Eccentricity (0-1)
        w : float
            Longitude of periastron [0, 360] [astropy.units]
        """
        
        if source == 'pandora':
            # Default example from Pandora tutorial
            params = {'R'   : 5.0 * u.R_earth,
                      'M'   : 10.0 * u.M_earth,
                      'P'   : 0.3 * u.d,
                      'tau' : 0.07,
                      'W'   : 0.0  * u.deg,
                      'i'   : 80.0 * u.deg,
                      'e'   : 0.9,
                      'w'   : 20.0 * u.deg}

        return params

    
    #--------------------------------------------------------------#
    #                       STELLAR PARAMETERS                     #
    #--------------------------------------------------------------#
    
    def stellar_source(self):
        """Select the stellar paramters.
        """

        if self.verbose > 1:
            errorcode('module', '\nStellar parameters\n')
        
        # Check star source or use Sun as default
        if self.star:
            self.star_source = self.star
        else:
            self.star_source = 'Sun'
                    
        # If stellar parameters are parsed
        if args.star_params:            
            star = args.star_params[0]
            self.M    = star[0] * u.M_sun
            self.R    = star[1] * u.R_sun
            self.Teff = star[2] * u.K
            self.logg = star[3]
            self.Z    = star[4]
            self.spec = None
            
        # If Gaia ID is parsed
        elif self.mocka:
            self.star_source = self.df.gaiaDR3
            self.spec = self.df.spec
            self.Teff = self.df.Teff * u.K
            self.logg = self.df.logg
            self.Z    = self.df.Z
            self.R    = self.df.R * u.R_sun
            self.M    = self.df.M * u.M_sun
            self.L    = self.df.L * u.L_sun

        # Else use a benchmark star is parsed
        else:
            star = self.load_star(self.star_source)
            self.spec = None
            self.M    = star[0]
            self.R    = star[1]
            self.Teff = star[2]
            self.logg = star[3]
            self.Z    = star[4]

        # Use theoretical logg if user has set logg=0
        if self.logg == 0:
            g = c.G.cgs * self.M.cgs / self.R.cgs**2
            self.logg = np.log10(g.value)
        
        # Derive theoretical luminosity
        if not self.mocka:
            self.L = self.R.value**2 * (self.Teff.value/self.Teff_sun)**4 * u.L_sun

        # Secure spectral type exist (assuming main-sequence star):
        # https://sites.uni.edu/morgans/astro/course/Notes/section2/spectraltemps.html 
        if self.spec in [None, '', 'CSTAR', 'unknown']:
            Teff = self.Teff.value
            if   Teff > 29200                  : self.spec = 'O'
            elif Teff >  9600 and Teff <= 29200: self.spec = 'B'
            elif Teff >  7350 and Teff <=  9600: self.spec = 'A'
            elif Teff >  6050 and Teff <=  7350: self.spec = 'F'
            elif Teff >  5240 and Teff <=  6050: self.spec = 'G'
            elif Teff >  3750 and Teff <=  5240: self.spec = 'K'
            elif                  Teff <=  3750: self.spec = 'M'

        # Return parameters
        self.df['L_Lsun'] = self.L.to('L_sun').value
        self.df['M_Msun'] = self.M.to('M_sun').value
        self.df['R_Rsun'] = self.R.to('R_sun').value
        self.df['Teff_K'] = self.Teff.value
        self.df['logg']   = self.logg
        self.df['Z']      = self.Z
        self.df['alpha']  = 0.0
            
        # Print available stellar model parameters
        if self.verbose > 1:
            print(f"Stellar ID      : {self.star_source}")
            print(f"Spectral type   : {self.spec}")
            print(f"Stellar Teff    : {self.df.Teff_K:4.0f}")
            print(f"Surface gravity : {self.df.logg:.2f} dex")
            print(f"Stellar [M/Fe]  : {self.df.Z:.3f} dex")
            print(f"Stellar radius  : {self.df.R_Rsun:.2f}")
            print(f"Stellar mass    : {self.df.M_Msun:.2f}")
            print(f"Luminosity      : {self.df.L_Lsun:.2f}")

        
    def stellar_spectrum(self):
        """Calculates the bolometric correction from high-res spectra.
        
        NOTE to compare theo L while using PhoenixAtmos, divide with np.pi
        """

        if self.verbose > 1:
            errorcode('module', '\nStellar spectrum\n')
        
       # Load parameters
        Teff = self.Teff.value
        logg = self.logg
        Z    = self.Z
        R    = self.R
        L    = self.L
        wvl_tele  = self.wvl_tele.to('AA').value
        tran_tele = self.tra_tele
        
        # Initialise class for synthetic spectra
        spec = Spectrum(verbose=self.verbose)

        # Select temeperature ranges
        if Teff <= 12200:
            library = 'PhoenixHiRes'
            if Teff < 7000: dT = 50
            else: dT = 100
        else:
            library = 'Atlas9'
            if Teff < 13000: dT = 125
            else: dT = 500

        # Make sure parameters are within limits
        Teff_lower, logg, Z, alpha = spec.nearest_parameters(Teff-dT, logg, Z, 0, library)
        Teff_upper, logg, Z, alpha = spec.nearest_parameters(Teff+dT, logg, Z, 0, library)

        # Fetch high resolution spectra
        if Teff <= 12200:
            self.wvl1_in, self.flux1_in = spec.getPhoenixHiResFITS(Teff_lower, logg, Z, 0)
            self.wvl2_in, self.flux2_in = spec.getPhoenixHiResFITS(Teff_upper, logg, Z, 0)
        else:
            self.wvl1_in, self.flux1_in = spec.getAtlasFITS(Teff_lower, logg, Z, alpha=0)
            self.wvl2_in, self.flux2_in = spec.getAtlasFITS(Teff_upper, logg, Z, alpha=0)
                
        # Check that the interpolation between the two SEDs can be done
        if len(self.wvl1_in) != len(self.wvl2_in):
            errorcode('error', 'Spectra are not of the same size! Check interpolation')

        # Make sure that interpolation converge between the two libaries
        if Teff_lower == Teff_upper:
            Teff_diff = 1
        else:
            Teff_diff = float(Teff_upper-Teff_lower)
            
        # Create SED for star by interpolating nearby absolutely calibrated spectra
        self.wvl_star = (self.wvl1_in + self.wvl2_in) / 2.
        self.flux_star = (self.flux1_in + (Teff-Teff_lower) * (self.flux2_in-self.flux1_in) *
                          Teff_diff**-1)
        
        # Consistnecy check
        if self.verbose > 1:
            Lum = 4*np.pi*(R.cgs.value)**2 * np.trapz(self.flux_star, self.wvl_star)
            print(f'Theoretical luminosity : {L.to("erg/s"):.3e}')
            print(f'Synthetic   luminosity : {Lum * u.erg/u.s:.3e}\n')            

        # Rebinned spectrum
        wvl_equi = np.arange(wvl_tele[0], wvl_tele[-1], 1)
        wvl_equi, flux_equi = ut.rebin3(wvl_equi, self.wvl_star, self.flux_star)

        # Get bolometric correction for solar-like oscillations
        self.luminosity_correction()
        
        # Get passband correction
        self.scale_tess   = self.passband_correction(passband_a='plato', passband_b='tess')
        self.scale_kepler = self.passband_correction(passband_a='plato', passband_b='kepler')
        if self.verbose > 1:
            print(f'Amplitude correction for oscillations  : {self.bol_coeff:.3f}')
            print(f'Passband  correction (Kepler -> PLATO) : {self.scale_kepler:.3f}')
            print(f'Passband  correction (TESS   -> PLATO) : {self.scale_tess:.3f}')
        self.df['PC_tess']   = self.scale_tess
        self.df['PC_kepler'] = self.scale_kepler
        
        # Plot interpolation
        if args.plot and self.verbose == 3:
            pt.plotSED(self.wvl_star,  self.wvl1_in,  self.wvl2_in,  wvl_equi,
                       self.flux_star, self.flux1_in, self.flux2_in, flux_equi,
                       Teff, Teff_upper, Teff_lower)

            
    def passband_correction(self, passband_a='plato', passband_b='kepler'):

        """Fetch passband data.
        """

        # Fetch passbands
        N = 10000
        wave_a, tran_a = ut.get_passband(passband_a, response='absolute', interpolate=True, n=N)
        wave_b, tran_b = ut.get_passband(passband_b, response='absolute', interpolate=True, n=N)

        # Fetch stellar spectrum
        wave_star = self.wvl_star / 10 # [AA -> nm]
        flux_star = self.flux_star
        
        # Flux within passband A and rebin to equidistant grid
        dex_wave_min_a = ut.findNearestIndex(wave_star, wave_a[0])
        dex_wave_max_a = ut.findNearestIndex(wave_star, wave_a[-1])
        wave_star_a    = wave_star[dex_wave_min_a:dex_wave_max_a]
        flux_star_a    = flux_star[dex_wave_min_a:dex_wave_max_a]
        wave_equi_a, flux_equi_a = ut.rebin3(wave_a, wave_star, flux_star)

        # Flux within passband B and rebin to equidistant grid
        dex_wave_min_b = ut.findNearestIndex(wave_star, wave_b[0])
        dex_wave_max_b = ut.findNearestIndex(wave_star, wave_b[-1])
        wave_star_b    = wave_star[dex_wave_min_b:dex_wave_max_b]
        flux_star_b    = flux_star[dex_wave_min_b:dex_wave_max_b]
        wave_equi_b, flux_equi_b = ut.rebin3(wave_b, wave_star, flux_star)

        # Flux within passbands
        flux_tran_a = flux_equi_a * tran_a * wave_equi_a
        flux_tran_b = flux_equi_b * tran_b * wave_equi_b

        # Debug-------------------------------------------
        # plt.figure(figsize=(8,6))
        # plt.plot(wave_equi_a, flux_tran_a, 'k-', lw=1, label=passband_a)
        # plt.plot(wave_equi_b, flux_tran_b, 'b-', lw=1, label=passband_b, alpha=0.7)
        # ylab=r'$T_{\lambda} \cdot F_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$ sr$^{-1}$]'
        # plt.xlabel('Wavelength [nm]')
        # plt.ylabel(ylab)
        # plt.xlim(min(wave_equi_a[0], wave_equi_b[0]),
        #          max(wave_equi_a[-1], wave_equi_b[-1]))
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        #-------------------------------------------------

        # Integrate to find ratio for correction
        F_a = np.trapz(flux_tran_a, wave_equi_a)
        F_b = np.trapz(flux_tran_b, wave_equi_b)

        return F_a / F_b


    def luminosity_correction(self):

        """Compute the luminosity gradient.

        This correction is used to scale the p-mode oscillation
        ampltitudes as function of the luminosity gradient.

        This function uses the model grid method described in Sarkar+2018:
        https://academic.oup.com/mnras/article/481/3/2871/5092616
        """
        
        # Load parameters
        Teff = self.Teff.value
        logg = self.logg
        Z    = self.Z
        R    = self.R
        L    = self.L        
        wvl_tele  = self.wvl_tele.to('AA').value
        tran_tele = self.tra_tele
        
        # Measure bolometric luminosity from SED [ergs/s]
        L1_bolometric = 4*np.pi*(R.cgs.value)**2 * np.trapz(self.flux1_in, self.wvl1_in)
        L2_bolometric = 4*np.pi*(R.cgs.value)**2 * np.trapz(self.flux2_in, self.wvl2_in)
        
        # Luminosity amplitude gradient in passband
        dex_wvl_min = ut.findNearestIndex(self.wvl_star, wvl_tele[0])
        dex_wvl_max = ut.findNearestIndex(self.wvl_star, wvl_tele[-1])
        if dex_wvl_max - dex_wvl_min == 1:
            L1_passband = (4*np.pi * (R.cgs.value)**2 *
                           ( self.wvl1_in[dex_wvl_max] -  self.wvl1_in[dex_wvl_min]) *
                           (self.flux1_in[dex_wvl_max] + self.flux1_in[dex_wvl_min]) / 2.)
            L2_passband = (4*np.pi * (R.cgs.value)**2 *
                           ( self.wvl2_in[dex_wvl_max] -  self.wvl2_in[dex_wvl_min]) *
                           (self.flux2_in[dex_wvl_max] + self.flux2_in[dex_wvl_min]) / 2.)
        else:
            L1_passband = (4*np.pi * (R.cgs.value)**2 *
                           np.trapz(self.flux1_in[dex_wvl_min:dex_wvl_max],
                                     self.wvl1_in[dex_wvl_min:dex_wvl_max]))
            L2_passband = (4*np.pi * (R.cgs.value)**2 *
                           np.trapz(self.flux2_in[dex_wvl_min:dex_wvl_max],
                                     self.wvl2_in[dex_wvl_min:dex_wvl_max]))
        
        # Bolometric cofficient
        self.bol_coeff = (ut.diff(L2_passband,   L1_passband) /
                          ut.diff(L2_bolometric, L1_bolometric))

        # Store correction
        self.df['A_bol_corr'] = self.bol_coeff
            
        # # Absolute bolometric magnitude of star TODO delete?
        # M_bolometric = round(ut.diff(L2_bolometric, L1_bolometric) /
        #                      ut.diff(Teff_upper, Teff_lower), 2)
        # M_passband   = round(ut.diff(L2_passband, L1_passband) /
        #                      ut.diff(Teff_upper, Teff_lower), 2)
        # # Consistency check [mag]
        # if self.verbose > 1:
        #     print(f'Absolute magnitude bol : {round(M_bolometric, 4)}')
        #     print(f'Absolute magnitude lam : {round(M_passband,   4)}')            
        
            
    #--------------------------------------------------------------#
    #                   MODELS OF SOLAR-LIKE STARS                 #
    #--------------------------------------------------------------#

    
    def solar_granosc(self):

        """Model convection driven oscillations.
        """

        if self.verbose > 1:
            errorcode('module', '\nSolar-like oscillations\n')
            print(f'Scaling relation gran : {args.gran}')
            print(f'Scaling relation puls : {args.puls}')

        # Get amplitude correction
        self.luminosity_correction()
            
        # Initialize and prepare model input
        params = [self.Teff, self.R, self.M, self.L]
        model  = SolarLikeOscillator(self.time, params, self.idir, seed=self.seed)

        # Model granulation
        if args.gran is not None:
            params_gran = model.init_granulation(scaling=args.gran)
            self.lc['gran'] = model.eval_granulation() * self.bol_coeff
            self.df['A_gran_ppm'] = ut.rootMeanSquare(self.lc.gran)
            if self.verbose > 1:
                print(f'Granulation amplitude : {self.df.A_gran_ppm:.2f} ppm')

        # Model stochastic oscillations
        if args.puls is not None:
            params_puls = model.init_oscillations(scaling=args.puls)
            self.params_puls = params_puls[:3]
            self.lc['puls'] = model.eval_oscillations() * self.bol_coeff
            self.df['A_puls_ppm']   = ut.rootMeanSquare(self.lc.puls)
            self.df['numax_muHz']   = self.params_puls[0]
            self.df['deltanu_muHz'] = self.params_puls[0]
            if self.verbose > 1:
                print(f'Pulsation   amplitude : {self.df.A_puls_ppm:.2f} ppm')
                print(f'Frequency   nu_max    : {params_puls[0]:.2f} microHz')
                print(f'Splitting   Delta_nu  : {params_puls[1]:.2f} microHz')

        # Fix that the granulation time series is some time 1 time point shorter
        if args.gran and args.puls:
            if len(self.lc.gran) != len(self.lc.puls):
                self.lc.gran = np.append(self.lc.gran, self.lc.gran.mean())
                
        # Plot rsults
        if self.plot:
            pt.plotGranOscAmplitudeSpectrum(self.lc, numax=params_puls[0])
        
        # Convert to relative flux [ppm -> pp1]
        self.lc.gran = self.lc.gran / 1e6 + 1
        self.lc.puls = self.lc.puls / 1e6 + 1
            

    def solar_spots(self):

        """Model stellar spot modulations.
        """

        if self.verbose > 1:
            errorcode('module', '\nStellar spot cycle\n')

        # Use a random uniform distribution
        # NOTE secure a lower misalignment for planetary systems
        # Else use cos(i_star) uniform between 0-90 deg
        if self.planet or self.planet_params or self.kul20:
            incl = self.rng.uniform(85, 90)
        else:
            incl = None

        # Initialise model
        model = StellarSpots(seed=self.seed)
            
        # Generate model
        lc, params, area = model.evaluate(teff=self.Teff.value,
                                          time=self.time.to('d').value,
                                          dur=self.timeDur.to('d').value,
                                          cadence_hours=self.cadence.to('h').value,
                                          incl=incl)
        
        # print them to screen          
        if self.verbose > 1:
            print(f'B-V           : {params[0]:.3f} mag')
            print(f"log R'_HK     : {params[1]:.3f}")
            print(f'Activity rate : {params[2]:.3f} solar')
            print(f'Rot. period   : {params[3]:.3f} days')
            print(f'Min. period   : {params[4]:.3f} days')
            print(f'Max. period   : {params[5]:.3f} days')
            print(f'Cycle period  : {params[6]:.3f} years')
            print(f'Cycle overlap : {params[7]:.3f} years')
            print(f'Max. latitude : {params[8]:.3f} deg')
            print(f'Inclination   : {params[9]:.3f} deg')

        # Store global variables
        self.lc['spot'] = lc + 1
        self.df['B_V']       = params[0]
        self.df['logR_HK']   = params[1]
        self.df['AR_ARsun']  = params[2]
        self.df['Prot_day']  = params[3]
        self.df['Pmin_day']  = params[4]
        self.df['Pmax_day']  = params[5]
        self.df['Pcyc_year'] = params[6]
        self.df['Povl_year'] = params[7]
        self.df['Lmax_deg']  = params[8]
        self.df['I_deg']     = params[9]
        self.spot_coverage = area
        
        # Plot model
        if self.plot:
            model.plot(); plt.show()

            
    def solar_flares(self):

        """Model solar flares.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\nStellar flares\n')
        
        # Initialise model
        time  = self.time.to('d').value
        model = StellarFlares(time, scale=self.bol_coeff, seed=self.seed)
            
        # Select model
        
        if args.spot and self.spec in ['F','G','K','M']:
            if self.verbose > 1:
                print('Model generation : Daveport+2014')
                print('Model parameters : Doorsselaere+2017')
            if self.spot_coverage.sum() == 0:
                if self.verbose > 1:
                    print('Star is inactive (no spots), hence no flares..')
                return
            params = model.initDoorsselaere2017(self.spec, self.df.AR_ARsun,
                                                self.spot_coverage)
            
        elif args.flare == 'ToyModel':
            if self.verbose > 1:
                print('Model generation : Daveport+2014')
                print('Model parameters : Toy model')
            if self.verbose > 1 and not args.spot:
                errorcode('warning', 'Flares without stellar spots are unphysical..')
            params = model.initToyModel()

        else:
            return
            
        # print them to screen          
        if self.verbose > 1:
            print(f'Flaring rate     : {params[0]:.3f} events / quarter')
            print(f'Number of flares : {params[1]} events')
            
        # Return model
        lc, df = model.evaluate()

        # Store global variables
        self.lc['flare']   = lc
        self.df['R_flare'] = params[0]
        self.df['N_flare'] = params[1]
        
        # Plot model
        if self.plot: model.plot()

        
    #--------------------------------------------------------------#
    #                         OTHER PULSATORS                      #
    #--------------------------------------------------------------#
    

    def star_bcep(self):

        """Generate light curves for beta Cephei stars.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\nbeta Cephei pulsator\n')

        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=1.0, scale=self.scale_tess, seed=self.seed)

        if args.puls == 'Hey&Aerts2024':
            if self.verbose > 1:
                print('Selecting a mock object from TESS/Gaia sample (Hey&Aerts2024)')
            params = model.initFromFile(self.idir, args.puls, starID=self.starID)
            self.df['starname'] = params[0]

        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using TESS/Gaia sample (Hey&Aerts2024)')
            self.dm = model.initMockaHeyAerts2024(self.idir)            
            if self.verbose > 1:
                print(f'Number of pulsation modes : {self.dm.shape[0]}')
                print(f'Maximum mode amplitude    : {self.dm.ampl.max()*1e3:.1f} mmag')

        else:
            if self.verbose > 1:
                print('Generating mock object from toy model')
            model.initToyModel([3, 15], [0.01, 0.03])

        # Return model [mag -> ppm]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)


    def star_spb(self):

        """Generate light curve for a SPB star.

        Notes 
        -----
        This function uses the "varsouce.Pulsator" class.
        This class provide two model generations:
        1) Toy model using a characteristic power of 2.2
        2) Draw (freq, ampl, phase) from Kepler observations by
           using the flag "--puls pedersen2021".
        3) Create mock object using the Kepler sample, based on
           an analytic model and KDE histograms.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\nSPB pulsator\n')

        # Store correction
        self.df['A_corr'] = self.scale_kepler
            
        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=2.2, scale=self.scale_kepler, seed=self.seed)
        
        # Check model parsed
        
        if args.puls == 'Pedersen2021':
            if self.verbose > 1:
                print('Selecting a mock object from Kepler sample (Pedersen+2021)')            
            model.initFromFile(self.idir, args.puls, starID=self.starID)

        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using Kepler sample (Pedersen+2021)')
                
            params = model.initMockaPedersen2021(self.idir)
            self.df['N_modes']     = params[0]
            self.df['P0_day']      = params[1]
            self.df['DeltaP0_day'] = params[2]
            self.df['slope']       = params[3]
            self.df['Amax_mag']    = params[4]
            self.dm = params[5]
            
            if self.verbose > 1:
                print(f'Number of pulsation modes : {params[0]}')
                print(f'First period in pattern   : {params[1]:.5f} day')
                print(f'First period-spacing      : {params[2]:.5f} day')
                print(f'Slope of period-spacing   : {params[3]:.4f}')
                print(f'Maximum mode amplitude    : {params[4]*1e3:.3f} mmag')
            
        else:
            if self.verbose > 1:
                print('Generating mock object from toy model')            
            model.initToyModel([0.2, 3.2], [0.5, 2.5])

        # Return model [mag -> ppm]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)


    def star_dsct(self):

        """Generate light curves for delta-Scuti stars.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\ndelta Scuti pulsator\n')
            
        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=1.0, scale=self.scale_kepler, seed=self.seed)

        if args.puls == 'Bowman2018':
            if self.verbose > 1:
                print('Selecting a mock object from Kepler sample (Bowman+2018)')
            params = model.initFromFile(self.idir, args.puls, starID=self.starID)
            self.df['starname'] = params[0]

        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using Kepler sample (Bowman+2018)')
            self.dm = model.initMockaBowman2018(self.idir)            
            if self.verbose > 1:
                print(f'Number of pulsation modes : {self.dm.shape[0]}')
                print(f'Maximum mode amplitude    : {self.dm.ampl.max()*1e3:.1f} mmag')

        else:
            if self.verbose > 1:
                print('Generating mock object from toy model')
            model.initToyModel([5, 30], [0.01, 0.03])

        # Return model [mag -> pp1]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)
        

    def star_gdor(self):

        """Generate light curve for a gamma Doradus star.

        Notes 
        -----
        This function uses the "varsouce.Pulsator" class.
        This class provide two model generations:
        1) Toy model using a characteristic power of 2.2
        2) Draw (freq, ampl, phase) from Kepler observations by
           using the flag "--puls gang2020".
        3) Create mock object using the Kepler sample, based on
           an analytic model and KDE histograms.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\ngamma Doradus pulsator\n')

        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=2.2, scale=self.scale_kepler, seed=self.seed)
        
        # Check model parsed
        
        if args.puls == 'Gang2020':
            if self.verbose > 1:
                print('Selecting a mock object from Kepler sample (Gang+2020)')            
            model.initFromFile(self.idir, args.puls, starID=self.starID)

        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using Kepler sample (Gang+2020)')
                
            params = model.initMockaGang2020(self.idir)
            self.df['N_modes']     = params[0]
            self.df['P0_day']      = params[1]
            self.df['DeltaP0_day'] = params[2]
            self.df['slope']       = params[3]
            self.df['Amax_mag']    = params[4]
            self.dm = params[5]
            
            if self.verbose > 1:
                print(f'Number of pulsation modes : {params[0]}')
                print(f'First period in pattern   : {params[1]:.5f} day')
                print(f'First period-spacing      : {params[2]:.5f} day')
                print(f'Slope of period-spacing   : {params[3]:.4f}')
                print(f'Maximum mode amplitude    : {params[4]*1e3:.3f} mmag')
            
        else:
            if self.verbose > 1:
                print('Generating mock object from toy model')            
            model.initToyModel([0.5, 3], [0.5, 2.5])

        # Return model [mag -> ppm]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)

        
    def star_roap(self):

        """Generate light curve for roAp stars.

        This class is of BAF stars with so-called surface spots.
        """
        
        # Start script
        if self.verbose > 1:
            errorcode('module', '\nroAp variable pulsator\n')

        # Initialize class
        time  = self.time.to('d').value
        model = SurfaceModulations(time, self.scale_kepler, seed=self.seed)

        # Prepare model parameters
        params = model.initToyModel()
        if self.verbose > 1:
            print(f'Rotational period   : {round(params[0],3)} days')
            print(f'Random phase offset : {round(np.rad2deg(params[1]),1)} deg')
            print(f'Relative amplitude  : {round(params[2],3)}')
            print(f'Scaled amplitude    : {round(params[3],3)}')
        
        # Return model
        self.lc['puls'] = model.evaluate(plot=args.plot)
        self.df['Prot_day'] = params[0]
        self.df['dphi_rad'] = params[1]
        self.df['Arel']     = params[2]
        self.df['scale']    = params[3]

        
    #--------------------------------------------------------------#
    #                         EVOLVED STARS                        #
    #--------------------------------------------------------------#
        

    def star_rrlyr(self):

        """Generate ligth curve for RR Lyrae stars.

        This function uses precomputed models of RR Lyrae stars 
        to generate the light curve from their harmonics.
        """

        if self.verbose > 1:
            errorcode('module', '\nRR Lyrae pulsator\n')

        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=1, scale=self.scale_tess, seed=self.seed)
        
        # Check variable model parsed
        
        if args.puls == 'Bodi2023':
            if self.verbose > 1:
                print('Selecting mock object from TESS sample (Bodi+2023)')
            params = model.initFromFile(self.idir, sample=args.puls, variable='RRLyr')
            self.df['starname'] = params[0]
            
        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using TESS sample (Bodi+2023)')
            params = model.initMockaBodi2023(self.idir, variable='RRLyr')
            self.df['starname'] = params[0]
            self.df['f_corr']   = params[1]
            self.df['A_corr']   = params[2]
            self.dm             = params[3]

        # Return model [mag -> flux]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)
        

    def star_ceph(self):

        """Generate ligth curve for Cepheid stars.
        """

        if self.verbose > 1:
            errorcode('module', '\nCepheid pulsator\n')

        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, 1, self.scale_tess, self.seed)
        
        # Check variable model parsed

        if args.puls == 'Bodi2023':
            if self.verbose > 1:
                print('Selecting mock object from TESS sample (Bodi+2023)')
            params = model.initFromFile(self.idir, sample=args.puls, variable='Ceph')
            self.df['starname'] = params[0]
            
        elif args.puls == 'mocka':
            if self.verbose > 1:
                print('Generating mock object using TESS sample (Bodi+2023)')
            params = model.initMockaBodi2023(self.idir, variable='Ceph')
            self.df['starname'] = params[0]
            self.df['f_corr']   = params[1]
            self.df['A_corr']   = params[2]
            self.dm             = params[3]
                        
        # Return model [mag -> flux]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)

        
    def star_lpv(self):

        """Generate light curves for LPV stars.
        """

        # Start script
        if self.verbose > 1:
            errorcode('module', '\nLong Period Variable (LPV)\n')

        # Initialize and prepare model input
        time  = self.time.to('d').value
        model = Pulsator(time, power=1, scale=None, seed=self.seed)
        
        # Check model parsed        
        if self.verbose > 1:
            print('Generating mock object using OGLE sample')

        params = model.initMockaLPV(self.idir, startype=None)
        self.df['star_type']   = params[0]
        self.df['star_row_id'] = params[1]
        self.dm = params[2]
            
        if self.verbose > 1:
            print(f'Type of variable : {params[0]}')
            
        # Return model [mag -> ppm]
        mag = model.evaluate(plot=args.plot)
        self.lc['puls'] = ut.fromMagToFlux(mag)
        
                
    #--------------------------------------------------------------#
    #                          BINARY SYSTEMS                      #
    #--------------------------------------------------------------#
    
    def binary_source(self):
        """Select the stellar paramters of a binary system.
        """
        if self.verbose > 1:
            errorcode('module', '\nBinary sources\n')

            
    def binary_eb(self):
        """Function to generate a Eclipsing Binary (EB) light curve.
        """
        if self.verbose > 1:
            errorcode('module', '\nEclipsing binary\n')

        # Set the stellar source entry
        self.star_source = 'EB'
            
        # Fetch time array
        time  = self.time.to('d').value
        model = EclipsingBinary(time, seed=self.seed, verbose=self.verbose)

        # Fetch model parameters
        if self.verbose > 1:
            print('Selecting mock object from Kepler sample (IJspeert+2021)')
        params = model.initIJspeert2023(self.idir, starID=None)
        self.df['starname'] = params[0]
        self.df['P_day']    = params[1]
        if self.verbose > 1 and params[1] is not None:
            print(f'Orbital period : {params[1]:.3f} day')

        # Return model [mag -> flux]
        mag = model.evaluate(plot=self.plot)
        self.lc['flux'] = ut.fromMagToFlux(mag)


    #--------------------------------------------------------------#
    #                          SMBH BINARY                         #
    #--------------------------------------------------------------#

    def mode_smbhb(self):
        """Given SMBH binary properties asign variable signal.
        """
        if self.verbose > 1:
            errorcode('module', '\nSMBH binary\n')        
        self.smbhb_source()
        self.smbhb_model()
        self.run_prolog()

        
    def load_smbhb(self, source):
        """Module to load benchmark exoplanets to be used by varsim.
        """
        if source == 'Spikey':
            # From Hu+2020
            params = [
                0.918,  # z
                0.6528, # t0 [day]
                1.144,  # P  [day]
                81.95,  # i  [deg]
                0.524,  # e  [0,1]
                84.63,  # w  [deg]
                7.4,    # logM1 [logM_sun]
                6.7,    # logM2 [logM_sun]
                0.89,   # L  [0,1]
                2.09,   # alpha
                0.0,    # vz [-c,c]
                31,     # tau [day]
                9.25    # sigma [ppt]
            ]
        return params


    def smbhb_source(self):
        """Select the stellar paramters.
        """
        # If stellar parameters are parsed
        if self.smbhb_params:
            self.smbhb = 'SMBHB'
            params = self.smbhb_params[0]
        else:
            try:
                params = self.load_smbhb(self.smbhb)
            except UnboundLocalError:
                errorcode('error', 'No SMBHB with that name! Check --smbhb entry')

        # Store parameters
        self.df['z']            = params[0]
        self.df['t0_yr']        = params[1]
        self.df['P_yr']         = params[2]
        self.df['i_deg']        = params[3]
        self.df['e']            = params[4]
        self.df['w_deg']        = params[5]
        self.df['logM1_logMsun'] = params[6]
        self.df['logM2_logMsun'] = params[7]
        self.df['L']            = params[8]
        self.df['alpha']        = params[9]
        self.df['vz_c']         = params[10]
        self.df['tau_day']      = params[11]
        self.df['sigma_ppt']    = params[12]
        self.df['seed']         = self.seed

        # Print available stellar model parameters
        if self.verbose > 1:
            print(f"Source name             : {self.smbhb}")
            print(f"Redshift, z             : {self.df.z:.3f}")
            print(f"Time of ephemeris, t0   : {self.df.t0_yr:.3f} yr")
            print(f"Orbital period, P       : {self.df.P_yr:.3f} yr")
            print(f"Inclination, i          : {self.df.i_deg:.3f} deg")
            print(f"Eccentricity, e         : {self.df.e:.3f}")
            print(f"Argument of periapse, w : {self.df.w_deg:.3f} deg")
            print(f"Primary mass, logM1     : {self.df.logM1_logMsun:.3f} logMsun")
            print(f"Secondary mass, logM2   : {self.df.logM2_logMsun:.3f} logMsun")
            print(f"Luminosity fraction, L  : {self.df.L:.3f}")
            print(f"Spectral index, alpha   : {self.df.alpha:.3f}")
            print(f"Proper motion, vz       : {self.df.vz_c:.3f} c")
            print(f"DRW time scale, tau     : {self.df.tau_day:.3f} day")
            print(f"DRW amplitude, sigma    : {self.df.sigma_ppt:.3f} ppt")
            
    
    def smbhb_model(self):
        """Function to generate a SMBH binary light curve.

        A Super Massive Black Hole (SMBH) binary system consist of several 
        components, for which this model includes the effects:
          1) The doppler boosting
          2) The gravitational self-lensing effect
          3) The stochastic quasar variability
        """            
        # Fetch time array
        time = self.time.to('d').value

        # Initialise parameters
        params = smbhb.model_params()
        params.z     = self.df['z']
        params.t0    = self.df['t0_yr']
        params.P     = self.df['P_yr']
        params.i     = self.df['i_deg']
        params.e     = self.df['e']
        params.w     = self.df['w_deg']
        params.logM1 = self.df['logM1_logMsun']
        params.logM2 = self.df['logM2_logMsun']
        params.L     = self.df['L']
        params.alpha = self.df['alpha']
        params.vz    = self.df['vz_c']
        params.tau   = self.df['tau_day']
        params.sigma = self.df['sigma_ppt']
        params.seed  = self.seed

        # Generate model
        model = smbhb.model(params)
        self.lc = model.light_curve(time, df=True)

        
    #--------------------------------------------------------------#
    #                          PLANET MODELS                       #
    #--------------------------------------------------------------#
                
    def ldc(self):

        """Compute the Limb Darkening (LD) coefficients.

        This module uses the code: LDTk        
        """

        if self.verbose > 1:
            print('\nComputing limb darkening coefficients with LDTk')
            
        # Convert input parameters
        wvl_tele  = self.wvl_tele.value  # [nm]
        tran_tele = self.tra_tele        # [0-1]

        # Interpolate (piecewise cubic) into higher resolution grid
        grid_no  = 1000
        wvl_int  = np.linspace(wvl_tele[0], wvl_tele[-1], grid_no)
        passband = make_interp_spline(wvl_tele, tran_tele, k=3)
        tran_int = passband(wvl_int)

        # Create passband object
        filters = [TabulatedFilter('plato', wvl_int, tran_int)]

        # Create instance of class
        # NOTE uncertainties are vital for the software to work!
        sc = LDPSetCreator(teff=(self.Teff.value, 50),
                           logg=(self.logg, 0.20),
                           z=(self.Z, 0.05),
                           filters=filters)

        # Create the limb darkening profiles
        ps = sc.create_profiles()

        # Determine law coefficients
        # NOTE we only allow 2-term models
        try:
            if self.ldm == 'linear':
                u, _ = ps.coeffs_ln(do_mc=True)
            elif self.ldm == 'quadratic':
                u, _ = ps.coeffs_qd(do_mc=True)
            elif self.ldm == 'squareroot':
                u, _ = ps.coeffs_sq(do_mc=True)
            elif self.ldm == 'power2':
                u, _ = ps.coeffs_p2(do_mc=True)
        except:
            if self.ldm == 'linear':
                self.ldc = [0.570]
            elif self.ldm == 'quadratic':
                self.ldc = [0.470, 0.153]
            elif self.ldm == 'squareroot':
                self.ldc = [0.334, 0.364]
            elif self.ldm == 'power2':
                self.ldc = [0.670, 0.759]
            errorcode('warning', 'LD coefficients failed for ' +
                      f'(Teff, logg, Z) = ({self.Teff}, {self.logg}, {self.Z}')
        else:
            self.ldc = u[0]

        # Store parameters
        if len(self.ldc) == 1:
            self.df['u1'] = self.ldc[0]
            if self.verbose > 1:
                print(f"LD {self.ldm} coefficients : {self.ldc[0]:.3f}")
        else:
            self.df['u1'] = self.ldc[0]
            self.df['u2'] = self.ldc[1]
            if self.verbose > 1:
                print(f"LD {self.ldm} coefficients : {self.ldc[0]:.3f}, {self.ldc[1]:.3f}")

            
    def planet_model(self):

        """Calculation of exoplanet model parameters.

        Resources
        ---------
        Equations are from chapters Seager et al. (2010) "Exoplanets":
        Winn (2014)             : https://arxiv.org/pdf/1001.2010.pdf
        Murray & Correia (2011) : https://arxiv.org/abs/1009.1738v2
        
        Notes
        -----
        Both "a" and "Rs" are given in units of Rstar.

        Assumptions
        -----------
        - The calculations in this code block are under the following assumptions that
          that eclipses are centered around conjunction. This is not valid for extremely
          eccentric and close-in orbits with grazing eclipses. However, non-grazing close
          in orbits are still valid.
        - The time seperation between transit and occultation in the following is a
          first order approximation in "e" by integrating "dt/dF".
        """
        
        # Stellar parameters
        time = self.time.to('d')
        Ms   = self.M.to('kg')
        Rs   = self.R.to('m')
        Teff = self.Teff.to('K')
        
        # LOAD PLANET MODEL
        
        if self.kul20:

            # Select benchmark planets
            dex = self.rng.integers(low=0, high=2, size=1)[0]
            name_benchmark = ['Earth-like', 'Neptune-like', 'Jupiter-like']
            args.planet = name_benchmark[dex]
            Rp_benchmark = [1.0, 3.9, 11.2]
            Rp = Rp_benchmark[dex] * u.R_earth
            
            # Log-normal distribution of planet radii 
            #Rp = np.random.lognormal(5., 1.) * u.R_earth
            #if Rp.to('R_jup').value < 0.05: Rp = 0.05 * u.R_jup
            #if Rp.to('R_jup').value > 8.50: Rp = 1.80 * u.R_jup

            # Draw mass from M-R forecaster
            # Paper: Chen, J., & Kipping, D. M. 2018, MNRAS, 473, 2753
            # Code: https://github.com/chenjj2/forecaster
            # NOTE forecasting only valid between (0.05 < Rp/R_jup < 8.5) 
            if self.verbose > 1:
                classifier = 'yes'
            else:
                classifier = 'no'
            mr = PlanetMRforecast()
            Mp, _, _ = mr.Rstat2M(mean=Rp.to('R_jup').value, unit='Jupiter',
                                  std=0.01, sample_size=1000, grid_size=1000,
                                  classify=classifier)
            Mp = Mp * u.M_jup

            # Select random uniform period
            P  = np.random.uniform(0, 1) * u.yr

            # Make sure at least one transit fall within the duration of the observation
            t0 = np.random.uniform(0, self.timeDur.to('yr').value) * u.yr
            if P <= self.timeDur.to('yr'):
                t0 = t0 - self.timeDur.to('yr')

            # Simple for now
            e = 0

            # Unbiased uniform distribution [85-90 deg]
            #i = np.arccos(np.random.uniform(0, 90/85-1)) * 180/np.pi * u.deg 
            i = 90 * u.deg
            
            # Uniform orientation
            w = np.random.uniform(0, 360) * u.deg

            # Convert units 
            t0 = t0.to('s')
            P  = P.to('s')
            i  = i.to('rad')
            w  = w.to('rad')
            Rp = Rp.to('m')
            Mp = Mp.to('kg')
        
        elif args.planet_params is None:

            # Load exoplanet parameters [SI units]
            try:
                params = self.load_exoplanet(args.planet)
            except UnboundLocalError:
                errorcode('error', 'No planet with that name! Check --planet entry')
            else:
                t0 = params['t0'].to('s')
                P  = params['P'].to('s')
                e  = params['e']
                i  = params['i'].to('rad')
                w  = params['w'].to('rad')
                Rp = params['rp'].to('m')
                Mp = params['mp'].to('kg')
                xi = params['xi']
                Tn = params['Tn'].to('K').value
                dT = params['dT'].to('K').value
            
        else:

            # Load exoplanet parameters [SI units]
            params = args.planet_params[0]
            t0 = (params[0] * u.d).to('s')
            P  = (params[1] * u.d).to('s')
            e  = params[2]
            i  = (params[3] * u.deg).to('rad')
            w  = (params[4] * u.deg).to('rad')
            Rp = (params[5] * u.R_earth).to('m')
            Mp = (params[6] * u.M_earth).to('kg')
            xi = 0.
            Tn = 0.
            dT = 0.
            
        # ORBITAL DYNAMICS

        # Handy definitions
        x = np.sqrt(1 - e**2)   # Optimization constant
        k = Rp/Rs               # Radius constant

        # Semi-major axis (K3)
        a = (c.G * P**2 * (Ms + Mp) / (4*np.pi**2))**(1/3.)

        # Impact parameter: Winn (2014) Eq. 7 & 8
        b_tra = a*np.cos(i)/Rs * (1 - e**2)/(1 + e*np.sin(w))
        b_occ = a*np.cos(i)/Rs * (1 - e**2)/(1 - e*np.sin(w))

        # Transit and occultation times: Winn (2014) Eq. 14, 15 & 16
        ep = x/(1 + e*np.sin(w)) / u.rad
        em = x/(1 - e*np.sin(w)) / u.rad
        t_tra_tot = P/np.pi * np.arcsin(Rs/a * np.sqrt((1 + k)**2 - b_tra**2)/np.sin(i)) * ep
        t_tra_ful = P/np.pi * np.arcsin(Rs/a * np.sqrt((1 - k)**2 - b_tra**2)/np.sin(i)) * ep
        t_occ_tot = P/np.pi * np.arcsin(Rs/a * np.sqrt((1 + k)**2 - b_occ**2)/np.sin(i)) * em
        t_occ_ful = P/np.pi * np.arcsin(Rs/a * np.sqrt((1 - k)**2 - b_occ**2)/np.sin(i)) * em
        tau_tra = (t_tra_tot - t_tra_ful)/2.
        tau_occ = (t_occ_tot - t_occ_ful)/2.

        # Transits to occultations time seperation: Winn (2014) Eq. 33
        dt_c = P/2 * (1 + 4*e*np.cos(w)/np.pi)

        # Time of first (t0) central passage transit and occultation
        t0_tra_cen = t0
        t0_occ_cen = t0 + dt_c

        # Check for model applicability
        if b_tra > 1 - Rp/Rs and b_tra <= 1 + Rp/Rs:
            errorcode('warning', 'Planetary model consist of grazing eclipses!')
        elif b_tra > 1 + Rp/Rs:
            errorcode('warning', 'Planet model do not have any physical eclipses!')
            
        # Show parameters apce
        if self.verbose > 1:
            errorcode('module', '\nPlanet eclipse model')

        # Calculate LD coefficients
        self.ldc()

        # Show parameters apce
        if self.verbose > 1:
            print('')
            print("Planet name            : {}".format(args.planet))
            print("Planet mass            : {:.2f}".format(Mp.to('M_earth')))
            print("Planet radius          : {:.2f}".format(Rp.to('R_earth')))
            print('')
            print("Semimajor axis         : {:.2f} starRad".format(a.to('m')/Rs.to('m')))
            print("Eccentricity           : {:.3f}".format(e))
            print("Inclination            : {:.2f}".format(i.to('deg')))
            print("Argument of periastron : {:.2f}".format(w.to('deg')))
            print('')
            print("Time of emphemeris     : {:.2f}".format(t0.to('d')))
            print("Orbital period         : {:.2f}".format(P.to('d')))
            print("Transit-to-Occult time : {:.3f}".format(dt_c.to('d')))
            print('')
            print("Total transit duration : {:.3f}".format(t_tra_tot.to('h')))
            print("Full  transit duration : {:.3f}".format(t_tra_ful.to('h')))
            print("In/Eg transit duration : {:.3f}".format(tau_tra.to('min')))
            print("Impact parameter (tra) : {:.3f}".format(b_tra))
            print('')
            print("Total occult. duration : {:.3f}".format(t_occ_tot.to('h')))
            print("Full  occult. duration : {:.3f}".format(t_occ_ful.to('h')))
            print("In/Eg occult. duration : {:.3f}".format(tau_occ.to('min')))
            print("Impact parameter (occ) : {:.3f}\n".format(b_occ))

        # Parameters for parameterization file
        self.df['Mp_Mearth'] = Mp.to('M_earth').value
        self.df['Rp_Rearth'] = Rp.to('R_earth').value
        self.df['a_Rstar']   = (a.to('R_sun')/Rs).value
        self.df['P_day']     = P.to('d').value
        self.df['t0_day']    = t0.to('d').value
        self.df['e']         = e
        self.df['i_deg']     = i.to('deg').value
        self.df['w_deg']     = w.to('deg').value

        # Store parameters
        self.Mp = Mp
        self.Rp = Rp
        self.t0 = t0
        self.P  = P
        self.a  = a
        self.e  = e
        self.i  = i
        self.w  = w
        self.t_tra_tot  = t_tra_tot
        self.t_occ_tot  = t_tra_tot
        self.t0_tra_cen = t0_tra_cen
        self.t0_occ_cen = t0_occ_cen
        self.dt_c = dt_c
        self.b_tra = float(b_tra)
        
            
    def planet_transit(self):

        """Model exoplanet transits.

        In the following the exoplanet transits are being modelled with Batman:
        https://lweb.cfa.harvard.edu/~lkreidberg/batman/quickstart.html

        NOTE: t0 and P can principly be anything as long as they are consistant. 
        Here we make sure to use consistent reference time unit.
        """

        # Initialize batman model
        batman_params = batman.TransitParams()
        batman_params.limb_dark = self.ldm
        batman_params.u   = self.ldc
        batman_params.t0  = self.t0.to('d').value
        batman_params.per = self.P.to('d').value
        batman_params.a   = (self.a.to('m')/self.R.to('m')).value
        batman_params.ecc = self.e
        batman_params.inc = self.i.to('deg').value
        batman_params.w   = self.w.to('deg').value
        batman_params.rp  = (self.Rp.to('m')/self.R.to('m')).value

        # Model parameters for eclipse
        #batman_params.fp          = 0.001
        #batman_params.t_secondary = 0.5
        
        # Initializes transit model and extract light curve [ppm]
        model = batman.TransitModel(batman_params, self.time.value) #,transittype='secondary')
        self.lc['tran'] = model.light_curve(batman_params)

        # True anomaly at each time: This will be used in our custom models later
        self.nu = model.get_true_anomaly()

        # Time of periastron passage (calculated from t0)
        self.tau = model.get_t_periastron(batman_params)

        # Print to bash
        if self.verbose > 1:
            depth = np.abs(np.min((self.lc.tran-1)*1e6))
            print(f"Mid-transit depth  : {depth:.1f} ppm")


    def planet_occultation(self):
        """
        TODO Module do not work anymore! Spiderman has now been integrated into
        the batman package!

        In the following the exoplanet transits are being modelled with Batman:
        https://spiderman.readthedocs.io/en/latest/index.html

        Spiderman models the secondary eclipses plus phase curves, however, not the
        transit hence Batman is needed still for this. The two codes are compatible
        in terms of parameter definitions, as they are developed in collaboration.

        NOTE the unit of "t0" and "P" can in principly be anything as long as they
        are consistant. Here we make sure to use consistent reference time unit.
        NOTE "n_layer" is the number of layers in the 2D "web" defining the integration
        grid. A value of 20 give an error less than 0.1 ppm.
        """

        # TODO small fix until module is tested again!
        self.lc['occu'] = np.ones(len(self.time))
        return

        
        # Initialize model and assign parameters
        import spiderman

        # TODO brightness model should be loaded from file by user
        exo_brightness_model = 'zhang'

        # TODO The phase curve is now calculated using a blackbody model but it is also
        # possible to use PHOENIX med-res spectra. This not work now - fix it!
        spider_params = spiderman.ModelParams(brightness_model=exo_brightness_model,
                                              stellar_model='PHOENIX')
                                              #stellar_model= datapath + 'stellar_model.txt')
        spider_params.t0    = t0.to('d').value
        spider_params.per   = P.to('d').value
        spider_params.a     = (a/Rs).value
        spider_params.a_abs = a.to('AU').value
        spider_params.ecc   = e
        spider_params.inc   = i.to('deg').value
        spider_params.w     = w.to('deg').value
        spider_params.rp    = (Rp/Rs).value
        spider_params.p_u1  = ldc[0]
        spider_params.p_u2  = ldc[1]
        spider_params.T_s   = Teff.value
        spider_params.l1    = wvl_tele[0].to('m').value
        spider_params.l2    = wvl_tele[-1].to('m').value
        spider_params.n_layer = 100  # The number of layers in the 2D "web"

        # TODO convolve with the acual bandpass!
        # Use spiderman's weithing scheme with the instrument response

        spider_params.filter = os.getcwd() + '/data/Passbands/response_plato_spiderman.txt'

        # USE USER DEFINED BRIGHTNESS MODELS

        # TODO Verify all the available brigthness models

        # Spherical hamonics: Non-physical model
        # NOTE there is nothing in the implementation to prevent negative surface fluxes!

        if exo_brightness_model == 'spherical':
            spider_params.degree = degree
            spider_params.la0    = la0
            spider_params.lo0    = lo0
            spider_params.sph    = sph

        # Zhang and Showman (2017): http://adsabs.harvard.edu/abs/2017ApJ…836…73Z
        # A temperature map model based on semi-physical reproducing the main features
        # of hot Jupiter phase-curves - offset hotspots. Called with “zhang”

        if exo_brightness_model == 'zhang':
            spider_params.xi      = xi   # Ratio of radiative to advective timescale
            spider_params.T_n     = Tn   # Temperature of nightside
            spider_params.delta_T = dT   # Day-night temperature contrast

        # Offset hotspot and Two sided planet:

        if (exo_brightness_model == 'hotspot_b' or exo_brightness_model == 'hotspot_t'):
            spider_params.la0  = la0
            spider_params.lo0  = lo0
            spider_params.size = size
            spider_params.grid_size = grid_size

            if exo_brightness_model == 'hotspot_b':

                if spot_b is not None and p_b is not None:
                    spider_params.spot_b = spot_b
                    spider_params.p_b    = p_b
                if pb_d is not None and pb_n is not None:
                    spider_params.pb_d = pb_d
                    spider_params.pb_n = pb_n

            if exo_brightness_model == 'hotspot_t':
                spider_params.spot_T = spot_T
                spider_params.p_T    = p_T

        # Two sided planet:
        if exo_brightness_model == 'spherica':
            spider_params.degree = degree
            spider_params.la0    = la0
            spider_params.lo0    = lo0
            spider_params.sph    = sph

        # Contruct model light curve [ppm]
        lc_occ = (spider_params.lightcurve(time.value) - 1) * 1e6
        lc_occu = np.zeros(len(self.time))
        self.lc['occu'] = lc_occu.tolist()

        
    def planet_beaming(self):

        """Doppler beaming model.

        TODO Implement into class
        """

        # Central wavelength of PLATO bandpass [m]
        wvl_c = self.wvl_tele[np.argmax(self.tra_tele)]

        # Initialize and prepare model input
        model_beam  = DopplerBeaming()
        params_beam = {'wvl_c': wvl_c,
                       'Teff': self.Teff,
                       't0': self.t0,
                       'P': self.P,
                       'a': self.a,
                       'e': self.e,
                       'i': self.i,
                       'w': self.i,
                       'Mp': self.Mp,
                       'Ms': self.M}

        # Assign and calculate model
        model_beam.assignValue(params_beam)
        lc_beam, self.A_beam = model_beam.evaluate(self.time, self.nu)
        self.lc['beam'] = lc_beam / 1e6 + 1


    def planet_ellipsoidal(self):

        """Model ellipsoidal distortion.

        TODO Implement into class
        """

        # Initialize and prepare model input
        model_elli  = EllipsoidalDistortion()
        params_elli = {'g': 0.05,
                       'u': self.ldc[0],
                       't0': self.t0,
                       'P': self.P,
                       'e': self.e,
                       'i': self.i,
                       'w': self.w,
                       'a': self.a,
                       'Rs': self.R,
                       'Mp': self.Mp,
                       'Ms': self.M}

        # Assign and calculate model
        model_elli.assignValue(params_elli)
        lc_elli, self.A_elli = model_elli.evaluate(self.time, self.nu)
        self.lc['elli'] = lc_elli / 1e6 + 1

        
    def plot_phase_curve(self):

        # Plot exoplanet model
        if (self.time[-1] >= self.P.to('d') + self.t0.to('d')):
            fig, ax = pt.plotOrbitalPhaseCurve(self.time.value,
                                               self.lc['tran'].to_numpy(),
                                               self.lc['occu'].to_numpy(),
                                               self.lc['beam'].to_numpy(),
                                               self.lc['elli'].to_numpy(),
                                               self.t0.to('d').value,
                                               self.P.to('d').value,
                                               self.dt_c.to('d').value,
                                               self.t0_tra_cen.to('d').value,
                                               self.t_tra_tot.to('d').value,
                                               self.t0_occ_cen.to('d').value,
                                               self.t_occ_tot.to('d').value,
                                               self.A_beam, self.A_elli)
            plt.show()
            
        elif (self.time[-1] < self.P.to('d') + self.t0.to('d')):
            errorcode('warning',
                      'No phase plot, time series is shorter than the orbital period!')


    def moon_transit(self):

        """Model exomoon transits.

        In the following the exomoon transits are being modelled with Pandora:
        https://github.com/hippke/Pandora

        NOTE: t0 and P can principly be anything as long as they are consistant. 
        Here we make sure to use consistent reference time unit.

        FIXME This module are under construction!
        """

        import pandoramoon as pandora
        params = pandora.model_params()

        #time = self.time.to('d')

        # if args.moon_params is None:

        #     # Load exoplanet parameters [SI units]
        #     try:
        #         params = self.load_exoplanet(args.planet)
        #     except UnboundLocalError:
        #         errorcode('error', 'No planet with that name! Check --planet entry')
        #     else:
        #         t0 = params['t0'].to('s')
        #         P  = params['P'].to('s')
        #         e  = params['e']
        #         i  = params['i'].to('rad')
        #         w  = params['w'].to('rad')
        #         Rp = params['rp'].to('m')
        #         Mp = params['mp'].to('kg')
        #         xi = params['xi']
        #         Tn = params['Tn'].to('K').value
        #         dT = params['dT'].to('K').value
            
        # else:

            # # Load exoplanet parameters [SI units]
            # params = args.planet_params[0]
            # t0 = (params[0] * u.d).to('s')
            # P  = (params[1] * u.d).to('s')
            # e  = params[2]
            # i  = (params[3] * u.deg).to('rad')
            # w  = (params[4] * u.deg).to('rad')
            # Rp = (params[5] * u.R_earth).to('m')
            # Mp = (params[6] * u.M_earth).to('kg')
            # xi = 0.
            # Tn = 0.
            # dT = 0.
        
        # Stellar parameters
        params.R_star = self.R.to('m').value
        params.u1     = self.ldc[0]
        params.u2     = self.ldc[1]
        
        # Planet parameters
        params.per_bary       = self.P.to('d').value
        params.a_bary         = self.a.to('m').value / self.R.to('m').value
        params.r_planet       = self.Rp.to('m').value / self.R.to('m').value
        params.b_bary         = self.b_tra
        params.t0_bary        = self.t0.to('d').value
        params.t0_bary_offset = 0  # [days]
        params.M_planet       = self.Mp.to('kg').value
        params.w_bary         = self.w.to('deg').value
        params.ecc_bary       = self.e
        
        # Moon parameters
        params_moon = self.load_exomoon(args.moon)
        print(params_moon)
        params.r_moon = params_moon['R'].to('m').value / self.R.to('m').value
        params.M_moon = 0.05395 * params.M_planet  # [0..1]
        params.per_moon = 0.3 # [days]
        params.tau_moon = 0.07  # [0..1]
        params.Omega_moon = 0  # [0..180]
        params.i_moon = 80  # [0..180]
        params.e_moon = 0.9  # [0..1]
        params.w_moon = 20  # [deg]
        

        # Other model paramters
        params.epochs = 3  # [int]
        params.epoch_duration = 0.6  # [days]
        params.cadences_per_day = 250  # [int]
        params.epoch_distance = 365.26   # [days]
        params.supersampling_factor = 1  # [int]
        params.occult_small_threshold = 0.1  # [0..1]
        params.hill_sphere_threshold = 1.2

        time = pandora.time(params).grid()

        model = pandora.moon_model(params)

        flux_total, flux_planet, flux_moon = model.light_curve(time)

        noise_level = 100e-6  # Gaussian noise to be added to the generated data
        noise = np.random.normal(0, noise_level, len(time))
        testdata = noise + flux_total
        yerr = np.full(len(testdata), noise_level)

        plt.figure()
        plt.plot(time, flux_planet, color="blue")
        plt.plot(time, flux_moon, color="red")
        plt.plot(time, flux_total, color="black")
        plt.scatter(time, testdata, color="black", s=0.5)
        plt.xlabel("Time (days)")
        plt.ylabel("Relative flux")
        plt.xlim(min(time), min(time)+params.epoch_duration)
        plt.show()
        exit()
        
    #--------------------------------------------------------------#
    #                      PROLOGUE AND SAVING                     #
    #--------------------------------------------------------------#
        

    def run_prolog(self):
        
        if self.verbose > 1:
            errorcode('module', '\nPrologue\n')

        # SORT LIGHT CURVE

        # Variability classes
        stars    = ['bCep', 'SPB', 'dSct', 'gDor', 'roAp',
                    'RRLyr', 'Ceph', 'V361Hya', 'ZZCeti', 'LPV']
        binaries = ['EB', 'SMBH']

        # If all signals are ignored then it is a constant star
        if ((args.gran is False or args.puls is False) and
            args.spot is False and args.flare is False and
            args.planet_params is False):
            self.star = 'constant'

        elif args.smbhb or args.smbhb_params:
            fig, ax = smbhb.plot_model(self.lc)
            if self.plot: plt.show()
            self.lc.time *= 86400

        # Combine all signals any other variable source
        else:
            # Empty array to store flux [pp1]
            self.lc['flux'] = np.ones(len(self.lc.time))

            # Stellar variability
            if 'gran' in self.lc:
                self.lc['flux'] += (self.lc.gran - 1)
            if 'puls' in self.lc:
                self.lc['flux'] += (self.lc.puls - 1)
            if 'spot' in self.lc:
                self.lc['flux'] += (self.lc.spot - 1)
            if 'flare' in self.lc:
                self.lc['flux'] += (self.lc.flare - 1)

            # Planet/binary variability
            if 'tran' in self.lc:
                flux_planet = self.lc.tran
                if 'occu' in self.lc:
                    flux_planet += (self.lc.occu - 1)
                if 'beam' in self.lc:
                    flux_planet += (self.lc.beam - 1)
                if 'elli' in self.lc:
                    flux_planet += (self.lc.elli - 1)

                # Spots and transits are multiplicative
                self.lc['flux'] *= flux_planet

            # Plot combined light curve
            fig, ax = pt.plotVarsimLC(self.lc)
            if self.plot and self.lc.shape[1] > 3: plt.show()
                
        # SAVE DATA

        if self.ofile:

            # Filenames
            ofile_parameters = self.ofile.parents[0] / f'{self.ofile.stem}_parameters.ftr'
            ofile_components = self.ofile.parents[0] / f'{self.ofile.stem}_components.ftr'
            ofile_pulsations = self.ofile.parents[0] / f'{self.ofile.stem}_pulsations.ftr'
            ofile_lightcurve = self.ofile.parents[0] / f'{self.ofile.stem}_lightcurve.png'
            
            # Save light curve [pp1 -> mag]
            df = self.lc.flux.to_numpy()
            dm = -2.5 * np.log10(df)                        
            if self.verbose > 1:
                print(f'Saving file : {self.ofile}')
            data = np.transpose([self.lc['time'], dm])
            np.savetxt(self.ofile, data, fmt=['%.1f', '%.8f'])

            # Save parameter space
            if self.verbose > 1:
                print(f'Saving file : {ofile_parameters}')
            self.df = self.df.to_frame().T
            self.df = self.df.reset_index(drop=True)
            self.df.to_feather(ofile_parameters)

            # Save components (if multiple)
            if self.lc.shape[1] > 2:
                if self.verbose > 1:
                    print(f'Saving file : {ofile_components}')                
                self.lc.to_feather(ofile_components)
            
            # Save pulsation modes for MOCKA
            if args.puls == 'mocka':
                try:
                    self.dm
                except AttributeError:
                    return
                else:
                    if self.verbose > 1:
                        print(f'Saving file : {ofile_pulsations}')                
                    self.dm.to_feather(ofile_pulsations)

            # Save final plot
            if self.verbose > 1:
                print(f'Saving file : {ofile_lightcurve}')            
            fig.savefig(ofile_lightcurve, bbox_inches='tight', dpi=200)

            
    #--------------------------------------------------------------#
    #                         SOFTWARE MODES                       #
    #--------------------------------------------------------------#


    def mode_single(self):

        """Given stellar properties asign variable signal.
        """
        
        # Select star
        self.stellar_source()

        # Bolometric correction
        self.stellar_spectrum()
        
        # Include stellar variability

        if args.star == 'bCep':
            v.star_bcep()
        
        elif args.star == 'SPB':
            v.star_spb()

        elif args.star == 'dSct':
            v.star_dsct()

        elif args.star == 'gDor':
            v.star_gdor()
                        
        elif args.star == 'roAp':
            v.star_roap()
            
        elif args.star == 'RRLyr':
            v.star_rrlyr()
            
        elif args.star == 'Ceph':
            v.star_ceph()

        elif args.star == 'LPV':
            v.star_lpv()

        elif args.star == 'V361Hya':
            v.star_v361hya()            
            
        elif args.star == 'ZZCeti':
            v.star_zzceti()            
            
        else:
            # Constant star
            if args.star == 'constant':
                pass

            # Solar-like stars
            elif args.star or args.star_params:
                if args.gran and args.puls:
                    v.solar_granosc()
                if args.spot:
                    v.solar_spots()
                if args.flare is not False:
                    v.solar_flares()
                    
        # Include exoplanet
        if args.planet or args.planet_params:
            v.planet_model()
            v.planet_transit()
            v.planet_occultation()
            v.planet_beaming()
            v.planet_ellipsoidal()
            if args.plot and args.kul20 is None:
                v.plot_phase_curve()

            # Include exomoon
            if args.moon:
                v.moon_transit()
                    
        # Combine and save
        self.run_prolog()

        
    def mode_binary(self):

        """Given stellar properties asign variable signal.
        """
        
        # Select binary system
        #self.binary_source()

        # Bolometric correction
        #self.stellar_spectrum()

        if args.binary == 'EB':
            v.binary_eb()
        
        elif args.binary == 'SMBH':
            v.binary_smbh()

        # Combine and save
        self.run_prolog()

        
    def mode_kul20(self):

        """Mode designed for KUL20 -> Called by "--kul20 <int>".
        """

        # Meaning of integer parsed:
        # x -> Constant star for any other than [0, 1, 2, 3]
        # 0 -> Std star (roAp with 2 hamonics)
        # 1 -> Gran, Puls
        # 2 -> Gran, Puls, Spots
        # 3 -> Gran, Puls, Spots, Transit
        
        if args.kul20 == 0:
            args.star = 'roAp'

        if args.kul20 in [1, 2, 3]:
            args.star   = 'Sun'
            args.planet_params = False
            
        if args.kul20 == 1:
            args.spot   = False
            args.planet = False
            
        if args.kul20 == 2:
            args.spot   = True
            args.planet = False
            
        if args.kul20 == 3:
            args.spot   = True
            args.planet = 'kul20'
            
        if not args.kul20 in [0, 1, 2, 3]:
            args.kul20 = False

        # Add steps from default mode
        self.mode_single()

        
    def mode_mocka(self):

        """Given the stellar properties asign variable signal.
        """

        # I/O EXTRA

        project, starType, starID, starVar, odir = args.mocka[0]
        idir = Path(os.getenv('PLATO_WORKDIR')) / project / 'input'        
        odir = Path(odir).resolve()
        self.starID = int(starID)
        
        # Load feather
        df0 = pd.read_feather(idir / f'starcat_GaiaDR3_PlatoCS_{starType}_targets.ftr')
        ds0 = pd.read_feather(idir / f'starcat_GaiaDR3_PlatoCS_{starType}_contaminants.ftr')
        
        # Output directory
        starDir = f'{self.starID}'.zfill(9)
        self.odir = odir / starDir
        self.odir.mkdir(parents=True, exist_ok=True)

        # NOTE hard-coded VSC directory
        vsc_scratch = f'/scratch/leuven/341/vsc34166/platosim/{project}/{starType}/varsource/{starDir}'
        
        # Select target star
        df_i = df0.loc[self.starID-1]
        ds_i = ds0[ds0.gaiaDR3 == df_i.gaiaDR3]
        df   = pd.concat([df_i.to_frame().T, ds_i])

        # Placeholders
        starIDs = []
        varSourceFiles = []
        
        # Check mode
        if starVar == 'tar':
            nstar = 1
            istar = range(nstar)
        elif starVar == 'con':
            nstar = df.shape[0]
            istar = range(1, nstar)
            varSourceFiles.append(f'{vsc_scratch}/varsource_001.txt')
            starIDs.append(1)
        else:
            nstar = df.shape[0] + 1
            istar = range(nstar)
            
        # Print to bash
        if self.verbose > 1:
            errorcode('message', '\nMOCKA mode is activated!\n')
            print(f'Target star is a {starType} pulsator')
            print(f'Generating noise-less light curves for {nstar} stars')

        # GENERATE LIGHT CURVES

        # Loop over each star in subfield
        for i in istar:

            # Fetch star and print
            self.df = df.iloc[i]
            if self.verbose > 1:
                errorcode('message', f'\n---------- Simulating star ID {i+1} ----------')

            # FETCH STELLAR PARAMETERS

            if np.isnan([self.df.M, self.df.R, self.df.Teff, self.df.logg, self.df.Z]).sum() > 0:

                # If one or more parameters are missing then:
                # Case 1) Only SpecType, BP-RP, Teff, logg
                # Case 2) Only SpecType, BP-RP
                # Case 3) Only SpecType
                dx = pd.read_feather(f'{idir}/starcat_GaiaDR3_Teff_SpecType_{self.df.spec}.ftr')
                
                # Case 1: Try small parameter space around Teff and logg
                ds = dx[(dx.Teff > self.df.Teff_low) &
                        (dx.Teff < self.df.Teff_upp) &
                        (dx.logg > self.df.logg_low) &
                        (dx.logg < self.df.logg_upp)]
                if not ds.empty:
                    #print('case 1')
                    ds = ds.sample(n=1, weights=ds.density, random_state=self.rng).squeeze(axis=0)

                # Case 2: Draw from nearest match in Gaia colour
                elif not pd.isna(self.df.Teff):
                    #print('case 2')
                    ds = dx.iloc[ut.findNearestIndex(dx.BP_RP, self.df.BP_RP)]
                    
                # Case 3: Draw from spectral type distributions
                else:
                    #print('case 3')
                    ds = dx.sample(n=1, weights=dx.density, random_state=self.rng).squeeze(axis=0)
                    self.df.Pmag = ut.passbandConversionG2P(self.df.Gmag, ds.BP_RP)

                # Store values
                self.df.L    = ds.L
                self.df.M    = ds.M
                self.df.R    = ds.R
                self.df.Teff = ds.Teff
                self.df.logg = ds.logg
                self.df.Z    = ds.Z
                
            # GENERIC STEPS

            self.stellar_source()
            self.stellar_spectrum()

            # SELECT VARIABLE SIGNAL

            if i > 0:

                # Initialise parameters
                p_gran  = 0.0
                p_puls  = 0.0
                p_spot  = 0.0
                p_flare = 0.0
                starType = None
                vals = np.array([0, 1])

                # Redefine base data frame for signals
                self.lc = pd.DataFrame(data=self.time.to('s').value, columns=['time'])
                
                # Functions
                def Mg_WD_limit(x): return 4.0*x + 8 
                def Mg_RG_upper(x): return 2.0*x - 1 
                def Mg_RG_lower(x): return 2.0*x - 4

                # Connection points of kinks for dwarf/giant and dwrf/compact
                xlim0 = 0.8
                xlim1 = 0.3

                # Connection line between two kinks (for SPB and beta Cep)
                x2 = np.linspace(0.3, 0.8, 100)
                def IMS_line(x): return -12.6*x + 7.76
                
                # Limit for dSct stars
                poly_low = np.polyfit([7300, 6500], [0.7,  1.4],  deg=1)
                poly_upp = np.polyfit([9300, 8500], [1.18, 2.05], deg=1)
                xx = np.arange(6000, 10000)
                L_low = np.polyval(poly_low, xx)
                L_upp = np.polyval(poly_upp, xx)
                
                # Eclipsing binaries
                if self.df.ruwe > 1.2:
                    starType = 'EB'
                    
                # Stars from MOCKA
                elif ((self.df.BP_RP < xlim1) and (self.df.Mg < Mg_WD_limit(self.df.BP_RP)) |
                      (self.df.BP_RP > xlim1) and (self.df.BP_RP < xlim0) and
                      (self.df.Mg < IMS_line(self.df.BP_RP))):
                    if self.df.spec == 'O':
                        starType = 'bCep'
                    elif self.df.spec in ['B', 'A']:
                        starType = 'SPB'                                     
                elif ((np.log10(self.df.L) > np.polyval(poly_low, self.df.Teff)) and
                      (np.log10(self.df.L) < np.polyval(poly_upp, self.df.Teff)) and
                      (self.df.M > 1.5) and (self.df.M < 2.5) and  
                      (self.df.logg > 3.5) and (self.df.L > 2)):
                    starType = 'dSct'
                elif ((self.df.M > 1.2) and (self.df.M < 2.0) and
                      (self.df.Teff > 6500) and (self.df.Teff < 9000) and
                      (self.df.logg > 3.5) and (self.df.L > 2.0)):
                    starType = 'gDor'
                elif ((self.df.M > 0.6) and (self.df.M < 0.8)):
                    starType = 'RRLyr'
                elif ((self.df.M > 0.6) and (self.df.M < 0.8)):
                    starType = 'Ceph'                    
                    # elif self.df.Mg < Mg_WD_limit(self.df.BP_RP):
                    #    starType = 'WD'

                # Miscellaneous variables
                elif ((self.df.BP_RP > xlim0) & (self.df.Mg < Mg_RG_upper(self.df.BP_RP))):
                    starType = 'LPV'
                elif self.df.spec in ['A', 'unknown', 'CSTAR', '']:
                    starType = 'SPV'
                    
                # Check massive stars if missed
                if starType == None and self.df.spec in ['O', 'B', 'A']:
                    if self.df.spec == 'O':
                        starType = 'bCep'
                    elif self.df.spec == 'B':
                        starType = 'SPB'
                    elif self.df.spec == 'A':
                        starType = 'dSct'
                    
                # Low mass dwarf stars
                elif self.df.spec in ['F', 'G', 'K', 'M']:
                    
                    # Probability of dwarf solar-like oscillator
                    if ((self.df.spec == 'F' and self.df.BP_RP > 0.7 and
                         self.df.Mg > 3 and self.df.logg > 4.4) or
                        (self.df.spec == 'G' and self.df.BP_RP > 0.7 and
                         self.df.Mg > 2 and self.df.logg > 4.4) or
                        (self.df.spec == 'K' and self.df.BP_RP > 0.7 and
                         self.df.Mg > 1 and self.df.logg > 4.4)):
                        p_puls = 1.0

                    # Probability of RG solar-like oscillator
                    if ((self.df.BP_RP > 0.7) and (self.df.logg < 3.5) and
                        (self.df.Mg < Mg_RG_upper(self.df.BP_RP)) and
                        (self.df.Mg > Mg_RG_lower(self.df.BP_RP))):
                        p_puls = 1.0
                    
                    # Probability of star spots
                    # NOTE Colour cut is transition region -> rad. vs. conv. envelopes
                    if self.df.BP_RP > 0.4:
                        p_spot = 1
                        
                    # Probability of flares:
                    # Later spectral types are more likely to have flares
                    if self.df.BP_RP > 0.4:
                        if   self.df.spec == 'F': p_flare = 0.2
                        elif self.df.spec == 'G': p_flare = 0.8
                        elif self.df.spec == 'K': p_flare = 0.9
                        elif self.df.spec == 'M': p_flare = 1.0
                        p_flare = ss.rv_discrete(values=(vals, (1-p_flare, p_flare)), seed=self.rng).rvs()
                        
                    # Probability of active M dwarf
                    if self.df.spec == 'M' and self.df.BP_RP > 1.7 and self.df.Mg > 6:
                        p_puls, p_spot, p_flare = 0, 1, 1                        
                        
                    # Select combined variability
                    if p_puls == 1 and p_spot == 0 and p_flare == 0:
                        starType = 'solar_puls'
                    elif p_puls == 1 and p_spot == 1 and p_flare == 0:
                        starType = 'solar_spot'
                    elif p_puls == 1 and p_spot == 1 and p_flare == 1:
                        starType = 'solar_flare'
                    elif p_puls == 0 and p_spot == 1 and p_flare == 1:
                        starType = 'dwarf_red'

                    #---- Back-up checks ----#
                    
                    # Check for F-type stars
                    if self.df.spec == 'F' and starType in [None, 'solar_flare', 'dwarf_red']:
                        if self.df.M > 1.3:
                            starType = 'gDor'
                        else:
                            starType = 'solar_puls'

                    # Check for GK-type stars
                    if self.df.spec in ['G', 'K'] and starType in [None, 'dwarf_red']:
                        if self.df.spec == 'G':
                            starType = 'solar_spot'
                        else:
                            starType = 'solar_flare'

                    # Check for K-type stars
                    if self.df.spec == 'K' and self.df.M < 0.85:
                        starType = 'dwarf_red'
                            
                    # Check for M-type stars
                    if self.df.spec == 'M' and starType in [None, 'solar_puls', 'solar_spot', 'solar_flare']:
                        starType = 'dwarf_red'
                        
            # Just as a sanity check, select SPV if nothing else
            if starType == None:
                starType = 'SPV'

            # SELECT VARIABLE CLASS

            if starType in ['solar_puls', 'solar_spot', 'solar_flare', 'dwarf_red']:
                args.puls = 'Corsaro2013'
                self.mocka_solar = True
            else:
                args.puls = 'mocka'
                self.mocka_solar = False
            
            # Eclipsing binary
            if starType == 'EB':
                self.binary_eb()
            
            # Massive pulsators
            elif starType == 'bCep':
                self.star_bcep()
            elif starType == 'SPB':
                self.star_spb()
            elif starType == 'dSct':
                self.star_dsct()
            elif starType == 'gDor':
                self.star_gdor()

            # Evolved stars
            elif starType == 'RG':
                self.solar_granosc()            
            elif starType == 'RRLyr':
                self.star_rrlyr()
            elif starType == 'Ceph':
                self.star_ceph()
            #elif starType == 'WD':
            #    self.star_zzceti()
                
            # Solar-like stars
            elif starType == 'solar_puls':
                self.solar_granosc()
            elif starType == 'solar_spot':
                self.solar_granosc()
                self.solar_spots()
            elif starType == 'solar_flare':
                self.solar_granosc()
                self.solar_spots()
                self.solar_flares()
            elif starType == 'dwarf_red':
                self.solar_spots()
                self.solar_flares()

            # Miscellaneous varibales
            elif starType == 'LPV':                
                self.star_lpv()
            elif starType == 'SPV':
                self.star_roap()

            # GENERATE LIGHT CURVE

            if starType:
                
                # Save each varsource to file
                sfile = 'varsource_' + f'{i+1}'.zfill(3) + '.txt'
                self.ofile = self.odir.joinpath(sfile)
                self.run_prolog()

                # Use cluster name for PLATOnium
                # NOTE Directory is defined in job script
                varSourceFiles.append(vsc_scratch + '/' + sfile)
                starIDs.append(i+1)
                
        # GENERATE VARIABLE CATALOG FILE

        if not starVar == 'tar':
            
            varSourceList = self.odir / 'varSourceList.txt'        
            if isinstance(varSourceFiles, str):
                varSourceFiles = [varSourceFiles]

            with open(varSourceList, 'w') as f:
                for j in range(len(starIDs)):
                    f.write(f'{starIDs[j]} {varSourceFiles[j]}\n')

#--------------------------------------------------------------#
#                PARSING COMMAND-LINE ARGUMENTS                #
#--------------------------------------------------------------#

parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-p', '--plot',    action='store_true',     help='Flag to plot the synthetic models')
parser.add_argument('-n', '--notes',   action='store_true',     help='Flag to show the available models')
parser.add_argument('-o', '--ofile',   metavar='STR', type=str, help='Output filename [<path/to/ofile.txt>]')
parser.add_argument('-v', '--verbose', metavar='INT', type=int, help='Verbosity level [0, 1, 3] (Default: 1)')

obs_group = parser.add_argument_group('OBS PARAMETERS')
obs_group.add_argument('--time',    metavar='DAY',  type=int, help='Duration of simulation (Default: 90 days)')
obs_group.add_argument('--quarter', metavar='NUM',  type=str, help='Quarter number or range of quaters to simulate (Default: 1)')
obs_group.add_argument('--cadence', metavar='SEC',  type=int, help='Cadence of observation (Default: 25 seconds)')
obs_group.add_argument('--inst',    metavar='NAME', type=str, help='Photometric instrument (Default: plato)')
obs_group.add_argument('--seed',    metavar='INT',  type=int, help='Option to bootstrap seed to reproduce results')

star_group = parser.add_argument_group('STAR PARAMETERS')
star_group.add_argument('--star',     metavar='NAME',     type=str, help='Benchmark star (check --notes)')
star_group.add_argument('--star_params', action='append', type=float, nargs=5, metavar=('M', 'R', 'Teff', 'logg', 'Z'),
                        help='Stellar model parameters (check --notes)')
star_group.add_argument('--gran',     metavar='MODEL', type=str, help='Model of stellar granulation [Kallinger2014, no]')
star_group.add_argument('--puls',     metavar='MODEL', type=str, help='Model of stellar pulsations [Corsaro2013, no]')
star_group.add_argument('--spot',     metavar='MODEL', type=str, help='Model of stellar spots [Aigrain2015, no]')
star_group.add_argument('--flare',    metavar='MODEL', type=str, help='Model of stellar flares [ToyModel, Doorsselaere2017, no]')
star_group.add_argument('--pulslist', metavar='FILE',  type=str, help='Use file with pulsations [frequencies/(c/d), amplitudes/dmag, phases/rad]')

planet_group = parser.add_argument_group('PLANET PARAMETERS')
planet_group.add_argument('--planet', metavar='NAME', type=str, help='Benchmark planet (check --notes)')
planet_group.add_argument('--planet_params', action='append', type=float, nargs=7, metavar=('t0', 'P', 'e', 'i', 'w', 'Rp', 'Mp'),
                          help='Planet model parameters (check --notes)')
#planet_group.add_argument('--phase_curve', action='store_true', help='Flag orbital phase curve (occultation, beaming, ellipsoidal)')
planet_group.add_argument('--ldm',  metavar='MODEL', type=str, help='Limb darkening model [power2, squareroot, quadratic, linear]')
planet_group.add_argument('--moon', metavar='NAME',  type=str, help='Benchmark moon (check --notes)')


binary_group = parser.add_argument_group('BINARY PARAMETERS')
binary_group.add_argument('--binary', metavar='NAME', type=str, help='Benchmark eclipsing binary (check --notes)')
#star_group.add_argument('--binary_params', action='append', type=float, nargs=5, metavar=('M', 'R', 'Teff', 'logg', 'Z'),
#                        help='Stellar model parameters with units [M/Msun, R/Rsun, Teff/K, logg/rel, Z/rel]')

smbhb_group = parser.add_argument_group('PLANET PARAMETERS')
smbhb_group.add_argument('--smbhb', metavar='NAME', type=str, help='Benchmark planet (check --notes)')
smbhb_group.add_argument('--smbhb_params', action='append', type=float, nargs=13,
                         metavar=('z', 't0', 'P', 'e', 'i', 'w', 'logM', 'q', 'L', 'alpha', 'vz', 'tau', 'sigma'),
                         help='SMBH binary model parameters (check --notes)')

mode_group = parser.add_argument_group('DISTRIBUTION MODES')
mode_group.add_argument('--kul20', metavar='INT',   type=int, help='Option designed for KUL-TN-20 [0, 1, 2, 3]')
mode_group.add_argument('--mocka', action='append', type=str, nargs=5, metavar=('PROJECT', 'STAR', 'ID', 'VAR', 'ODIR'),
                        help='Option designed for MOCKA')

args = parser.parse_args()

#--------------------------------------------------------------#
#                            WORKFLOW                          #
#--------------------------------------------------------------#

# Monitor script speed
tic = datetime.datetime.now()

# Initialize instance of class
v = VarSim(args)

# Mode for PLATO-CS
if args.mocka:
    v.mode_mocka()

# Mode for KUL-TN-20
elif args.kul20:
    v.mode_kul20()

# Default mode for binaries
elif args.binary:
    v.mode_binary()

# Default mode for binaries
elif args.smbhb or args.smbhb_params:
    v.mode_smbhb()
    
# Default mode for single stars
else:
    v.mode_single()

# Print run time
if (args.verbose is None) or (args.verbose > 1):
    toc = datetime.datetime.now()
    print(f'\nTotal execution time : {toc-tic} [hh:mm:ss]\n')

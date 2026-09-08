#!/usr/bin/env python

"""
This file contains all relevant functions and classes 
that are used by the PLATOnium script "varsim.py".

NOTE This class needs the Poetry install: 
     >> poetry install --with platonium 
"""

# Built-in
import os
import glob
import math
import random
import warnings
import urllib.request
from pathlib import Path

# PlatoSim standard
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import scipy.stats as ss
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d, make_interp_spline
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy import constants as c
from tqdm import tqdm
import h5py

# PLATOnium extra
from numba import njit
from PyAstronomy import funcFit, pyasl

# PlatoSim functions
import platosim.plot      as pt
import platosim.noise     as ns
import platosim.utilities as ut
from platosim.utilities import errorcode

#==============================================================#
#                        SOLAR-LIKE STARS                      #
#==============================================================#
    
class StellarSpots(object):
    """Class to generate rotational star spot modulations.

    This function simulate a synthetic noise-less light curve of main-sequence
    stars that include stellar activity in the form of cyclic spot modulations.

    Resources
    ---------
    Noyes et al.            (1984) : https://adsabs.harvard.edu/pdf/1984ApJ...279..763N
    Pillet et al.           (1993) : https://www.cambridge.org/core/journals/international-
                                     astronomical-union-colloquium/article/distribution-of-
                                     sunspot-decay-rates/9D2174592A1C0CD0E9DD8B1DF347B7FF#
    Baumann and Solanki     (2005) : https://www.aanda.org/articles/aa/abs/2005/45/aa3415-
                                     05/aa3415-05.html
    Mamajek and Hillenbrand (2008) : https://iopscience.iop.org/article/10.1086/591785/meta
    Llama et al.            (2012) : https://academic.oup.com/mnrasl/article/422/1/L72/
                                     971190?login=true
    Aigrain et al.          (2012) : https://academic.oup.com/mnras/article/419/4/3147/
                                     2908053?login=true
    Meunier et al.          (2019) : https://arxiv.org/abs/1911.05319

    Assumptions
    -----------
    - Model of spots only (missing e.g. faculae)
    - Model only valid for main-sequence stars

    Code courtesy
    -------------
    Suzanne Aigrain : Aigrain et al. (2012)
    """
    def __init__(self, seed=False):
        
        # Random number generator
        self.rng = ut.rng(seed)
        
        # Constants
        self.BV_SUN    = 0.656
        self.LRHK_SUN  = -5.025 # from Lorenzo-Oliveira et al. (2018, A&A 619, A73)
        self.PROT_SUN  = 27.0
        self.OMEGA_SUN = 2 * np.pi / (self.PROT_SUN * 86400)

        # Load table
        idir = os.getenv('PLATO_PROJECT_HOME') + '/inputfiles/data_varsim'
        self.t1 = Table.read(f'{idir}/varsim_meunier19a_t1.txt', format = 'ascii')

    ####################################
    # FROM TEFF TO ACTIVITY PARAMETERS #
    ####################################
    # All relations used are based on Meunier et al. (2019, A&A, 627, A56)
    # except where specified otherwise

    def get_stpar_from_teff(self, teff):

        # Check if Teff is within grid
        if teff < min(self.t1['Teff']) or teff > max(self.t1['Teff']):
            errorcode('Warning', 'Teff is outside range of conversion table. ' +
                      f'Returning B-V for Teff={min(self.t1["Teff"])}')
            if teff < min(self.t1['Teff']): return max(self.t1['BV'])
            if teff > max(self.t1['Teff']): return min(self.t1['BV'])
        g = interp1d(self.t1['Teff'], self.t1['BV'])
        return g(teff)

    
    def get_lrhk_from_S_and_bv(self, S, bv):
        # conversion from S to R (Noyes et al. 1984a, ApJ 279 763, Appendix a)
        lCcf = 1.13 * bv**3 - 3.91 * bv**2 + 2.84 * bv - 0.47 
        if bv < 0.63:
            x = 0.63 - bv
            lCcf += 0.135 * x - 0.814 * x**2 + 6.03 * x**3
        lrhk = -4 + np.log10(1.34) + lCcf + np.log10(S)    
        # photospheric correction (Noyes et al. 1984a, ApJ 279 763, Appendix b)    
        lrphot  = -4.898 + 1.918 * bv**2 -2.893 * bv**3
        lrhk = np.log10(10**lrhk - 10**lrphot)
        return lrhk
    

    def get_Smin_from_bv(self, bv):
        if bv < 0.94:
            Smin = 0.144
        elif bv < 1.07:
            x = (bv - 0.94) / (1.07 - 0.94)
            Smin = 0.144 + x * (0.19 - 0.144)
        else:
            x = (bv - 1.07) / (1.2 - 1.07)
            Smin = 0.19 + x * (0.48 - 0.19)
        return Smin


    def get_lrhk_from_bv(self, bv):
        Smin = self.get_Smin_from_bv(bv)
        lrhkmin = self.get_lrhk_from_S_and_bv(Smin, bv)
        lrhkmax = -0.375 * bv - 4.4
        return self.rng.random() * (lrhkmax - lrhkmin) + lrhkmin

    
    def get_ltauc_from_bv(self, bv):
        # cf Noyes et al. (1984, ApJ 279 763, Eqn 4)
        x = 1.0 - bv
        if x > 0:
            return 1.362 - 0.166 * x + 0.025 * x**2 - 5.323 * x**3
        else:
            return 1.362 - 0.14 * x

        
    def get_prot_from_lrhk_and_bv(self, lrhk, bv):
        Ro = 0.808 - 2.966 * (lrhk + 4.52)
        delta = self.rng.random() * 0.4 - 0.2
        ltc = self.get_ltauc_from_bv(bv)
        return (Ro + delta) * 10**ltc 

    
    def get_prange_from_teff_and_prot(self, teff, prot):
        p0 = -3.485 + 2.47810e-4 * teff
        p1 = 1.597 - 1.3510e-4 * teff
        alpha = 10**(p0 + p1 * np.log10(prot))
        pmax = 2 * prot / (2 - alpha)
        pmin = pmax * (1 - alpha)
        return pmin, pmax

    
    def get_latrange(self): 
        lat_min = 0.0
        lat_max = 32.0 + 20.0 * self.rng.random()
        return lat_min, lat_max # in degrees

    
    def get_omega01_from_prange_and_latrange(self, pmin, pmax, lat_min, lat_max):
        omega_min = 2 * np.pi / pmax / 86400
        omega_max = 2 * np.pi / pmin / 86400
        s2min = np.sin(np.deg2rad(lat_min))**2
        s2max = np.sin(np.deg2rad(lat_max))**2
        omega_1 = (omega_min - omega_max) / (s2max - s2min)
        omega_0 = omega_max - omega_1 * s2min 
        return omega_0, omega_1 

    
    def get_omega_from_lat_and_omega01(self, lat, omega_0, omega_1):
        # In radians per second
        return omega_0 + omega_1 * np.sin(np.deg2rad(lat))**2


    def get_pcyc_from_prot(self, prot):
        delta = self.rng.random() * 0.6 - 0.3
        y = 0.84 * np.log10(1/prot) + 3.14 + delta
        return prot * 10**y

    
    def get_acyc_from_bv_and_lrhk(self, bv, lrhk, level='random'):
        if bv < 0.851:
            Acyc_max = 0.727 * bv - 0.292
        else:
            Acyc_max = 0.727 * 0.851 - 0.292
        Acyc_min = max([0.28 * bv - 0.196, 0.342 * lrhk + 1.703, 0.005])
        # User defined activity level
        if level == 'random':
            tmp = self.rng.random()
        elif level == 'high':
            tmp = 0.9 + 0.1*tmp
        elif level == 'low':
            tmp = 0.0 + 0.1*tmp
        else:
            errorcode('error', f'Invalid activity level "{level}"')
        return tmp * (Acyc_max - Acyc_min) + Acyc_min

    
    def get_arate_from_acyc(self, acyc):
        asun = self.get_acyc_from_bv_and_lrhk(self.BV_SUN, self.LRHK_SUN)
        return acyc/asun


    def get_decay_rate(self, nspots):
        # spot emergence and decay timescales
        # The settings below are designed approximately match the distributions used in
        # Borgniet et al. (2015) and Meunier et al. (2019)
        mea = 15 * 1e-6
        med = 10 * 1e-6
        mu = np.log(med)
        sig = np.sqrt(2*np.log(mea/med))
        return self.rng.lognormal(mean=mu, sigma=sig, size=nspots)    
    

    def regions(self, activity_rate=1, cycle_period=10, cycle_overlap=0, randspots=False,
                maxlat=70, minlat=0, tsim=1000, tstart=0, verbose=False):
        """ACTIVE REGION EMERGENCE

        According to Schrijver and Harvey (1994), the number of active regions
        emerging with areas in the range [A,A+dA] in a time dt is given by:

        n(A,t) dA dt = a(t) A^(-2) dA dt ,

        where A is the "initial" area of a bipole in square degrees, and t is
        the time in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle
        maximum.

        The bipole area is the area within the 25-Gauss contour in the
        "initial" state, i.e. time of maximum development of the active region.
        The assumed peak flux density in the initial sate is 1100 G, and
        width = 0.2*bsiz (see disp_region). The parameters written onto the
        file are corrected for further diffusion and correspond to the time
        when width = 4 deg, the smallest width that can be resolved with lmax=63.

        In our simulation we use a lower value of a(t) to account for "correlated"
        regions.
        """
        nbin = 5                              # number of area bins
        delt = 0.5                            # delta ln(A)
        amax = 100.                           # orig. area of largest bipoles (deg^2)
        dcon = np.exp(0.5*delt)-np.exp(-0.5*delt)   # contant from integ. over bin
        latrmsd = 6
        atm     = 10 * activity_rate    
        # a(t) at cycle maximum (deg^2/day)
        # cycle period (days)
        # cycle duration (days)

        ncycle = int(cycle_period * 365)     # cycle length in days   
        nclen = int((cycle_period + cycle_overlap) * 365)
        fact = np.exp(delt*np.arange(nbin))  # array of area reduction factors
        ftot = fact.sum()                    # sum of reduction factors
        bsiz = np.sqrt(amax/fact)            # array of bipole separations (deg)
        tau1 = 5                             # first and last times (in days) for
        tau2 = 15                            #   emergence of "correlated" regions
        prob = 0.001                         # total probability for "correlation"
        nlon = 36                            # number of longitude bins
        nlat = 16                            # number of latitude bins       
        nday1 = 0                            # first day to be simulated
        ndays = int(tsim)                    # number of days to be simulated
        dt = 1

        # Initialize time since last emergence of a large region, as function
        # of longitude, latitude and hemisphere:
        tau = np.zeros((nlon, nlat, 2), 'int') + tau2
        dlon = 360. / nlon
        dlat = maxlat / nlat

        # Create arrays to store regions properties
        reg_tims = []
        reg_lats = []
        reg_lons = []
        reg_angs = []

        # Loop over time (in days):
        ncnt = 0
        ncur = 0
        start_day = 0

        for nd in range(ndays):
            nday = nd + nday1

            # Compute index of most recently started cycle:
            ncur_now = int(nday / ncycle)
            ncur_prev = int((nday-1) / ncycle)
            if ncur_now > ncur_prev:
                ncur = ncur + 1

            #  Initialize rate of emergence for largest regions, and add 1 day
            #  to time of last emergence:
            tau = tau + 1
            rc0 = np.zeros((nlon,nlat,2))
            l = (tau > tau1) & (tau <= tau2)
            if l.any():
                rc0[l] = prob / (tau2 - tau1)

            #  Loop over current and previous cycle:
            for icycle in [0,1]:
                nc = ncur-icycle # index of cycle
                if ncur == 0:
                    start_day = nc * ncycle
                else:  
                    if ncur == 1:
                        if icycle == 0:
                            start_day = ncycle * nc
                        elif icycle == 1:
                            start_day = 0
                    else:
                        start_day = ncycle * nc

                nstart = start_day        # start date of cycle
                if (nday-nstart) < nclen:  
                    ic = 1 - 2 * ((nc + 2) % 2) # +1 for even, -1 for odd cycle
                    phase = float(nday-nstart) / nclen # phase within the cycle

                    # Emergence rate of largest "uncorrelated" regions (number per day,
                    # both hemispheres), from Schrijver and Harvey (1994):
                    ru0_tot = atm * np.sin(np.pi * phase)**2 * (1.0 * dcon) / amax

                    # Emergence rate of largest "uncorrelated" regions per latitude/longitude
                    # bin (number per day), as function of latitude:
                    if randspots:
                        latavg = (maxlat - minlat) / 2. 
                        latrms = maxlat - minlat
                        nlat1 = np.floor(minlat / dlat).astype(int)
                        nlat2 = np.floor(maxlat / dlat).astype(int)
                        nlat2 = min([nlat2, nlat - 1])
                    else:
                        latavg = maxlat + (minlat - maxlat)*phase #+ 5.*phase**2
                        latrms = (maxlat/5.) - latrmsd * phase # rms latitude (degrees)
                        nlat1 = np.floor(max([maxlat * 0.9 - 1.2 * maxlat * phase, 0.0]) /
                                         dlat).astype(int) # first and last index
                        nlat2 = np.floor(min([maxlat + 15. - maxlat * phase, maxlat]) /
                                         dlat).astype(int)
                        nlat2 = min([nlat2, nlat - 1])

                    js = np.arange(nlat2 - nlat1).astype(int)

                    p = np.zeros(nlat)
                    for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                        p[j] = np.exp( - ((dlat * (0.5 + j) - latavg) / latrms)**2)
                    ru0 = ru0_tot * p / (p.sum() * nlon * 2)

                    # Loops over hemisphere and latitude:
                    for k in [0,1]:
                        for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                            # Emergence rates of largest regions per
                            # longitude/latitude bin (number per day):
                            r0 = ru0[j] + rc0[:,j,k]
                            rtot = r0.sum()
                            ssum = rtot * ftot
                            x = self.rng.random()
                            if x <= ssum:
                                nb = 0
                                sumb = rtot * fact[0]
                                while x > sumb:
                                    nb = nb + 1
                                    sumb = sumb + rtot * fact[nb]
                                i = 0
                                sumb = sumb + (r0[0] - rtot) * fact[nb]
                                while x > sumb:
                                    i = i + 1
                                    sumb = sumb + r0[i] * fact[nb]
                                lon = dlon * (self.rng.random() + float(i))
                                lat = dlat * (self.rng.random() + float(j))
                                if (nday > tstart):
                                    reg_tims.append(self.rng.random() + nday)
                                    reg_lons.append(lon)
                                    if k == 0:                  # Insert on N hemisphere
                                        reg_lats.append(lat)
                                    else:
                                        reg_lats.append(-lat)
                                    x = self.rng.normal()
                                    while abs(x) > 1.6:
                                        x = self.rng.normal()
                                    y = self.rng.normal()
                                    while abs(y) >= 1.6:
                                        y = self.rng.normal()
                                    z = self.rng.random()
                                    if z > 0.14:
                                        # Tilt angle (degrees)
                                        ang = 0.5 * lat + 2.0 + 27. * x * y
                                    else:
                                        z = self.rng.normal()
                                        while z > 0.5:
                                            z = self.rng.normal()
                                        ang = np.deg2rad(z)
                                    reg_angs.append(ang)
                                    if verbose:
                                        print(reg_tims[-1], reg_lats[-1],
                                              reg_lons[-1], reg_angs[-1])
                                ncnt = ncnt + 1
                                if nb < 1:
                                    tau[i,j,k] = 0

        if verbose:
            print('Total number of regions: ', ncnt)

        reg_arr = np.zeros((4, len(reg_tims)))
        reg_arr[0] = np.array(reg_tims)
        reg_arr[1] = np.array(reg_lats)
        reg_arr[2] = np.array(reg_lons)
        reg_arr[3] = np.deg2rad(np.array(reg_angs))
        return reg_arr
    
    #---------------------------------------------------- v3

    def spots(self, reg_arr, incl=None, omega_0=None, omega_1=0.0,
              dur=None, threshold=0.1):

        """Holds parameters for spots for a given star.
        
        Generate initial parameter set for spots (emergence times
        and initial locations.
        """

        # Set global parameters which are the same for all spots inclination [deg]
        if incl == None:
            self.incl = np.rad2deg(np.arcos(self.rng.uniform()))
        else:
            self.incl = incl

        if omega_0 is None:
            omega_0 = self.OMEGA_SUN
            
        # Rotation and differential rotation [rad/s]
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        
        # Regions parameters
        t0  = reg_arr[0,:]
        lat = reg_arr[1,:]
        lon = reg_arr[2,:]
        ang = reg_arr[3,:]

        # Keep only spots emerging within specified time-span with:
        # peak B-field > threshold
        if dur == None:
            self.dur = t0.max() 
        else:
            self.dur = dur
        l = (t0 < self.dur) * (ang > threshold)
        self.nspots = l.sum()
        self.t0 = t0[l]
        self.lat = lat[l]
        self.lon = lon[l]
        
        # The settings below are designed approximately match the distributions
        # used in Borgniet et al. (2015) and Meunier et al. (2019) spot sizes
        self.amax = ang[l]**2 * 300 * 1e-6
        
        # spot emergence and decay timescales
        mea = 15 * 1e-6
        med = 10 * 1e-6
        mu = np.log(med)
        sig = np.sqrt(2*np.log(mea/med))
        self.decay_rate = self.rng.lognormal(mean=mu, sigma=sig, size=self.nspots)


    def calci(self, time, i):

        """Single spot calculation.

        Evolve one spot and calculate its impact on the stellar flux.
        NOTE: Currently there is no spot drift or shear.
        """
        
        # Spot area (linear growth and decay)
        area        = np.zeros(len(time)) 
        decay_time  = self.amax[i] / self.decay_rate[i]
        emerge_time = decay_time / 10.0

        # squared exponential growth and decay
        l = time < self.t0[i]
        area[l] = self.amax[i] * np.exp(-0.5*(self.t0[i]-time[l])**2 / emerge_time**2)
        l = time >= self.t0[i]
        area[l] = self.amax[i] * np.exp(-0.5*(time[l]-self.t0[i])**2 / decay_time**2)

        # exponential growth and decay
        # l = time < self.t0[i]
        # area[l] = self.amax[i] * np.exp(-(self.t0[i]-time[l]) / emerge_time)
        # l = time >= self.t0[i]
        # area[l] = self.amax[i] * np.exp(-(time[l]-self.t0[i]) / decay_time)
        
        # linear growth and decay
        # l = (time >= (self.t0[i]-emerge_time)) * (time < self.t0[i])
        # area[l] = self.amax[i] * (self.t0[i]-time[l]) / emerge_time
        # l = (time >= self.t0[i]) * (time < (self.t0[i]+decay_time))
        # area[l] = self.amax[i] * (1-(time[l]-self.t0[i]) / decay_time)

        # Rotation rate [rad/s]
        ome = self.get_omega_from_lat_and_omega01(self.lat[i], self.omega_0, self.omega_1)
        
        # Fore-shortening 
        phase = ome * time * 86400 + np.deg2rad(self.lon[i]) # [rad]
        beta  = (np.cos(np.deg2rad(self.incl)) * np.sin(np.deg2rad(self.lat[i])) +
                 np.sin(np.deg2rad(self.incl)) * np.cos(np.deg2rad(self.lat[i])) *
                 np.cos(phase))
        
        # Differential effect on stellar flux
        dF = - 2 * area * beta
        dF[beta < 0] = 0
        return area, ome, beta, dF


    def calc(self, time):

        """Calculations for all spots
        """
        
        N = len(time)
        M = self.nspots

        # Don't go beyond 5 Gb RAM memory!
        bytemax = int(5/8*1e9)
        X = N*M
        if X > bytemax:
            errorcode('warning', f'Too many spots -> Array of size larger than 5 Gb!')
            return None, None, None, None
        else:
            area = np.zeros((M, N))
            ome = np.zeros(M)
            beta = np.zeros((M, N))
            dF = np.zeros((M, N))
            for i in np.arange(M):
                area_i, omega_i, beta_i, dF_i = self.calci(time, i)
                area[i,:] = area_i
                ome[i] = omega_i
                beta[i,:] = beta_i
                dF[i,:] = dF_i
            return area, ome, beta, dF

    #---------------------------------------------------- v4

    def compute_spot_area(self,
                          time: np.ndarray,
                          spot_params: Table,
                          min_area: float,
                          evolution: str = 'exponential'
                          ) -> np.ndarray:
        """ Compute the area of a single spot as a function of time.

        Parameters
        ----------
        time: np.ndarray
            Array of times at which to compute spot parameters.
        spot_params: Table
            A row from an astropy Table containg the parameters of the spot.
        min_area: float
            The smallest spot area to be considered in units of hemispheres.
        evolution: str
            The temporal evolution of the spot area (default: 'exponential').

        Returns
        -------
        area: np.ndarray
            The size of the spot in units of hemispheres.

        """

        # Extract spot parameters.
        amax = spot_params['A_MAX']*1e-6
        tmax = spot_params['T_MAX']
        decay_time = spot_params['TAU']
        emerge_time = decay_time/10.0

        # Compute the spot area.
        tmp1 = (time - tmax)/emerge_time
        tmp2 = (time - tmax)/decay_time

        if evolution == 'exponential':
            area = np.where(time < tmax, np.exp(-np.abs(tmp1)), np.exp(-np.abs(tmp2)))
        elif evolution == 'squared-exponential':
            area = np.where(time < tmax, np.exp(-0.5 * tmp1 ** 2), np.exp(-0.5 * tmp2 ** 2))
        else:
            raise ValueError(f"Unknown value for spot evolution profile: {evolution}")

        area = amax*area
        area = np.where(area < min_area, 0, area)

        return area


    def compute_spot_location(self,
                              time: np.ndarray,
                              spot_params: Table
                              ) -> tuple[np.ndarray, np.ndarray]:
        """ Compute the location of a single spot as a function of time.

        Parameters
        ----------
        time: np.ndarray
            Array of times at which to compute spot parameters.
        spot_params: Table
            A row from an astropy Table containg the parameters of the spot.

        Returns
        -------
        lat: np.ndarray
            The spot latitude.
        lon: np.ndarray
            The spot longitude.

        """

        # Extract spot parameters.
        lat = spot_params['LAT']
        lon = spot_params['LON']
        prot = spot_params['PROT']

        # Compute projected spot position as a function of time.
        lat = lat * np.ones_like(time)
        lon = lon + 360 * time / prot
        lon = np.mod(lon - 180, 360) - 180

        return lat, lon


    def compute_spot_parameters(self,
                                time: np.ndarray,
                                spot_params: Table,
                                inc_star: float = 90.,
                                min_area: float = 0.,
                                evolution: str = 'exponential'
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Compute the latitude, longitude, angular distance from the center of the
            stellar disk, and radius in angular units for a given starspot.

        Parameters
        ----------
        time: np.ndarray
            Array of times at which to compute spot parameters.
        spot_params: Table
            A row from an astropy Table containg the parameters of the spot.
        inc_star: float
            The stellar inclination in degrees (default: 90 degrees).
        min_area: float
            The smallest spot area to be considered in units of hemispheres
            (default: 0).
        evolution: str
            The temporal evolution of the spot area.

        Returns
        -------
        alpha: np.ndarray
            The spot radius in degrees.
        beta: np.ndarray
            The spot location in degrees from the center of the stellar disk.
        lat: np.ndarray
            The spot latitude.
        lon: np.ndarray
            The spot longitude.

        """
        
        # Compute the spot position and area.
        lat, lon = self.compute_spot_location(time, spot_params)
        area = self.compute_spot_area(time, spot_params, min_area, evolution=evolution)

        # Convert degrees to radians.
        inc_star = np.deg2rad(inc_star)
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)

        # Compute the spot angle beta.
        cos_beta = np.cos(inc_star) * np.sin(lat) + np.sin(inc_star) * np.cos(lat) * np.cos(lon)
        beta = np.arccos(cos_beta)

        # Convert spot areas to spot radii.
        sa = 2 * np.pi * area  # Spot area in steradians.
        alpha = np.arccos(1 - sa / (2 * np.pi))  # Spot radius in radians.

        return alpha, beta, lat, lon, area

    
    def filter_spots_table(self,
                           time: np.ndarray,
                           spots_table: Table,
                           min_area: float = 1e-8,
                           evolution: str = 'exponential'
                           ) -> Table:
        """ Given an array of times compute which spots can contribute to the
            lightcurve.

        Parameters
        ----------
        time: np.ndarray
            Array of times at which to compute the lightcurve.
        spots_table: Table
            An astropy Table containg the parameters of the spots to be used.
        min_area: float
            The smallest spots area to be considered in units of hemispheres
            (default: 1e-8).
        evolution: str
            The temporal evolution of the spot area.

        Returns
        -------
        spots_table: Table
            A version of the input spots_table containing only the relevant spots.

        """

        tmin = np.amin(time)
        tmax = np.amax(time)

        area = self.compute_spot_area(tmin, spots_table, min_area=min_area, evolution=evolution)
        mask1 = (spots_table['T_MAX'] < tmin) & (area < min_area)

        area = self.compute_spot_area(tmax, spots_table, min_area=min_area, evolution=evolution)
        mask2 = (spots_table['T_MAX'] > tmax) & (area < min_area)
        
        mask = mask1 | mask2

        return spots_table[~mask]


    def parse_limb_darkening(self,
                             ld_type: str,
                             ld_pars: ArrayLike
                             ) -> tuple[np.ndarray, np.ndarray]:
        """ Parse various limb-darkening laws into the non-linear form.
        """

        if ld_type not in ['uniform', 'linear', 'quadratic', 'nonlinear']:
            raise ValueError(f"Unknown limb-darkening law: {ld_type}")

        ld_idx = np.arange(5)
        ld_pars_ = np.zeros(5)

        if ld_type == 'uniform':
            pass
        if ld_type == 'linear':
            ld_pars_[2] = ld_pars[0]
        if ld_type == 'quadratic':
            ld_pars_[2] = ld_pars[0] + 2*ld_pars[1]
            ld_pars_[4] = -ld_pars[1]
        if ld_type == 'nonlinear':
            ld_pars_[1] = ld_pars[0]
            ld_pars_[2] = ld_pars[1]
            ld_pars_[3] = ld_pars[2]
            ld_pars_[4] = ld_pars[3]

        ld_pars_[0] = 1 - np.sum(ld_pars_[1:])

        return ld_idx, ld_pars_

    
    def zeta_func(self, x: np.ndarray) -> np.ndarray:
        """ The zeta function defined in Kipping 2012, equation 17.
        """
        return np.cos(x)*np.heaviside(x, 0.5)*np.heaviside(np.pi/2 - x, 0.5) + np.heaviside(-x, 0.5)


    def kipping_spot_model(self,
                           time: np.ndarray,
                           spots_table: Table,
                           inc_star: float = 90.,
                           ld_type: str = 'linear',
                           ld_pars: ArrayLike = (0.6,),
                           min_area: float = 0.,
                           evolution: str = 'exponential'
                           ) -> np.ndarray:
        """ Computes the flux coming from a star covered in evolving starspots
            following Kipping 2012.

        Parameters
        ----------
        time : np.ndarray
            The times for which to compute the flux.
        spots_table : astropy.table.Table
            The parameters of the star spots.
        inc_star : float
            The inclination of the star in degrees (default: 90.).
        ld_type : str
            The limd-darkening law used (default: 'linear').
        ld_pars : tuple
            The limb-darkening parameters to use (default: (0.6,)).
        min_area : float
            The smallest spot areas to consider in hemispheres,
            when the spot area is below this threshold it will be set to zero (default: 0.).
        evolution : str
            Time evolution of the spot-area, either an 'exponential' or
            'squared-exponential' profile may be used.

        Returns
        -------
        flux : np.ndarray
            The stellar flux values.

        """

        spots_table = self.filter_spots_table(time,
                                              spots_table,
                                              min_area=min_area,
                                              evolution=evolution)

        ld_idx, ld_pars = self.parse_limb_darkening(ld_type, ld_pars)
        const1 = np.sum(ld_idx * ld_pars / (ld_idx + 4))

        flux = np.ones_like(time) - const1
        area = []
        for i in range(len(spots_table)):

            alpha, beta, _, _, A = self.compute_spot_parameters(time,
                                                                spots_table[i],
                                                                inc_star=inc_star,
                                                                min_area=min_area,
                                                                evolution=evolution)
            area.append(A)
            mask = alpha > 0
            if not np.any(mask):
                continue

            args, = np.where(mask)
            imin = np.amin(args)
            imax = np.amax(args) + 1  # Add 1 because slices are exclusive.

            alpha = alpha[imin:imax]
            beta = beta[imin:imax]

            # Convert to complex for use with Kipping 2012, equation 14.
            alpha = alpha.astype('complex256')
            beta = beta.astype('complex256')

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                cot_alpha = 1/np.tan(alpha)
                cot_beta = 1/np.tan(beta)
                xi = np.sin(alpha)*np.arccos(-cot_alpha*cot_beta)
                psi = np.sqrt(1 - np.cos(alpha)**2/np.sin(beta)**2)
                a = np.arccos(np.cos(alpha)/np.sin(beta))
                b = np.cos(beta)*np.sin(alpha)*xi
                c = np.cos(alpha)*np.sin(beta)*psi

            alpha = alpha.real
            beta = beta.real
            sky_area = (a + b - c).real

            # The above equations seems to contain some singularities at beta=0,pi, this fixes them.
            sky_area = np.where(beta > np.pi/2 - alpha, sky_area, np.pi * np.sin(alpha)**2 * np.cos(beta))
            sky_area = np.where(beta < np.pi/2 + alpha, sky_area, 0)

            # Equations C23.
            zeta_neg = self.zeta_func((beta - alpha).real)
            zeta_pos = self.zeta_func((beta + alpha).real)

            denom = zeta_neg**2 - zeta_pos**2
            denom = np.where(denom < 1e-6, 1, denom)

            const2 = 0
            for j in ld_idx:
                exp = (j + 4)/2
                num = zeta_neg**exp - zeta_pos**exp
                const2 += (4*ld_pars[j])/(j + 4)*num/denom

            flux[imin:imax] = flux[imin:imax] - sky_area/np.pi*const2

        flux = flux/(1 - const1)

        return flux.astype('float64'), np.array(area)
    
    #----------------------------------------------------
    
    def evaluate(self,
                 teff,
                 time,
                 dur,
                 cadence_hours,
                 incl=None,
                 activity_level: str = 'random',
                 activity_phase: tuple[float, float] = (0., 1.),
                 min_area: float = 1e-8,
                 evolution: str = 'squared-exponential',
                 ld_type: str = 'linear',
                 ld_pars: ArrayLike = (0.6,),
                 odir=None,
                 verbose=False,
                 save=False):
        """Generate spot modulated light curve.
        """
        if odir is None:
            odir = os.getcwd()
            
        # select parameters
        bv   = self.get_stpar_from_teff(teff)
        lrhk = self.get_lrhk_from_bv(bv)
        prot = self.get_prot_from_lrhk_and_bv(lrhk, bv)
        pmin, pmax       = self.get_prange_from_teff_and_prot(teff, prot)
        lmin, lmax       = self.get_latrange()
        omega_0, omega_1 = self.get_omega01_from_prange_and_latrange(pmin, pmax, lmin, lmax)
        pcyc     =  self.get_pcyc_from_prot(prot)
        clen     = pcyc / 365.
        coverlap = self.rng.random() * 0.1 * clen 
        acyc     = self.get_acyc_from_bv_and_lrhk(bv, lrhk, level=activity_level)
        arate    = self.get_arate_from_acyc(acyc)
        if incl is None:
            incl = np.rad2deg(np.arccos(self.rng.random()))
        
        # simulate LC
        #------------------------------------------------------------------ v3
        # # simulate regions
        # reg_arr = self.regions(activity_rate=arate,
        #                        cycle_period=clen, cycle_overlap=coverlap, verbose=False,
        #                        maxlat=lmax, minlat=lmin, tsim=dur+pcyc, tstart=0)
        # # make the simulation start at a random point in the cycle
        # reg_arr[0] -= self.rng.random() * pcyc
        # # Calculate spots
        # self.spots(reg_arr, incl=incl, omega_0=omega_0, omega_1=omega_1,
        #            threshold=0.1, dur=dur)
        # # Decrease the sampling to 30 min for increased performance
        # time0 = np.copy(time)
        # time = time[::72] - time0[0]
        # area, ome, beta, dF = self.calc(time)
        # # We interpolate back to original time grid of 25s cadence
        # # Interpolate (piecewise cubic) into higher resolution grid
        # spline = make_interp_spline(time, dF.sum(0), k=3)
        # flux   = spline(time0)
        # # Finito!
        # self.dur, self.time, self.flux, self.area = dur, time, dF.sum(0), area.sum(0)
        # self.params = [bv.tolist(), lrhk, arate, prot, pmin, pmax, clen, coverlap, lmax, incl]
        # return flux, self.params, self.area
        #------------------------------------------------------------------ v4
        # Simulate a generous time-span to ensure we have all the spots we need.
        n = np.ceil(dur/pcyc/2) + 1
        span = (2*n + 1) * pcyc
        # simulate regions
        reg_arr = self.regions(activity_rate=arate, cycle_period=clen, cycle_overlap=coverlap,
                               maxlat=lmax, minlat=lmin, tsim=span, tstart=0, verbose=verbose)
        # Pick a time t0 such that the middle of the duration falls within a certain phase range of the activity cycle.
        phase0 = activity_phase[0] + self.rng.random() * (activity_phase[1] - activity_phase[0])
        t0 = (n + phase0)*pcyc - dur/2
        reg_arr[0] -= t0
        # Unpack the regions
        tmax  = reg_arr[0,:]
        lat   = reg_arr[1,:]
        lon   = reg_arr[2,:]
        amax  = reg_arr[3,:]**2 * 300 * 1e-6  # Area in hemispheres.
        self.t0, self.lat, self.amax = tmax, lat, amax
        # compute omega and decay_rate for each spot
        omega      = self.get_omega_from_lat_and_omega01(lat, omega_0, omega_1)
        decay_rate = self.get_decay_rate(len(tmax))
        prot_diff  = 2 * np.pi / omega / 86400
        lifetime = amax / decay_rate
        # Store number of spots for plotting
        self.nspots = len(omega)
        # Create the spots table
        meta_data = dict()
        meta_data["T_eff"] = teff
        meta_data["B-V"] = bv
        meta_data["log R'_HK"] = lrhk
        meta_data["P_rot"] = prot
        meta_data["P_min"] = pmin
        meta_data["P_max"] = pmax
        meta_data["max. latitude"] = lmax
        meta_data["P_cycle"] = clen
        meta_data["Cycle overlap"] = coverlap
        meta_data["Activity rate"] = arate
        spots_table = Table([lat, lon, prot_diff, tmax, amax*1e6, lifetime, lifetime/prot_diff],
                            names=('LAT', 'LON', 'PROT', 'T_MAX', 'A_MAX', 'TAU', 'TAU_R'),
                            meta=meta_data)

        # Remove spots that do not contribute to the simulated lightcurve.
        time0 = time - time[0]
        spots_table = self.filter_spots_table(time0,
                                              spots_table,
                                              min_area=min_area,
                                              evolution=evolution)
        # Decrease the sampling to 30 min for increased performance: (30*60)s/25s = 72
        time1 = time0[::72]        
        # Simulate the lightcurve.
        flux1, area1 = self.kipping_spot_model(time1,
                                               spots_table,
                                               inc_star=incl,
                                               ld_type=ld_type,
                                               ld_pars=ld_pars,
                                               min_area=min_area,
                                               evolution=evolution)
        # We interpolate back to original time grid of 25s cadence
        # NOTE Interpolate (piecewise cubic) into higher resolution grid
        flux1 -= 1 # Normalised around zero 
        spline = make_interp_spline(time1, flux1, k=3)
        flux0  = spline(time0)
        
        # Finito!
        self.dur, self.time, self.flux, self.area = dur, time1, flux1, area1.sum(0)
        self.params = [bv.tolist(), lrhk, arate, prot, pmin, pmax, clen, coverlap, lmax, incl]
        return flux0, self.params, self.area        
        #------------------------------------------------------------------ end
    

    def plot(self, title='params', panels=3, figsize=(11,8)):
        """Plot spot modulation model.
        """
        fig, axes = plt.subplots(panels,1, figsize=figsize, sharex=True)
        if title == 'params': 
            title= ('Model: ' +
                    r'${\rm AR} = $'+f'{self.params[2]:5.3f}, ' +
                    r'${\rm CL} = $'+f'{self.params[6]:6.3f} yr, ' +
                    r'$P_{\rm min} = $'+f'{self.params[4]:6.2f} d, ' +
                    r'$P_{\rm max} = $'+f'{self.params[5]:6.2f} d, ' +
                    r'$L_{\rm max} = $'+f'{self.params[8]:5.2f}'+r'$^{\circ}$, ' +
                    r'$i = $'+f'{self.params[9]:6.2f}'+r'$^{\circ}$')
        if title:
            axes[0].set_title(title, fontsize='18')
        axes[0].set_facecolor('lemonchiffon')
        axes[0].axhline(y=0, color='gray', linestyle='--')
        for j in range(self.nspots):
            if self.t0[j] < -10: continue
            if self.t0[j] > self.dur: continue
            axes[0].plot(self.t0[j], self.lat[j], 'ko', alpha=0.8, ms=self.amax[j]*(1./3e-4)*5)
        axes[0].set_ylim(-90, 90)
        axes[0].set_ylabel(r'Latitude [$^{\circ}$]')
        axes[1].plot(self.time, self.area*1e3, 'k-')
        axes[1].set_ylabel(r'Coverage [\%]')        
        axes[1].set_xlim(0, self.dur)
        axes[2].plot(self.time, self.flux*1e3, 'k-')
        axes[2].set_ylabel('Spot flux [ppt]')
        axes[-1].set_xlabel(r'Time from $t_0$ [days]')
        plt.tight_layout(h_pad=0.1)
        return fig, axes
        
        
class StellarFlares(object):
    """Model stellar flares.

    A simplistic analytical description of stellar flares described by
    an sudden flux increase followed by an exponential decay. Given the
    time of the time series the corresponding flux is returned including
    the wanted flares.
    """
    def __init__(self, time, scale=None, seed=False):
        
        # Store array
        self.time  = time
        self.scale = scale
        self.rng   = ut.rng(seed)


    def initToyModel(self):
        """Uniform distribution of toy model.

        Model parameter from Jasper Thys MSc thesis, which are imspired
        by the distribution from Van Doorsselaere et al. (2017). 
        """

        # Rate of flaring events [events/quarter]
        n_rate = self.rng.uniform(5, 15, 1)[0]

        # Number of flares scales with lenght of time series
        n_max    = int(n_rate * self.time[-1] / ut.quarter())
        n_flares = self.rng.integers(1, n_max, 1)[0]

        # Time of peak flare flux [d]
        self.tmax = self.rng.uniform(0, self.time[-1], n_flares)

        # FWHM time scale of flare [min -> d]
        self.tscale = self.rng.uniform(1, 150, n_flares) / (24 * 60.)

        # Amplitude distibution of flares (< 10 ppt) [norm]
        self.ampl = self.rng.exponential(0.001, n_flares)

        return n_rate, n_flares
    

    def initDoorsselaere2017(self, spec_type, activity_rate, spot_coverage):
        """Model parameters from Kepler M-dwarf (Van Doorsselaere et al. 2017)

        Notes
        -----
        Since the distributions of this paper used the long cadence
        (i.e. 30-min) Kepler observations, there may be a potential
        of missing short (<30 min) flaring events.
        """
        
        # Spectral type dependence
        if spec_type == 'F':
            a_A, b_A = -0.36, -1.06
            a_rate, b_rate = -0.21, -0.33
        elif spec_type == 'G':
            a_A, b_A = -0.36, -1.35
            a_rate, b_rate = -0.15, -0.69
        elif spec_type in ['K', 'M']:
            a_A, b_A = -0.23, -1.38
            a_rate, b_rate = -0.13, -0.54
        else:
            errorcode('warning', f'Spectral type {spec_type} is not in [F, G, K, M]. ' +
                      'Ignoring stellar flares..')
            return

        # Draw random number of grid points to sample from
        N = self.rng.integers(1000, 5000, 1)[0]
        
        # Use rate benchmark of 20 [events/star/quater]
        # Number of flares scales with lenght of time series and activity rate
        n_range = np.linspace(0, 12, N)
        n_func  = 10**(a_rate * n_range + b_rate)
        n_rate  = pd.Series(n_range).sample(1,
                                            weights=n_func,
                                            random_state=self.rng).to_numpy()[0]
        n_rate_a = n_rate * activity_rate
        n_flares = int(n_rate_a * self.time[-1] / ut.quarter())
        # Secure at least one flare
        if n_flares == 0:
            n_flares += 1
        
        # Time a peak flux of flare [d]
        # We use spot coverage as weight for drawing the flares
        area = spot_coverage * 100
        time = np.linspace(self.time[0], self.time[-1], len(area))
        spline = make_interp_spline(time, area, k=3)
        self.area = np.abs(spline(self.time))        
        self.tmax = pd.Series(self.time).sample(n_flares,
                                                weights=self.area,
                                                random_state=self.rng).to_numpy()

        # Amplitude distibution of flares (< 10 ppt) [norm]
        A_range = np.linspace(0, 10, N)
        A_func  = 10**(a_A * A_range + b_A)
        self.ampl = pd.Series(A_range).sample(n_flares,
                                              weights=A_func,
                                              random_state=self.rng).to_numpy() / 1e3
        
        # Secure lower amplitudes for less active stars
        if activity_rate < 1:
            self.ampl *= activity_rate
                
        # FWHM time scale of flare [min -> d]
        scale_fwhm = self.ampl / self.ampl.max() / 2
        self.tscale = self.rng.uniform(10, 200, n_flares) / (24 * 60.) * scale_fwhm
        return n_rate_a, n_flares

    
    def evaluate(self):
        # Apply bolometric correction
        if self.scale:
            self.ampl *= self.scale

        # Prepare flux array
        self.flux = np.ones_like(self.time)
        
        for i in range(len(self.tmax)):
            flux_rel = model_flares(self.time, self.tmax[i], self.tscale[i])
            flux_ratio = flux_rel * self.ampl[i]
            self.flux *= (flux_ratio + 1)

        # Return parameters
        df = pd.DataFrame({'tmax_day': self.tmax,
                           'tscale_day': self.tscale,
                           'ampl_norm': self.ampl})    
        return self.flux, df

        
    def plot(self):
        """Function to plot result.
        """
        try:
            self.area
        except:
            plt.figure(figsize=(9, 4))        
            plt.plot(self.time, (self.flux-1)*1e6, 'k-')
            plt.xlabel('Time [d]')
            plt.ylabel(r'Flux [ppm]')
            plt.xlim(np.min(self.time), np.max(self.time))
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(2, 1, figsize=(9, 6))
            ax[0].plot(self.time, self.area, 'k-')
            ax[1].plot(self.time, (self.flux-1)*1e6, 'k-')
            ax[1].set_xlabel('Time [d]')
            ax[0].set_ylabel(r'Spot coverage [\%]')
            ax[1].set_ylabel(r'Flux [ppm]')
            ax[0].set_xlim(np.min(self.time), np.max(self.time))
            ax[1].set_xlim(np.min(self.time), np.max(self.time))
            plt.tight_layout()
            plt.show()

        
@njit
def model_flares(time, tmax, tscale, asym=1):
    """Analytic model of stellar flares (Daveport+2014)

    Parameters
    ----------
    time : ndarray
        Time points arrray [days]
    tmax : float
        Full-width at half-maximum of the flare(s) maximum intensity [days]
    tscale : float
        Time scale duration of the flare(s) [days]
    asym : float
        Asymmetry factor of the flare(s)
    """

    # Placeholders
    flux = np.zeros_like(time)

    # Sampling
    dt = np.diff(time)[0]

    # Start and end of flare event
    t0 = (time[0]  - tmax)
    t1 = (time[-1] - tmax)

    # Time array during flare event
    tn = np.arange(t0, t1, dt)
    t  = tn / tscale

    # Model parameters of flare
    B = asym
    C = 1/B
    b = -1.941 - 0.175 + 2.246 + 1
    c = 1 - 0.689

    # Loop over every time-step in the flare time interval:
    # NOTE: this is defined relative to this flares maxima
    # and put in units of the time-scale here the analytic
    # expressions for the rise and decay are used to determine
    # the flux of this flare

    for i in range(len(t)):

        # Rise of flare
        if t[i]*B > -1 and t[i]*B <= 0:
            flux[i] += (1
                        + 1.941 * (t[i]*B)
                        - 0.175 * (t[i]*B)**2
                        - 2.246 * (t[i]*B)**3
                        - b     * (t[i]*B)**4)

        # Decay of flare
        elif t[i]*C > 0:
            flux[i] += (0.689 * np.exp(-1.6    * t[i]*C) +
                        c     * np.exp(-0.2783 * t[i]*C))

        # No flare
        else:
            flux[i] += 0

    return flux


        
class SolarLikeOscillator(object):
    """Class to generate gravity oscillation time series.

    Function to simulate stellar granulation and stochastic oscillation (p-modes).
    Each synthetic component of stellar variability is generated in the time domain
    and from asteroseismic scaling relations, for which a vararity of relations are
    are made available to choose from for the user. This function apply the bolometric
    correction when observing in the PLATO passband to each signal compponent.

    Resources
    ---------
    Kjeldsen & Bedding (1995) : https://arxiv.org/abs/astro-ph/9403015
    Michel et al.      (2005) : https://arxiv.org/abs/0809.1078
    Chaplin et al.     (2009) : https://arxiv.org/abs/0905.1722
    Kjeldsen & Bedding (2011) : https://arxiv.org/abs/1104.1659
    Ballot et al.      (2011) : https://arxiv.org/abs/1105.4557
    Kallinger et al.   (2014) : https://arxiv.org/abs/1408.0817

    Code & Data (p-modes)
    ---------------------
    De Ridder et al.   (2006) : https://academic.oup.com/mnras/article/365/2/595/976827
    Broomhall et al.   (2009) : Tables in paper
    """
    def __init__(self, time, star_params, path, seed=False):
        
        # Convert units [Ms => microHz in frequency]
        self.time    = time.to('Ms').value
        self.cadence = np.diff(time)[0].to('Ms').value
        self.Teff_sun = 5777.
        
        # Load stellar parameters [solar]
        self.Teff = star_params[0].to('K').value
        self.R    = star_params[1].to('R_sun').value
        self.M    = star_params[2].to('M_sun').value
        self.L    = star_params[3].to('L_sun').value
        self.T    = self.Teff / self.Teff_sun
        
        # Random number generator
        self.rng = ut.rng(seed)
                    
        # Solar frequency spectrum: mode line-widths and frequencies:
        # Frequencies of 96 modes from BiSON
        data = np.loadtxt(f'{path}/varsim_mainFitsBiSON.txt')
        freq = data[:,2]

        # Frequency of maximum power and the primary frequency splitting
        # Keldsen & Bedding (1995) eq. 10 and 9:
        numax_sun    = 3140.                                             # [muHz]
        deltanu_sun  = 134.9                                             # [muHz]
        self.numax   = self.M / self.R**2 / np.sqrt(self.T) * numax_sun  # [Sun] 
        self.deltanu = np.sqrt(self.M) * self.R**-1.5 * deltanu_sun      # [Sun]
            
        # Scaling solar distinct pulsation frequencies of the 96 modes of the Sun [microHz]
        self.freq = (freq-numax_sun)/deltanu_sun * self.deltanu + self.numax
        
        # From Chaplin et al. (2009) Eq. 1
        self.eta = np.exp(data[:,6])*np.pi

        
    def init_granulation(self, scaling='Kallinger2014'):
        """Initialize granulation model.
        """
        tau_gran_sun = 375.   # [s]

        # SELECT SCALING RELATION
        
        if scaling == 'KjeldsenBedding2011':  # TODO incomplete model?

            # Timescale from Keldsen & Bedding (2011) Eq. 9 [Sun]
            tau_gran = self.L * self.M**-1 * self.T**-3.5

            # Amplitude from Kallinger et al. 2014 table 3 TODO where is this equation from?
            A_gran_bol = self.L * tau_gran**0.5 * self.M**-1.5 * self.T**-2.75 * 41.

        elif scaling == 'Kallinger2014':
            # The idea to construct the PSD from Kallinger et al. (2014) Eq. 2 using only the
            # granulation component modelled by the contribution of 1-3 super-Lorentzian
            # functions. Here we only use 2 super-Lorentzian functions.

            #A_gran_bol_sun = 41.    # [ppm] Kallinger et al. (2014) 
            
            # Bolometric correction that scales with the Kepler bandpass: Sec. 5.4
            # Originally from Ballot et al. (2011); Michel et al. (2005)
            T0    = 5934.  # [K]
            C_bol = (self.Teff/T0)**0.8

            # Parameters of power law fits to granulation: Sec. 5.2, from Tab. 2:
            # a**2 correspond to the area under the super-Loretzian in PSD, hence,
            # the variance in the time series, and bolmetric correction account for the bandpass.
            # {b1, b2} are the characteristic frequencies defined by the chracteristic timescale tau
            self.a  = C_bol * 3710. * self.numax**-0.613 * self.M**-0.26  # [ppm]
            self.b1 = 0.317 * self.numax**0.970                           # [microHz]
            self.b2 = 0.948 * self.numax**0.992                           # [microHz]

        else:
            errorcode('error', 'Invalid scaling relation!')

        # Return model parameters
        return self.a, self.b1, self.b2
            

    def eval_granulation(self, a=False, b1=False, b2=False):
        """Model stochastic oscillations.
        """
        # Fetch input parameters
        if not a:  a  = self.a
        if not b1: b1 = self.b1
        if not b2: b2 = self.b2
        
        # Prepare PSD model
        Nfreq = int(len(self.time)/2 + 1)
        frequencies = np.arange(float(Nfreq)) / (Nfreq-1) * 0.5 / self.cadence

        # Add Gaussian noise TODO add new numpy rng!
        realPart = np.random.normal(0., 0.5, Nfreq)
        imagPart = np.random.normal(0., 0.5, Nfreq)

        # Construct PSD [ppm^2/muHz] and the full fourier spectrum
        psd1 = ut.superLorentzian(frequencies, b1, a)
        psd2 = ut.superLorentzian(frequencies, b2, a)
        psd  = psd1 + psd2

        fourier = np.sqrt(2. * psd * (Nfreq-1) / self.cadence) * (realPart + imagPart * 1j)
        self.signal_gran = np.real(np.fft.irfft(fourier))

        # Return generated signal
        return self.signal_gran


    def init_oscillations(self, scaling='Corsaro2013'):
        """Model stochastic oscillations.
        """
        # We use the solar frequency spectrum as a template for the pulsations but
        # scale the frequencies and amplitudes according to the scaling relations
        # (we do not scale the mode lifetimes) -> Michel et al. (2009)
        numax_sun      = 3140.                         # [muHz]
        deltanu_sun    = 134.9                         # [muHz]
        tau_puls_sun   = (2.88 * u.d).to('Ms').value   # [Ms]
        A_puls_bol_sun = 3.6                           # [ppm]

        # Handy definitions
        delta_Teff   = self.Teff_sun - self.Teff     # [K]
        numax_frac   = self.numax / numax_sun        
        deltanu_frac = self.deltanu / deltanu_sun
        
        # SELECT SCALING RELATION

        if scaling == 'KB1995Brown1991':
            # According to Corsaro et al. (2013) Eq. 6 [ppm]
            A_puls_bol = numax_frac**-1 * self.T**1.5 * A_puls_bol_sun

        elif scaling == 'KjeldsenBedding1995':
            # According to Kjeldsen and Bedding (1995), Eq. 4 using Eq. 8 [ppm]
            A_puls_bol = self.L * self.T**-1 * self.M**-1 * 4.7 * 550/623

        elif scaling == 'Mosser2010': # TODO Not working yet
            r = 1.5  # free parameter
            tau0 = convert('d','Ms', 2.65)
            # According to Corsaro et al. 2013, equ. (24)
            tau_puls = convert('d', 'Ms', np.exp(delta_Teff / 601.) * 2.65)
            # According to Kjeldsen and Bedding 2011 equ. (6), [ppm]
            A_puls_bol = self.L * (tau_puls/tau0)**0.5 * self.M**-1.5 * (self.T)**-(1.25+r) *3.6
            
        elif scaling == 'Huber2011':
            # According to the relation by Huber et al. 2011b [ppm]
            s = 0.886
            t = 1.89
            r = 2.0
            A_puls_bol = self.L**s * self.M**-t * self.T**(1-r) * A_puls_bol_sun

        elif scaling == 'KjeldsenBedding2011':  # TODO Not working yet
            # According to Corsaro et al. 2013, equ. (24)
            tau_puls = convert('d', 'Ms', np.exp(delta_Teff / 601.) * 2.65)
            r = 2.0
            # According to Kjeldsen and Bedding 2011 equ. (6), [ppm]
            tau0 = convert('d','Ms',2.65)
            A_puls_bol = self.L * (tau_puls/tau)**0.5 * self.M**-1.5 * self.T**-(1.25+r) * 3.6

        elif scaling == 'Corsaro2013huber2011':
            # Reestimated of Huber et al. (2011b)'s values by
            # Corsaro et al. (2013) using Bayesian inference
            s = 0.984
            t = 1.66
            r = 2.79
            # According to the relation by Huber et al. 2011b,
            # and the fit by Corsaro et al. (2013), Eq. (19) [ppm]
            A_puls_bol = (numax_frac**(2*s - 3*t) *
                          deltanu_frac**(-4*s + 4*t) *
                          self.T**(5*s - 1.5*t - r + 0.2) * 3.6)
            
        elif scaling == 'Corsaro2013':
            # According to Corsaro et al. 2013, Eq. 24:
            # NOTE The value of T0 was calibrated using Kepler RGs in the open clusters
            # NGC 6791 and NGC 6819, and a sample of MS and subgiant Kepler field stars.
            T0       = 601.                            # [K]
            tau0     = (2.65 * u.d).to('Ms').value     # [Ms]
            tau_puls = np.exp(delta_Teff / T0) * tau0  # [Ms]
            # According to Corsaro et al. (2013) Eq. 26 (Model 6) [ppm]
            r = -2.8
            t = 1.56
            A_puls_bol = (numax_frac**(2 - 3*t) *
                          deltanu_frac**(4*t - 4) *
                          (tau_puls/tau_puls_sun)**0.5 *
                          self.T**(4.55 - r - 1.5*t) * A_puls_bol_sun)
            
        # Amplitude [ppm]
        self.ampl = A_puls_bol * np.exp(-(self.freq-self.numax)**2/(2*(1.5*self.deltanu)**2))
        
        # Return model parameters
        return self.numax, self.deltanu, self.freq, self.ampl

            
    def eval_oscillations(self, time=False, freq=False, ampl=False, eta=False):
        """Compute time series of stochastically excited damped modes.
        
        Parameters
        ----------
        time : ndarray
            Time points [0..Ntime-1] (unit: e.g. Ms)
        freq : ndarray 
            Oscillation freqs [0..Nmodes-1] (unit: e.g. microHz)
        ampl : ndarray 
            Amplitude of each oscillation mode rms amplitude = ampl / sqrt(2.)
        eta : ndarray
            Damping rates (unit: e.g. (Ms)^{-1})

        Returns
        -------
        signal : ndarray
            Pulsation signal[0..Ntime-1]

        Resources
        ---------
        De Ridder et al., 2006, MNRAS 365, pp. 595-605.
        """
        # Parsing of arguments
        if not time: time = self.time
        if not freq: freq = self.freq
        if not ampl: ampl = self.ampl
        if not eta:  eta  = self.eta

        # Constants
        Ntime = len(time)
        Nmode = len(freq)

        # Set the kick (= reexcitation) timestep to be one 100th of the
        # shortest damping time. (i.e. kick often enough).
        kicktimestep = (1.0 / max(eta)) / 100.0

        # Init start values of amplitudes, and the kicking amplitude
        # so that the amplitude of the oscillator will be on average be
        # constant and equal to the user given amplitude
        amplcos = 0.0
        amplsin = 0.0
        kick_amplitude = ampl * np.sqrt(kicktimestep * eta)

        # Warm up the stochastic excitation simulator to forget the
        # initial conditions. Do this during the longest damping time.
        # But put a maximum on the number of kicks, as there might
        # be almost-stable modes with damping time = infinity
        damp = np.exp(-eta * kicktimestep)
        Nwarmup = min(20000, int(math.floor(1.0 / np.min(eta) / kicktimestep)))
        for i in range(Nwarmup):
            amplsin = damp * amplsin + np.random.normal(np.zeros(Nmode), kick_amplitude)
            amplcos = damp * amplcos + np.random.normal(np.zeros(Nmode), kick_amplitude)

        # Initialize the last kick times for each mode to be randomly chosen
        # a little before the first user time point. This is to avoid that
        # the kicking time is always exactly the same for all of the modes.

        last_kicktime = np.random.uniform(time[0] - kicktimestep, time[0], Nmode)
        next_kicktime = last_kicktime + kicktimestep

        # Generate and return model [ppm -> normalised]
        self.signal_puls = pulsations(time, freq, eta, Ntime, Nmode, amplsin, amplcos,
                                      kicktimestep, kick_amplitude,
                                      last_kicktime, next_kicktime)
        return self.signal_puls

            
# Numba cannot be integrated into a class currently
@njit
def pulsations(time, freq, eta, Ntime, Nmode, amplsin, amplcos,
               kicktimestep, kick_amplitude, last_kicktime, next_kicktime):

    # Prepare for loop
    signal = np.zeros(Ntime)
    for j in range(Ntime):

        # Compute the contribution of each mode separatly
        for i in range(Nmode):

            # Let the oscillator evolve until right before 'time[j]'
            while (next_kicktime[i] <= time[j]):

                deltatime = next_kicktime[i] - last_kicktime[i]
                damp = np.exp(-eta[i] * deltatime)
                amplsin[i] = damp * amplsin[i] + kick_amplitude[i] * np.random.normal(0.,1.)
                amplcos[i] = damp * amplcos[i] + kick_amplitude[i] * np.random.normal(0.,1.)
                last_kicktime[i] = next_kicktime[i]
                next_kicktime[i] = next_kicktime[i] + kicktimestep

            # Now make the last small step until 'time[j]'
            deltatime = time[j] - last_kicktime[i]
            damp = np.exp(-eta[i] * deltatime)
            signal[j] = ( signal[j] + damp * (amplsin[i] * np.sin(2*np.pi*freq[i]*time[j])
                                    + amplcos[i] * np.cos(2*np.pi*freq[i]*time[j])) )

    return(signal)


#==============================================================#
#                         MASSIVE STARS                        #
#==============================================================#

class SurfaceModulations(object):
    """Class to generate variability of roAp stars.

    Rotationally modulated chemcically perculiar A-type (roAp) stars
    are common and readily available objects that efficiently can be 
    used as photometric calibrators in larger surveys. Rotational 
    periods can be drawn randomly from a uniform distribution between
    1 and 3 days. The shape of light curves of these stars is typically
    sinusoidal with frequent contribution of the second harmonic. 
    The amplitude is typically 10-30 mmag in the Kepler passband. 
    We here assume that Kepler passband is representative for the 
    PLATO passband.
    """
    def __init__(self, time, scale=None, seed=None):
        
        self.time  = time
        self.scale = scale       
        self.rng   = ut.rng(seed)
    
        
    def initToyModel(self, period_range=[1,3], amplitude_range=[10,30]):
        """Draw pulsations from uniform distribution.
        Author: Oleg Kochukhov (oleg.kochukhov@physics.uu.se)

        Typical ratational periods [1, 3] days
        Typical amplitude ranges [10, 30] mmag
        """

        # Random value of rotational period
        P = self.rng.uniform(period_range[0], period_range[1])

        # Relative amplitudes between main frequency and harmonic
        A = self.rng.uniform(0, 1)

        # Random phase offset [0, 2*pi] between main frequency and harmonic
        phi = self.rng.uniform(0, 2*np.pi)
        
        # Create light curve
        mag = np.cos(2*np.pi * (self.time/P)) + A * np.cos(4*np.pi * (self.time/P) + phi)
    
        # Apply passband correction
        if self.scale:
            amplitude_range = np.array(amplitude_range) * self.scale

        # Scale amplitude within the range
        scale = self.rng.uniform(amplitude_range[0], amplitude_range[1]) / np.abs(mag.max())
        self.mag = mag * scale

        # Return model parameters
        return [P, phi, A, scale]

        
    def evaluate(self, plot=False):
        """Evaluate and return generated model.
        """        
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(self.time, self.mag, 'k-')
            plt.xlabel('Time [d]')
            plt.ylabel(r'$\mathcal{P}$ [mmag]')
            plt.xlim(self.time.min(), self.time.max())
            plt.tight_layout()
            plt.show()
        
        return self.mag/1e3


#==============================================================#
#                        PULSATING STARS                       #
#==============================================================#

class Pulsator(object):
    """Class to generate time series from list of pulsation modes.
    """
    def __init__(self, time, power, scale=None, seed=None):

        self.time  = time
        self.power = power
        self.scale = scale
        
        # Random number generator
        self.rng = ut.rng(seed)
            

    def download(self, odir, filename):
        """Utility to download data.
        """
        filepath = Path(f'{odir}/{filename}')

        # Check if a file or a folder is requested

        if not filepath.is_file():

            if filepath.suffix in ['.ftr', '.txt']:
                print(f'Downloading {filename}')
                ut.downloadFromFTP(filename=filename, outputDir=odir, server='plato')
        
            elif not filepath.is_dir() and filepath.suffix != '.ftr':
                zipfile = f'{filename}.zip'
                print(f'Downloading {zipfile} files..')
                ut.downloadFromFTP(filename=zipfile, outputDir=odir, server='plato')
                os.system(f'unzip {odir}/{zipfile} -d {odir} > /dev/null')
                os.system(f'rm {odir}/{zipfile}')

        
    def initToyModel(self, freq_range, ampl_range, nmodes=False):
        """Draw pulsations from uniform distribution.

        Parameters
        ----------
        freq_range : list
            Range of pulsation frequencies [fmin, fmax] in unit of [c/d]
        ampl_range : list
            Range of pulsation amplitudes [Amin, Amax] in units of [mag]
        nmodes : int
            Number of pulsation modes to include
        """
        
        # Number of pulsation modes
        if not nmodes:
            nmodes = int(self.rng.normal(50, 5))
        
        # Generate pulsations using uniform distributions
        self.df = pd.DataFrame()
        self.df['freq']  = self.rng.uniform(freq_range[0], freq_range[1], nmodes)
        self.df['ampl']  = self.rng.uniform(ampl_range[0], ampl_range[1], nmodes)
        self.df['phase'] = self.rng.uniform(0, 2*np.pi, nmodes)
        self.starname = 'Toy model'


    def initFromFile(self, odir, sample, starID=None, variable=None):
        """Draw pulsation modes from Kepler/TESS legacies.
        """

        # Select sample
        if sample == 'Gang2020':
            suffix   = 'dat'
            sep      = ' '
            comment  = '#'
            freq_unit = 'c/d'
            ampl_unit = 'norm'
            filename = 'varsource_gdor_gang2020'
            names    = ['freq', 'ampl', 'phase', 'snr']
            
        elif sample == 'Pedersen2021':
            suffix   = 'dat'
            sep      = ' '
            comment  = '#'
            freq_unit = 'day'
            ampl_unit = 'ppm'
            filename = 'varsource_SPB_pedersen2021'
            
        elif sample == 'Bowman2018':
            suffix   = 'txt'
            sep      = '  '
            comment  = None
            freq_unit = 'c/d'
            ampl_unit = 'mmag'
            filename = 'varsource_dsct_bowman2018'
            names    = ['niter', 'freq', 'freq_err', 'ampl', 'ampl_err', 
                        'phase', 'phase_err', 'snr']
            
        elif sample == 'Bodi2023':
            suffix   = 'fou'
            sep      = '  '
            comment  = None
            freq_unit = 'c/d'
            ampl_unit = 'norm'
            names    = ['freq', 'ampl', 'phase']
            if variable == 'RRLyr':
                filename = 'varsource_rrly_bodi2023'
            elif variable == 'Ceph':
                filename = 'varsource_ceph_bodi2023'
            else:
                errorcode('error', 'Not valid variable! Use "RRLyr" or "Ceph"')

        else:
            errorcode('error', f'No sample named {sample}!')
        
        # Download files if not done
        self.download(odir, filename)

        # If requested, select specific star or else do a random draw
        filenames = glob.glob(f'{odir}/{filename}/*.{suffix}')
        if starID is None:
            starfile = Path(self.rng.choice(filenames))
        else:
            starfile = Path(filenames[starID-1])
            
        # Load file containing columns
        if sample == 'Pedersen2021':
            dn = np.loadtxt(starfile, comments=comment, usecols=[0,2,4])
            self.df = pd.DataFrame({'freq':dn[:,0], 'ampl':dn[:,1], 'phase':dn[:,2]})
        else:
            self.df = pd.read_csv(starfile, sep=sep, comment=comment, names=names)

        # Filename of star
        self.starname = f'{sample}: {starfile.name}'

        # Convert freq unit [c/d]
        if freq_unit == 'day':
            self.df.freq = 1 / self.df.freq
                
        # Convert ampl unit [mag]
        if ampl_unit == 'mmag':
            self.df.ampl /= 1e3  
        elif ampl_unit == 'norm':
            self.df.ampl = -2.5*np.log10(1-self.df.ampl)
        elif ampl_unit == 'ppm':
            self.df.ampl = -2.5*np.log10(1-self.df.ampl/1e6)
            
        # Return the star ID
        return starfile.stem
    

    def initMockaGang2020(self, odir):
        """Draw pulsation modes from Kepler GDOR legacy.
        """

        # Download analysis file
        filename = 'varsim_mocka_gdor_gang2020.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)
        
        # Load file containing columns
        dm = pd.read_feather(filepath)
        
        # Generate KDEs
        N_kde     = scipy.stats.gaussian_kde(dm.N)
        P0_kde    = scipy.stats.gaussian_kde(dm.P0)
        dP0_kde   = scipy.stats.gaussian_kde(dm.dP0)
        slope_kde = scipy.stats.gaussian_kde(dm.slope)
        
        # Select number modes (secure at least 5 modes)
        N_min = 5
        N_ran = np.arange(N_min, dm.N.max(), 1)
        N = int(random.choices(N_ran, weights=N_kde(N_ran), k=1)[0])
        
        # Randomly select grid step to 
        n = self.rng.integers(10000, 100000, 1)[0]

        # Prevent unphysical patterns with P_i > 3.3
        P_max = 5
        while P_max > 3.3:

            # Select maximum period from KDE [day]
            P0_ran = np.linspace(dm.P0.min(), dm.P0.max(), n)
            P0 = random.choices(P0_ran, weights=P0_kde(P0_ran), k=1)[0]

            # First period spacing in pattern from KDE [day]
            dP0_ran = np.linspace(dm.dP0.min(), dm.dP0.max(), n)
            dP0 = random.choices(dP0_ran, weights=dP0_kde(dP0_ran), k=1)[0]

            # Select slope from fit to distribution (cf. Fig. 10 of L20)
            a, b, c, d, e = np.array([0.47980586, 1.27007297, 0.44030565, 0.11122096, 0.26489501])
            slope = a * np.exp(-b * P0) + c * np.log10(d * P0) + e

            # Create period-spacing pattern [day]
            P_i = np.array([dP0 * ((1 + slope)**i - 1)/slope + P0 for i in range(N)])

            # Check maximum period
            P_max = P_i.max()
            
        # Draw amplitude below maximum (20 mmag) [mag]
        A_i_ran = np.linspace(0, 0.02, n)
        param = [1.3177087487666639, 2.1808585006453023e-06, 3.156249403328533e-05]
        A_i_fit = ss.lognorm.pdf(A_i_ran, param[0], loc=param[1], scale=param[2]) + 5e-5
        A_i = np.array(random.choices(A_i_ran, weights=A_i_fit, k=N))

        # Max peak amplitude
        n_max = np.argmax(A_i)
        A_max = A_i[n_max]

        # Swap max peak location with offset
        n_off = np.random.randint(-5, 5)
        n_dex = int(N/2 + n_off)
        if n_dex > n_off/2:
            n_dex = int(n_dex - 1) 
        try:
            A_i[n_max] = A_i[n_dex]
            A_i[n_dex] = A_max
        except:
            pass
        
        # Apply passband correction
        if self.scale:
            A_i = (1 - ut.fromMagToFlux(A_i)) * self.scale
            A_i = -2.5 * np.log10(1 + A_i)

        # Create new data frame
        self.df = pd.DataFrame()
        self.df['freq']  = 1 / P_i
        self.df['ampl']  = A_i
        self.df['phase'] = self.rng.uniform(0, 2*np.pi, N)
        self.starname = 'MOCKA: gamma Doradus (Gang+2020)'

        # Return parameters
        return N, P0, dP0, slope, A_max, self.df
    
    
    def initMockaPedersen2021(self, odir):
        """Draw pulsation modes from Kepler SPB star legacy.
        """

        # Download analysis file
        filename = 'varsim_mocka_SPB_pedersen2021.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)

        # Load file containing columns
        dm = pd.read_feather(filepath)
        
        # Generate KDEs
        N_kde     = scipy.stats.gaussian_kde(dm.N)
        P0_kde    = scipy.stats.gaussian_kde(dm.P0)
        dP0_kde   = scipy.stats.gaussian_kde(dm.dP0)
        slope_kde = scipy.stats.gaussian_kde(dm.slope)

        # Select number modes (secure at least 5 modes)
        N_min = 5
        N_max = dm.N.max()
        N_ran = np.arange(N_min, N_max, 1)
        N = int(random.choices(N_ran, weights=N_kde(N_ran), k=1)[0])

        # Randomly select grid step to
        n = self.rng.integers(10000, 100000, 1)[0]

        # Prevent unphysical patterns with P_i > 3.3
        P_max = 4
        while P_max > 3.3:
            
            # First period in pattern from KDE [day]
            P0_ran = np.linspace(dm.P0.min(), dm.P0.max(), n)
            P0 = random.choices(P0_ran, weights=P0_kde(P0_ran), k=1)[0]

            # First period spacing in pattern from KDE [day]
            dP0_ran = np.linspace(dm.dP0.min(), dm.dP0.max(), n)
            dP0 = random.choices(dP0_ran, weights=dP0_kde(dP0_ran), k=1)[0]

            # Select slope from fit to distribution (cf. Fig. 10 of L20)
            # Compared to gDor stars, we here use the KDE
            slope_ran = np.linspace(dm.slope.min(), dm.slope.max(), n)
            slope = random.choices(slope_ran, weights=slope_kde(slope_ran), k=1)[0]

            # Create period-spacing pattern [day]
            P_i = np.array([dP0 * ((1 + slope)**i - 1)/slope + P0 for i in range(N)])

            # Check maximum
            P_max = P_i.max()
            
        # Draw amplitude below maximum (20 mmag) [mag]
        A_i_ran = np.linspace(0, 0.02, n)
        param   = [1.4225080146060183, 8.415648200068788e-07, 0.00012715214085614303]
        A_i_fit = ss.lognorm.pdf(A_i_ran, param[0], loc=param[1], scale=param[2]) + 5e-5
        A_i = np.array(random.choices(A_i_ran, weights=A_i_fit, k=N))
        
        # Max peak amplitude
        n_max = np.argmax(A_i)
        A_max = A_i[n_max]

        # Swap max peak location with offset
        n_off = np.random.randint(-5, 5)
        n_dex = int(N/2 + n_off)
        if n_dex > n_off/2: n_dex = int(n_dex - 1)
        try:
            A_i[n_max] = A_i[n_dex]
            A_i[n_dex] = A_max
        except:
            pass

        # Apply passband correction
        if self.scale:
            A_i = (1 - ut.fromMagToFlux(A_i)) * self.scale
            A_i = -2.5 * np.log10(1 + A_i)
        
        # Create new data frame
        self.df = pd.DataFrame()
        self.df['freq']  = 1 / P_i
        self.df['ampl']  = A_i
        self.df['phase'] = self.rng.uniform(0, 2*np.pi, N)
        self.starname = 'MOCKA: SPB star (Pedersen+2021)'

        # Return parameters
        return N, P0, dP0, slope, A_max, self.df
    

    def initMockaBowman2018(self, odir):
        """Draw pulsations modes from Kepler DSCT legacy.
        """

        # Download file containing all modes of stars
        filename = 'varsim_mocka_dsct_bowman2018.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)
        df = pd.read_feather(filepath)

        # Download file containing number statistics
        filename = 'varsim_mocka_dsct_bowman2018_modes.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)
        dm = pd.read_feather(filepath)

        # Select number modes (secure at least 5 modes)
        N_min = 5
        N_max = dm.N.max()
        N_ran = np.arange(N_min, N_max, 1)
        N_kde = scipy.stats.gaussian_kde(dm.N)
        N = int(random.choices(N_ran, weights=N_kde(N_ran), k=1)[0])

        # Randomly select grid step to
        n = self.rng.integers(10000, 100000, 1)[0]
        
        # Select frequcies from KDE [day]
        f_min = df.freq.min()
        f_max = df.freq.max()
        f_ran = np.linspace(f_min, f_max, n)
        f_kde = scipy.stats.gaussian_kde(df.freq)
        f_i = np.array(random.choices(f_ran, weights=f_kde(f_ran), k=N))

        # Draw amplitude below maximum [mag]
        A_ran = np.linspace(0, df.ampl.max(), n)
        param = [1.2918151399120281, 6.509817422443962e-06, 0.000379812324281023]
        A_fit = scipy.stats.lognorm.pdf(A_ran, param[0], loc=param[1], scale=param[2]) + 1e-5
        A_i = np.array(random.choices(A_ran, weights=A_fit, k=N))

        # Apply passband correction
        if self.scale:
            A_i = (1 - ut.fromMagToFlux(A_i)) * self.scale
            A_i = -2.5 * np.log10(1 + A_i)

        # Swap max peak location if not in [5, 25] c/d
        # This is based on observations from:
        # (Rodríguez & López-González 2000; Rodríguez et al. 2000)
        n_max = np.argmax(A_i)
        if f_i[n_max] < 5 or f_i[n_max] > 25:
            f_i[n_max] = self.rng.integers(5, 25, 1)[0]

        # Create new data frame
        self.df = pd.DataFrame()
        self.df['freq']  = f_i
        self.df['ampl']  = A_i
        self.df['phase'] = self.rng.uniform(0, 2*np.pi, N)
        self.df = self.df.sort_values('freq').reset_index(drop=True)
        self.starname = 'MOCKA: delta Scuti (Bowman+2018)'

        # Return parameters
        return self.df


    def initMockaHeyAerts2024(self, odir):
        """Draw pulsation modes from TESS/Gaia BCEP legacy.
        """

        # Download file containing all modes of stars
        filename = 'varsim_mocka_bcep_heyaerts2024.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)
        df = pd.read_feather(filepath)

        # Download file containing number statistics
        filename = 'varsim_mocka_bcep_heyaerts2024_modes.ftr'
        filepath = Path(f'{odir}/{filename}')
        self.download(odir, filename)
        dm = pd.read_feather(filepath)

        # Select number modes (secure at least 5 modes)
        N_min = 5
        N_max = dm.N.max()
        N_ran = np.arange(N_min, N_max, 1)
        N_kde = scipy.stats.gaussian_kde(dm.N)
        N = int(random.choices(N_ran, weights=N_kde(N_ran), k=1)[0])

        # Randomly select grid step to
        n = self.rng.integers(10000, 100000, 1)[0]
        
        # Select frequcies from KDE [day]
        f_min = df.freq.min()
        f_max = df.freq.max()
        f_ran = np.linspace(f_min, f_max, n)
        f_kde = scipy.stats.gaussian_kde(df.freq)
        f_i = np.array(random.choices(f_ran, weights=f_kde(f_ran), k=N))

        # Draw amplitude below maximum [mag]
        A_ran = np.linspace(0, df.ampl.max(), n)
        param = [1.387736448703769, 4.673526797156983e-05, 0.0008632607679823016]
        A_fit = scipy.stats.lognorm.pdf(A_ran, param[0], loc=param[1], scale=param[2]) + 1e-5
        A_i = np.array(random.choices(A_ran, weights=A_fit, k=N))

        # Apply passband correction
        if self.scale:
            A_i = (1 - ut.fromMagToFlux(A_i)) * self.scale
            A_i = -2.5 * np.log10(1 + A_i)

        # Swap max peak location if not in [5, 25] c/d
        # This is based on observations from:
        # (Rodríguez & López-González 2000; Rodríguez et al. 2000)
        n_max = np.argmax(A_i)
        if f_i[n_max] < 5 or f_i[n_max] > 25:
            f_i[n_max] = self.rng.integers(5, 25, 1)[0]

        # Create new data frame
        self.df = pd.DataFrame()
        self.df['freq']  = f_i
        self.df['ampl']  = A_i
        self.df['phase'] = self.rng.uniform(0, 2*np.pi, N)
        self.df = self.df.sort_values('freq').reset_index(drop=True)
        self.starname = 'MOCKA: beta Cephei (Hey \& Aerts 2024)'

        # Return parameters
        return self.df


    def initMockaBodi2023(self, odir, variable):
        """Draw pulsation modes from TESS RRLYR star legacy.
        """
        suffix   = 'fou'
        sep      = '  '
        comment  = None
        freq_unit = 'c/d'
        ampl_unit = 'norm'
        names    = ['freq', 'ampl', 'phase']
        filename = 'varsource_rrly_bodi2023'

        # Check variable class
        if variable == 'RRLyr':
            filename = 'varsource_rrly_bodi2023'
            self.starname = 'MOCKA: RR Lyr star (Bodi+2023) '
        elif variable == 'Ceph':
            filename = 'varsource_ceph_bodi2023'
            self.starname = 'MOCKA: Cepheid star (Bodi+2023) '
        else:
            errorcode('error', 'Not valid variable! Use "RRLyr" or "Ceph"')
        
        # Download files if not done
        self.download(odir, filename)

        # If requested, select specific star or else do a random draw
        filenames = glob.glob(f'{odir}/{filename}/*.{suffix}')
        starfile = Path(self.rng.choice(filenames))

        # Load data frame
        self.df = pd.read_csv(starfile, sep=sep, comment=comment, names=names)

        # Perturb amplitudes pm 10% with a constant multiplicative shift
        A_corr = self.rng.uniform(0.9, 1.1)
        self.df.ampl *= A_corr

        # Perturb frequencies pm 10% but secure original frequency ratios
        f_corr  = self.rng.uniform(0.9, 1.1)
        f0      = f_corr * self.df.freq.iloc[0]
        f_ratio = self.df.freq / self.df.freq.iloc[0]
        self.df.freq = f0 * f_ratio

        # Apply passband correction
        if self.scale:
            self.df.ampl *= self.scale

        # Convert units
        self.df.ampl = -2.5 * np.log10(1 + self.df.ampl)
            
        # Return parameters
        return starfile.stem, f_corr, A_corr, self.df

    
    def initMockaLPV(self, odir, startype=None):
        """Draw pulsation modes from OGLE survey legacy.
        """

        # Random draw of variable type
        if startype in [None, 'mocka', 'LPV']:
            types = ['Mira', 'SVR', 'OSARG']
            startype = self.rng.choice(types)

        # Select star type
        if startype == 'Mira':
            filename = 'varsim_OGLE_Mira.txt'
        elif startype == 'SVR':
            filename = 'varsim_OGLE_SVR.txt'
        elif startype == 'OSARG':
            filename = 'varsim_OGLE_OSARG.txt'
            
        # Download file containing all modes of stars
        self.download(odir, filename)

        # Load file with pulsators
        filepath = Path(f'{odir}/{filename}')
        dn = np.loadtxt(filepath, comments='#', usecols=[10, 11, 12, 13, 14, 15])

        # Randomly select star 
        i = self.rng.integers(0, dn.shape[0], 1)[0]
        self.df = pd.DataFrame({'freq':1/np.array([dn[i][0], dn[i][2], dn[i][4]]),
                                'ampl':np.array([dn[i][1], dn[i][3], dn[i][5]]),
                                'phase':self.rng.uniform(0, 2*np.pi, len(dn[i][::2]))})

        # Create new data frame
        self.starname = f'MOCKA: LPV of type {startype} (OGLE survey)'

        # Return parameters
        return startype, i, self.df

    
    def evaluate(self, plot=False):
        """Evaluate and return generated model.

        time [day], freq [c/d], ampl [mag], phase [rad]
        """
        return ns.timeSeriesFromFourier(self.time, self.df.freq, self.df.ampl, self.df.phase,
                                        power=self.power, title=self.starname, plot=plot)
   
    
#==============================================================#
#                      ECLIPSING BINARIES                      #
#==============================================================#

class EclipsingBinary(object):
    """Models Eclipsing Binaries (EBs).
    """
    def __init__(self, time, seed=None, verbose=2):

        """Open the HDF5 output file
        """

        self.time = time
        self.rng  = ut.rng(seed)
        self.verbose = verbose


    def read_parameters_hdf5(self, file_name, verbose=False):

        """Copy of method from STARSHADOW:
        Read the full model parameters of the linear, sinusoid
        and eclipse models to an hdf5 file.
        """
        # check some input
        ext = os.path.splitext(os.path.basename(file_name))[1]
        if (ext != '.hdf5'):
            file_name = file_name.replace(ext, '.hdf5')
        # create the file
        with h5py.File(file_name, 'r') as file:
            identifier = file.attrs['identifier']
            description = file.attrs['description']
            data_id = file.attrs['data_id']
            date_time = file.attrs['date_time']
            t_tot = file.attrs['t_tot']
            t_mean = file.attrs['t_mean']
            t_mean_s = file.attrs['t_mean_s']
            t_int = file.attrs['t_int']
            n_param = file.attrs['n_param']
            bic = file.attrs['bic']
            noise_level = file.attrs['noise_level']
            # orbital period and time of deepest eclipse
            p_orb = np.copy(file['p_orb'])
            t_zero = np.copy(file['t_zero'])
            # the linear model
            # y-intercepts
            const = np.copy(file['const'])
            c_err = np.copy(file['c_err'])
            c_hdi = np.copy(file['c_hdi'])
            # slopes
            slope = np.copy(file['slope'])
            sl_err = np.copy(file['sl_err'])
            sl_hdi = np.copy(file['sl_hdi'])
            # sector indices
            i_sectors = np.copy(file['i_sectors'])
            # the sinusoid model
            # frequencies
            f_n = np.copy(file['f_n'])
            f_n_err = np.copy(file['f_n_err'])
            f_n_hdi = np.copy(file['f_n_hdi'])
            # amplitudes
            a_n = np.copy(file['a_n'])
            a_n_err = np.copy(file['a_n_err'])
            a_n_hdi = np.copy(file['a_n_hdi'])
            # phases
            ph_n = np.copy(file['ph_n'])
            ph_n_err = np.copy(file['ph_n_err'])
            ph_n_hdi = np.copy(file['ph_n_hdi'])
            # passing criteria
            passed_sigma = np.copy(file['passed_sigma'])
            passed_snr = np.copy(file['passed_snr'])
            passed_b = np.copy(file['passed_b'])
            passed_h = np.copy(file['passed_h'])
            # the physical eclipse model parameters
            ecosw = np.copy(file['ecosw'])
            esinw = np.copy(file['esinw'])
            cosi = np.copy(file['cosi'])
            phi_0 = np.copy(file['phi_0'])
            log_rr = np.copy(file['log_rr'])
            log_sb = np.copy(file['log_sb'])
            # some alternate parametrisations
            e = np.copy(file['e'])
            w = np.copy(file['w'])
            i = np.copy(file['i'])
            r_sum = np.copy(file['r_sum'])
            r_rat = np.copy(file['r_rat'])
            sb_rat = np.copy(file['sb_rat'])
            # eclipse timings
            t_1 = np.copy(file['t_1'])
            t_2 = np.copy(file['t_2'])
            t_1_1 = np.copy(file['t_1_1'])
            t_1_2 = np.copy(file['t_1_2'])
            t_2_1 = np.copy(file['t_2_1'])
            t_2_2 = np.copy(file['t_2_2'])
            t_b_1_1 = np.copy(file['t_b_1_1'])
            t_b_1_2 = np.copy(file['t_b_1_2'])
            t_b_2_1 = np.copy(file['t_b_2_1'])
            t_b_2_2 = np.copy(file['t_b_2_2'])
            depth_1 = np.copy(file['depth_1'])
            depth_2 = np.copy(file['depth_2'])
            # variability to eclipse depth ratios
            ratios_1 = np.copy(file['ratios_1'])
            ratios_2 = np.copy(file['ratios_2'])
            ratios_3 = np.copy(file['ratios_3'])
            ratios_4 = np.copy(file['ratios_4'])

        sin_mean = [const, slope, f_n, a_n, ph_n]
        sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
        sin_hdi = [c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi]
        sin_select = [passed_sigma, passed_snr, passed_b, passed_h]
        ephem = [p_orb[0], t_zero[0]]
        ephem_err = [p_orb[1], t_zero[1]]
        ephem_hdi = [p_orb[2:4], t_zero[2:4]]
        phys_mean = np.array([ecosw[0], esinw[0], cosi[0], phi_0[0], log_rr[0], log_sb[0],
                              e[0], w[0], i[0], r_sum[0], r_rat[0], sb_rat[0]])
        phys_err = np.array([ecosw[1], esinw[1], cosi[1], phi_0[1], log_rr[1], log_sb[1],
                             e[1], w[1], i[1], r_sum[1], r_rat[1], sb_rat[1]])
        phys_hdi = np.array([ecosw[2:4], esinw[2:4], cosi[2:4], phi_0[2:4],
                             log_rr[2:4], log_sb[2:4], e[2:4], w[2:4], i[2:4],
                             r_sum[2:4], r_rat[2:4], sb_rat[2:4]])
        timings = np.array([t_1[0], t_2[0], t_1_1[0], t_1_2[0], t_2_1[0], t_2_2[0],
                            t_b_1_1[0], t_b_1_2[0], t_b_2_1[0], t_b_2_2[0],
                            depth_1[0], depth_2[0]])
        timings_err = np.array([t_1[1], t_2[1], t_1_1[1], t_1_2[1], t_2_1[1], t_2_2[1],
                                t_b_1_1[1], t_b_1_2[1], t_b_2_1[1], t_b_2_2[1],
                                depth_1[1], depth_2[1]])
        timings_hdi = np.array([t_1[2:4], t_2[2:4], t_1_1[2:4], t_1_2[2:4],
                                t_2_1[2:4], t_2_2[2:4], t_b_1_1[2:4], t_b_1_2[2:4],
                                t_b_2_1[2:4], t_b_2_2[2:4], depth_1[2:4], depth_2[2:4]])
        timings_indiv_err = np.array([t_1[4], t_2[4], t_1_1[4], t_1_2[4], t_2_1[4], t_2_2[4],
                                      t_b_1_1[4], t_b_1_2[4], t_b_2_1[4], t_b_2_2[4],
                                      depth_1[4], depth_2[4]])
        var_stats = [ratios_1[0], ratios_2[0], ratios_3[0], ratios_4[0],
                     ratios_1[1:], ratios_2[1:], ratios_3[1:], ratios_4[1:]]
        stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
        text = [identifier, data_id, description, date_time]
        # put everything in a dict
        results = {'sin_mean': sin_mean, 'sin_err': sin_err, 'sin_hdi': sin_hdi,
                   'sin_select': sin_select,
                   'ephem': ephem, 'ephem_err': ephem_err, 'ephem_hdi': ephem_hdi,
                   'phys_mean': phys_mean, 'phys_err': phys_err, 'phys_hdi': phys_hdi,
                   'timings': timings, 'timings_err': timings_err, 'timings_hdi': timings_hdi,
                   'timings_indiv_err': timings_indiv_err,
                   'var_stats': var_stats, 'stats': stats, 'i_sectors': i_sectors,'text': text}
        if verbose:
            print(f'Loaded analysis file with identifier: {identifier}, created on {date_time}. \n'
                  f'data_id: {data_id}. Description: {description} \n')
        return results

        
    def initIJspeert2023(self, odir, starID=None):

        """Draw frequencies from Kepler g-Dor legacy.
        """

        # Name of folder on FTP server
        filename = 'varsource_ebs_ijspeert2021'
        dataDir  = Path(f'{odir}/{filename}')
        
        # Select a random object from the list and load Fourier data
        if not dataDir.is_dir():
            zipfile = f'{filename}.zip'
            if self.verbose > 1:
                print(f'Downloading {zipfile} files..')
            ut.downloadFromFTP(filename=zipfile, outputDir=odir, server='plato')
            os.system(f'unzip {odir}/{zipfile} -d {odir}')
            os.system(f'rm {odir}/{zipfile}')
            
        # If requested, select specific star or else do a random draw
        folders = glob.glob(f'{odir}/{filename}/*')
        starDir = Path(self.rng.choice(folders))
                    
        # Load file containing columns
        starfile = starDir / f'{starDir.name}_2.hdf5' 
        result   = self.read_parameters_hdf5(starfile)
        data     = result['sin_mean']
        self.freq  = data[2]  # [c/d]
        self.ampl  = data[3]  # [mag]
        self.phase = data[4]  # [rad]
        self.starname = str(starDir.stem)

        # Extract period if avilable
        starfile = starDir / f'{starDir.name}_5.hdf5'
        try:
            result = self.read_parameters_hdf5(starfile)
            params = result['ephem']
            P = params[0]
        except:
            P = None

        return self.starname, P
        
    
    def evaluate(self, plot=False):
        """Evaluate and return generated model.
        """
        return ns.timeSeriesFromFourier(self.time, self.freq, self.ampl, self.phase,
                                        plot=plot, title=self.starname)

    
#==============================================================#
#                         SMBH BINARIES                        #
#==============================================================#
        
class SMBHB(object):
    """Models for Super Massive Black Hole Binaries (SMBHBs).
    """

    def __init__(self, time, seed=None):
        """Initialize class
        """
        self.time = time
        self.seed = seed
        self.rng  = ut.rng(seed)


    def __del__(self):
        """Destructor
        """
        pass


    def initToyModelSpikey(self):
        """Initialise toy model of Spikey.
        """
        A   = 0.04         # [mag]
        P   = 2 * 365.25   # [days]
        phi = 0.1 * np.pi  # [rad] #np.random.uniform(low=0, high=2*np.pi) 
        tscale    = 10     # [days] #np.ones(len(max)) * 10
        amplitude = 0.08   # [mag]
        return

    
    def initToyModel(self):
        """Simplified description of a SMBH binary system.

        The model consist of two components:
          1) Doppler beaming
          2) Gravitational lens
        
        Assumptions
        -----------
        - Circular orbits (e = 0)
        - The orbital period is drawn from the uniform distribution [1.5, 2.5] years
        - The beaming amplitude is drawn from the uniform distribution [0, 0.1] mag
        - The lensing amplitude is twice that of the beaming amplitude
        - The lensing duration scales linearly with the orbital period [5, 12] days
        """

        # Orbital period [days]
        P_min = 1.5
        P_max = 2.5
        year  = 365.25
        P = self.rng.uniform(P_min, P_max) * year

        # Beaming amplitude [mag]
        A_beam = self.rng.uniform(0, 0.1)
        
        # Lens paramters [mmag and days]
        A_lens = 2 * A_beam

        # Lens time maximum [days]
        phi_lens  = self.rng.uniform(0, 2*np.pi)
        tmax_lens = P * (1 - phi_lens/(2*np.pi))
        
        # Lens duration [days]
        tdur_lens = ut.evalLinReg(np.array([P_min, P_max]),
                                  np.array([5, 10]), P/year)

        # Return parameters
        return P, A_beam, A_lens, phi_lens, tmax_lens, tdur_lens


    def evalToyLensing(self, tmax=False, tscale=False, ampl=30, asym=1):
        """Simple analytic model for lensing event.
        
        Parameters
        ----------
        tmax : float, ndarray
            Time of maximum intensity due to lensing [days]
        tscale : float, ndarray
            Time scale duration of the flare(s) [days]
        ampl : float, ndarray
            Amplitude of the flare(s) [mmag]
        asym : float, ndarray
            Asymmetry factor of the flare(s)
        """

        # Convert units of input parameters
        time = self.time
        flux = np.zeros_like(time)

        # Secure that single flare works
        try: len(tmax)
        except: tmax = [tmax]
        
        # Loop over each flare event
        
        for m in range(len(tmax)):

            # Start and end of flare event
            t0 = (time[0]  - tmax[m])
            t1 = (time[-1] - tmax[m])
            dt = np.diff(time)[0]

            # Time array during flare event
            tn = np.arange(t0, t1, dt)
            t = tn/tscale

            # Model parameters of flare
            b = -1.941 - 0.175 + 2.246 + 1
            c = 1 - 0.689

            # Loop over every time-step in the flare time interval

            for i in range(len(t)):

                # Rise of lensing
                if t[i] <= 0:
                    flux[i] += 0.689 * np.exp(+1.6 * t[i])

                # Decay of lensing
                elif t[i] > 0:
                    flux[i] += 0.689 * np.exp(-1.6 * t[i])

                # Outside lens
                else:
                    flux[i] += 0

        # Return relative flux
        return flux * ampl + 1


    def evalToyModel(self, P, A_beam, A_lens, phi, tmax, tdur):
        """Uniform distribution of toy model.
        """
        
        # Model doppler beaming effect
        flux_beam = A_beam * np.sin(2*np.pi * (self.time/P) + phi) + 1

        # Model all lesing evens with P < duration of observation
        flux_lens = np.ones_like(flux_beam)
        for tm in [tmax-P, tmax, tmax+P]:
            flux = self.evalLensingEvent(tm, tdur, A_lens, asym=1)
            flux_lens += flux - np.ones_like(flux_beam)

        # Signals
        self.flux = flux_beam + flux_lens - 1
        self.flux_beam = flux_beam
        self.flux_lens = flux_lens
            
        # Return models
        return self.flux, self.flux_beam, self.flux_lens

    #--------------------------------------------------- Start of physical model

    
    def model_params(object):
        """Load model parameters.
        """
        def __init__(self):

            # Observational parameters
            self.t0 = 0    
            self.z  = 0

            # Orbital parameters
            self.P = 1     # Observed period [days]
            self.i = 0.01  # Inclination [deg]
            self.e = 0     # Eccentricity [0,1]
            self.w = 0     # Argument of periapse [0, 360] [deg]

            # Physical parameters
            self.M = 1e9
            self.q = 0.1
            self.L = 0.1

            # Damped Random Walk (DRW) parameters
            self.tau   = 50 * u.d  # Variability time scale [d] 
            self.sigma = 300       # [ppm]

            # Doppler boosting parameters
            self.alpha = 2.09  # Spectral slope
            self.v_z   = 0     # Relative motion of frames [m/s]
        
        # # Observational parameters
        # self.t0 = 0    
        # self.z  = 0
        
        # # Orbital parameters
        # self.P = 1     # Observed period [days]
        # self.i = 0.01  # Inclination [deg]
        # self.e = 0     # Eccentricity [0,1]
        # self.w = 0     # Argument of periapse [0, 360] [deg]

        # # Physical parameters
        # self.M = 1e9
        # self.q = 0.1
        # self.L = 0.1

        # # Damped Random Walk (DRW) parameters
        # self.tau   = 50 * u.d  # Variability time scale [d] 
        # self.sigma = 300       # [ppm]

        # # Doppler boosting parameters
        # self.alpha = 2.09  # Spectral slope
        # self.v_z   = 0     # Relative motion of frames [m/s]


    
    def initPhysicalModel(self, t0, P, M, q, i, e, w, L, z, verbose=False):
        """Initialise physical model.

        This function converts everything to CGS units.
        """
        # Default Parameters
        self.t0 = t0.cgs
        self.P  = P.cgs
        self.M  = M.cgs
        self.q  = q
        self.i  = i.to('rad')
        self.e  = e
        self.w  = w.to('rad')
        self.L  = L
        self.z  = z

        # Constants
        self.floor = 1e-15
        self.M1 = self.M / (1 + self.q)
        self.M2 = self.M - self.M1
        
        # Orbital period in binary rest frame [s] 
        self.T = self._period_observed(self.P, self.z)

        # Semi-major axis [cm]        
        self.a  = self._semimajor_axis(self.P, self.M)
        self.a1 = self.a * self.M2 / self.M
        self.a2 = self.a * self.M1 / self.M

        # Show parameters to screen
        if verbose:
            ut.errorcode('message', 'Input parameters:')
            print(f'Time of ephemeris,            t0 : {self.t0.to("yr"):.3f}')
            print(f'Orbital period in rest frame, P  : {self.P.to("yr"):.3f}')
            print(f'Orbital period in obs. frame, T  : {self.T.to("yr"):.3f}')
            print(f'Mass of primary,              M1 : {self.M1.to("M_sun")/1e6:.3f} x 1e6')
            print(f'Mass of secondary,            M2 : {self.M2.to("M_sun")/1e6:.3f} x 1e6')
            print(f'Mass total,       (M1 + M2) = M  : {self.M.to("M_sun")/1e6:.3f} x 1e6')
            print(f'Mass ratio,       (M2 / M1) = q  : {self.q:.4f}')
            print(f'Inclination to LOS,           i  : {self.i.to("deg"):.2f}')
            print(f'Eccentricity,                 e  : {self.e:.2f}')
            print(f'Argument of periapse,         w  : {self.w.to("deg"):.2f}')
            print(f'Luminosity ratio, L2/(L2+L1)= L  : {self.L:.3f}')
            print(f'Redshift,                     z  : {self.z:.3f}')
            # ut.errorcode('message', 'Orbital parameters:')
            # print(f'Semi-major axis of primary,   a1 : {self.a1.to("AU"):.2f}')
            # print(f'Semi-major axis of secondary, a2 : {self.a2.to("AU"):.2f}')
            # print(f'Semi-major axis of binaries,  a  : {self.a.to("AU"):.2f}')

        # Remove angle units [rad]
        self.omega = np.pi / 2
        self.w    = self.w.value
        self.i    = self.i.value
        self.sini = np.sin(self.i)
        self.cosi = np.cos(self.i)

        # Corse time grid for speed (interpolate back later)
        dt = 3600  # [s]
        time0 = self.time[0].to('s').value
        time1 = self.time[-1].to('s').value
        self.t = np.arange(time0, time1, dt) * u.s
        
        # Mean anomaly [rad/s]
        self.fm = self._mean_anomaly(self.t, self.t0, self.T).value

    #--------------------------------------------------- Public API
    
    def quasar_variability(self, tau, sigma, verbose=False,
                           plot=False, plot_psd=False):
        """Initialise ANG intrinsic variability.

        This model uses a damped random walk (i.e. red noise) descroption
        to compute the quasar variability.

        Parameters
        ----------
        tau : ndarray
            Time scale tau of each red noise component [d]
        sigma : ndarray 
            Variation scale of each red noise component [ppm]
        verbose : bool
            Option to print info to bash
        plot : bool
            Show normalised light curve [pp1]
        plot_psd : bool
            Show Power Spectral Density (PSD) plot.

        Returns
        -------
        Q : Signal containing all red noise components [pp1]
        """
        tau   = np.array([tau.to('d').value])
        sigma = np.array([sigma])

        # Show parameters to screen
        if verbose:
            ut.errorcode('message', '\nDRW parameters:')
            print(f'Damping timesclae,           tau : {tau[0]:.1f} d')
            print(f'Std of variability,        sigma : {sigma[0]:.1f} ppm')

        # Run model (NOTE we use original time array here)
        time  = self.time.to('d').value
        self.Q = ns.modelRedNoise(time, tau, sigma, seed=self.seed) * 1e-6 + 1

        if plot:
            fig = plt.figure(figsize=(9,5))
            plt.plot(time, self.Q, c='tomato')
            plt.xlabel(r"Time [day]")
            plt.ylabel(r"Relative flux")
            plt.xlim(0, time[-1])
            plt.tight_layout()
            plt.show()

        if plot_psd:
            dt = np.diff(time)[0]
            Nfreq = time.shape[0]
            freq  = np.arange(float(Nfreq)) / (Nfreq-1) / (2*dt)
            PSD   = ns.modelRedNoisePSD(freq, tau, sigma)
            # Show PSD plot
            fig = plt.figure(figsize=(9,5))
            for i in range(len(tau)):
                plt.loglog(freq, PSD, c='tomato', lw=2)
            plt.xlabel(r"Frequency [c/d]")
            plt.ylabel(r"PSD [ppm$^2$ d$^2$]")
            plt.xlim(np.min(freq), np.max(freq))
            plt.tight_layout()
            
        return self.Q

    
    def doppler_boosting(self, alpha=2, v_z=0, verbose=False, plot=False):
        """Model relativistic Doppler boosting.

        This function initialise the Doppler boosting model of a
        two-body gravitationally bound system.

        Parameters
        ----------
        alpha : float 
            Spectral index of both mini-discs
        v_z : float
            Barycentric velocity of system [cm/s]

        Return
        ------
        Relative flux time series of doppler boosting signal.
        """
                
        # The RV semi-amplitude of secondary [cm/s]
        K1,K2 = self._rv_semiamplitude(self.P, self.M1, self.M, self.q, self.a, self.i, self.e)
        print(K1, K2)
        # Show parameters to screen
        if verbose:
            ut.errorcode('message', '\nDoppler boosting parameters:')
            print(f'Spectral index minidiscs,  alpha : {alpha:.2f}')
            print(f'RV semi-amplitude of primary, K1 : {K1.to("km/s"):.2f}')
            print(f'RV semi-amplitude of second., K2 : {K2.to("km/s"):.2f}')

        # Check parameters
        try:
            E = self.E
            f = self.f
            r = self.r            
        except AttributeError:
            E = np.array([self._eccentric_anomaly(m, self.e) for m in self.fm])
            f = self._true_anomaly(E, self.e)
            r = self._radial_vector(E, self.e, self.a)

        # Projection of the velocity vector on to the line of sight [cm/s]
        vr1, vr2 = self._rv_vector(v_z, K1, K2, f, self.e, self.w)

        # Relativistic doppler boosting [pp1]
        arg1 = (self.M2 / self.M)**2 * c.G.cgs * self.M * (2/r - 1/self.a) / c.c.cgs**2
        arg2 = (self.M1 / self.M)**2 * c.G.cgs * self.M * (2/r - 1/self.a) / c.c.cgs**2
        v1_sqr = np.minimum(arg1.value, 1-self.floor)
        v2_sqr = np.minimum(arg2.value, 1-self.floor)
        gamma1 = 1 / np.sqrt(1 - v1_sqr)
        gamma2 = 1 / np.sqrt(1 - v2_sqr)
        self.d1 = 1 / (gamma1 * (1 - vr1/c.c.cgs))**(3 - alpha)
        self.d2 = 1 / (gamma2 * (1 - vr2/c.c.cgs))**(3 - alpha)
        self.d  = (1 - self.L) * self.d1 + self.L * self.d2

        if plot:
            t = self.t.to('d').value
            fig = plt.figure(figsize = (9, 5))
            plt.plot(t, self.d1, ':',  c='orange', label=r"$D_1$")
            plt.plot(t, self.d2, '-.', c='orange', label=r"$D_2$")
            plt.plot(t, self.d,  '-',  c='orange', label=r"$D$")
            plt.xlabel(r"Time [day]")
            plt.ylabel(r"Relative flux")
            plt.xlim(0, t[-1])
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Interpolate back to original time grid
        D_interp  = make_interp_spline(self.t.cgs.value, self.d,  k=3)        
        D1_interp = make_interp_spline(self.t.cgs.value, self.d1, k=3)
        D2_interp = make_interp_spline(self.t.cgs.value, self.d2, k=3)
        self.D  = D_interp( self.time.cgs.value)
        self.D1 = D1_interp(self.time.cgs.value)
        self.D2 = D2_interp(self.time.cgs.value)        

        return self.D, self.D1, self.D2

    
    def gravitational_lensing(self, J, wvl, u_max=30, u_num=300, v_num=100,
                              verbose=False, plot=False):
        """Model gravitational self-lensing.

        This function calculates the magnification of a SMBHB binary pair
        accretion discs during their orbital phase. The model return both
        the magnification in the point source (PS) and finite source (FS)
        limit.

        Parameters
        ----------
        J : float, astropy.unit [deg, rad]
        wvl : float, astropy.unit [nm, cm, m]
        u_max : Maximum number of Einstein radii to compute finite lensing.
        u_num : Number of grid points in u plane.
        v_num : Number of grid points in v plane.
        verbose : bool
        plot : bool

        Returns
        -------
        Magnification of primary and secondary: M1_ps [pp1], M2_ps [pp1]
        """
        J   = J.to('rad')
        wvl = wvl.cgs
        
        # Print to bash
        if verbose:
            RS1, RS2 = self._radius_schwarzchild(self.M, self.q)
            ut.errorcode('message', '\nSelf-lensing parameters:')
            print(f'Inclination of mini-disc,    J   : {J.to("deg"):.1f}')            
            print(f'Inclination of mini-disc,    wvl : {wvl.to("nm"):.0f}')
            print(f'Schwarchild radius primary,  Rs1 : {RS1.to("R_sun"):.2f}')
            print(f'Schwarchild radius second.,  Rs2 : {RS2.to("R_sun"):.2f}')
                
        # Find anomalies
        try:
            E = self.E
            f = self.f
        except AttributeError:
            E = np.array([self._eccentric_anomaly(m, self.e) for m in self.fm])
            f = self._true_anomaly(E, self.e)

        # Find cartesian position vectors
        x1, y1, z1, x2, y2, z2 = self._xyz_orbital_plane(self.a1.value, self.q, self.i,
                                                         self.e, self.w, E, f)

        # Switch to select secondary as lens (or primary as source)
        self.flip = (z1 < 0)

        # Point-source magnification
        delta = self._angular_separation_xy(x1, x2, y1, y2)
        theta = self._angular_einstein_radius(z1, z2, self.M1.value, self.M2.value)
        u = delta / (theta + self.floor)
        self.M_ps = self._magnification_point(u)

        # Finite-source magnification
        # delta_u = u_max / u_num        
        # delta_v = 2 * np.pi / v_num
        # u_array = np.linspace(0, u_max, u_num)
        # v_array = np.linspace(0, 2*np.pi, v_num)
        # u_grid, v_grid = np.meshgrid(u_array, v_array)
        # u0 = u
        # v0 = np.arctan(self.sini * np.tan(self.fm))
        # r0 = np.array([self._radius_disc(u_grid, v_grid, u, v, theta_E, J.value)
        #                for u, v, theta_E in zip(u0, v0, theta)])
        # flux = np.array([self._flux_disc(r, z, wvl.value, self.a, self.M, self.q)
        #                  for r,z in zip(r0, z1)])
        # M_fs = self._magnification_finite(flux, u_grid, delta_u, delta_v)
        
        # Compute point-source magnification with luminosity ratio
        x = self.flip
        D1 = D2 = np.ones_like(u)
        self.m_ps    = (1 - self.L) * D1                   + self.L * D2 * self.M_ps
        self.m_ps[x] = (1 - self.L) * D1[x] * self.M_ps[x] + self.L * D2[x]        
        # self.m_fs    = (1 - self.L) * D1                   + self.L * D2 * self.M_fs
        # self.m_fs[x] = (1 - self.L) * D1[x] * self.M_fs[x] + self.L * D2[x]

        if plot:
            t = self.t.to('d').value
            plt.figure(figsize=(9,5))
            plt.plot(t, self.M_ps, '-.', c='royalblue', label=r"$\mathcal{M}^{\rm PS}$")
            plt.plot(t, self.m_ps, '-',  c='royalblue', label=r"$\mathcal{m}^{\rm PS}$")
            # plt.plot(t, self.M_fs, '-.', c='b', label=r"$\mathcal{M}^{\rm FS}$")
            # plt.plot(t, self.m_fs, '-',  c='b', label=r"$\mathcal{m}^{\rm FS}$")
            plt.xlabel(r"Time [day]")
            plt.ylabel(r"Relative flux")
            plt.xlim(0, t[-1])
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.M_ps
            
    
    def evalPhysicalModel(self, plot=False, ofile_fig=False):
        """Evaluate physical SMBH binary model.
        """
        time = self.time.to('d').value

        # Check DRW model
        try:
            Q = self.Q
        except:
            Q = np.ones_like(t)

        # Check boosting model
        try:            
            D  = self.d
            D1 = self.d1
            D2 = self.d2
        except:
            D  = np.ones_like(t)
            D1 = np.ones_like(t)
            D2 = np.ones_like(t)

        # Check lensing model
        try:
            M_ps = self.M_ps
            m_ps = self.m_ps
        except:
            M_ps = np.ones_like(t)
            m_ps = np.ones_like(t)

        # Combute boosting+lensing model
        x = self.flip
        flux    = (1 - self.L) * D1             + self.L * D2 * M_ps
        flux[x] = (1 - self.L) * D1[x]* M_ps[x] + self.L * D2[x]
        
        # Interpolate back to original time grid and add DRW model
        D_interp    = make_interp_spline(self.t.to('s').value, D,    k=3)
        m_ps_interp = make_interp_spline(self.t.to('s').value, m_ps, k=3)
        flux_interp = make_interp_spline(self.t.to('s').value, flux, k=3)
        D    = D_interp(self.time.to('s').value)
        m_ps = m_ps_interp(self.time.to('s').value)
        flux = flux_interp(self.time.to('s').value)
        flux += (Q - 1)
        
        if plot:
            fig = plt.figure(figsize = (9, 5))
            plt.plot(time, Q,     color='tomato',    label="DRW",      lw=0.8)
            plt.plot(time, D,     color='orange',    label="Beaming",  lw=1.8)
            plt.plot(time, m_ps,  color='royalblue', label="Lensing",  lw=1.8)
            plt.plot(time, flux,  color='k',         label="Combined", lw=0.8)
            plt.xlabel(r"Time [day]")
            plt.ylabel(r"Relative flux")
            plt.xlim(0, time[-1])
            plt.legend()
            plt.tight_layout()
            plt.show()

            if ofile_fig:
                fig.savefig(ofile_fig, bbox_inches='tight', dpi=300)
            
        return flux, Q, D, m_ps
        
    #--------------------------------------------------- Internal methods

    def _period_observed(self, P, z):
        """Orbital period in observers frame [s].
        """
        return P * (1 + z)

    
    def _semimajor_axis(self, P, M):
        """Semi-major axis in binary rest frame [cm].
        """
        return (c.G.cgs * M * P**2 / (4 * np.pi**2))**(1/3)     


    def _mean_anomaly(self, t, t0, T):
        """Determine the mean anomaly, fm [rad].
        
        Parameters
        ----------
        t : Time array
        T : Orbital period (observed)
        """
        return 2 * np.pi * (t - t0) / T


    def _eccentric_anomaly(self, fm, e):
        """Determine the Eccentric anomaly, E.

        We here solve Kepler's equation (M = E – e sin E) and use the
        Newton-Raphson method to obtain the eccentric anomaly E.
        """
        f = lambda E: E - e * np.sin(E) - fm
        E = root_scalar(f, x0=fm, x1=fm + 0.1, method='secant').root
        return E
    
    #--------------------------------
    
    def r_ISCO(self, m):
        return 6 * self.G * m / self.cc**2

    def accretion_rate(self, m, radiative_efficiency=0.1):
        return 2.26*10**-2 * (radiative_efficiency/0.1)**-1 * (m/10**6) / ut.year()

    def reduced_mass (self):
        return self.q * self.M / (1 + self.q)
        
    def disc_temperature(self, q, r):
        a_rate = self.accretion_rate(self.Mq)
        r_isco = self.r_ISCO(self.Mq)
        power = 3 * self.G * self.Mq * a_rate * (1 - np.sqrt(r_isco/r)) / (8 * np.pi * r**3)
        return (power / self.sigma)**0.25  

    def planck_wavelength(self, temp):
        numerator   = 2 * self.h * self.cc**2
        exponent    = (self.h * self.cc) / (self.wvl * self.k_B * temp)
        denominator = (self.wvl**5) * (np.exp(exponent) - 1)
        return numerator / denominator

    def flux(self, r):
        r_isco = self.r_ISCO(self.Mq)
        condition1 = (r_isco < r) & (r < 0.27 * self.a * self.q**0.3)
        condition2 = np.pi * self.planck_wavelength(self.disc_temperature(self.q, r))
        return np.where(condition1, condition2, 0)
    
    def einstein_radius(self, t):
        """Einstein radius of primary and secondary [cm].
        """        
        sin_phase_p = np.sin(2*np.pi * t/self.T)
        sin_phase_s = np.sin(-np.pi + 2*np.pi * t/self.T)
        RE_p = np.sqrt(2 * self.RS_p * self.a * np.cos(self.I) * sin_phase_p)
        RE_s = np.sqrt(2 * self.RS_s * self.a * np.cos(self.I) * sin_phase_s)
        return RE_p, RE_s

    def position_uv(self, t):
        """Complex components of u (projected  binary seperation) [RE]
        """
        phi = 2 * np.pi * t / self.T
        if 0 < t/self.T < 0.5:
            phase = np.sqrt(np.cos(phi)**2 + np.sin(self.I)**2 * np.sin(phi)**2)
            u_0 = self.a / self.einstein_radius(t)[0] * phase
        else:
            phase = np.sqrt(np.cos(-np.pi+phi)**2 + np.sin(self.I)**2 * np.sin(-np.pi+phi)**2)
            u_0 = self.a / self.einstein_radius(t)[1] * phase
        v_0 = np.arctan(np.sin(self.I) * np.tan(phi))
        return u_0, v_0 

    def magnification_point_u(self, u):
        """Dummy function to calculate finite source magnification.
        """
        return (u**2 + 2) / (np.sqrt(u**2 + 4))

    def disc_radius(self, u, v, u0, v0, r_E):
        r_star  = np.sqrt(u0**2 + u**2 - 2 * u0 * u * np.cos(v - v0)) * r_E 
        sin_phi = (u * np.sin(v) - u0 * np.sin(v0)) / np.sqrt((u * np.sin(v) - u0 * np.sin(v0))**2 + (u * np.cos(v) - u0 * np.cos(v0))**2)
        phi = np.arcsin(sin_phi)
        r = r_star * np.sqrt(np.cos(phi)**2 + np.sin(phi)**2 / (np.cos(np.pi/2 - self.J)**2))
        return r, r_star, phi

    def magnification_point(self, u): #point-source magnification
        return (u ** 2 + 2) / (u * np.sqrt(u**2 + 4))



    def run_model(self, time, wvl, z, T, I, J, M, q, u_max=5, u_grid=300, v_grid=100):

        # Grid for accretion disc 
        u_array = np.linspace(0, u_max, u_grid)
        delta_u = u_max / u_grid
        v_array = np.linspace(0, 2 * np.pi, v_grid)
        delta_v = 2 * np.pi / v_grid
        u_grid, v_grid = np.meshgrid(u_array, v_array)

        # Calculate self-lensing in phase
        mag_e = []
        mag_p = []

        for t in tqdm(time, desc="Processing time steps"):
            r_E = self.einstein_radius(t)[0]
            u_0, v_0 = self.position_uv(t)
            radius = self.disc_radius(u_grid, v_grid, u_0, v_0, r_E)[0]
            flux_values = self.flux(radius)
            M_point_u_values = self.magnification_point_u(u_grid)

            numer = np.sum(flux_values * M_point_u_values * delta_u * delta_v)
            denom = np.sum(flux_values * u_grid * delta_u * delta_v)
            mag_e.append(numer / denom)

        for t in time:
            u_0, v_0 = self.position_uv(t)
            mag_p.append(self.magnification_point(u_0))

        return mag_p, mag_e



    def lensing(self, t0, z, P, M, q, I, J, wvl,
                u_max=10, u_grid=500, v_grid=500):

        from astropy import units as u
        from astropy import constants as c

        # Convert to SI units
        self.z  = z
        self.q  = q
        self.M  = M.to('M_sun')
        self.t0 = t0.to('yr')
        self.P  = P.to('yr')
        self.I  = np.pi/2 * u.rad - I
        self.J  = J
        self.wvl = wvl.to('cm')
        
        # Orbital period in binary rest frame [s] 
        self.T = self.period_observed()

        # # Semi-major axis [m]        
        self.a = self.semimajor_axis()
        
        # Schwarzchild radii (primary and secondary)
        self.RS = self.schwarzchild_radius()

        # Maximum Einstein radius (primary and secondary)
        RE1 = np.sqrt(2 * self.a * self.RS[0])
        RE2 = np.sqrt(2 * self.a * self.RS[1])
        
        # Print values
        ut.errorcode('message', 'Model parameters:')
        print(f'Orbital period in rest frame, P    : {self.P.to("yr"):.3f}')
        print(f'Orbital period in obs. frame, T    : {self.T.to("yr"):.3f}')
        print(f'Mass total, M = (M1 + M2)          : {self.M.to("M_sun")/1e6:.3f} x 1e6')
        print(f'Mass ratio, q = (M2 / M1)          : {self.q:.4f}')
        #print(f'Light ratio (L2 /(L1 + L2))   : {self.:.4f}')
        print(f'Inclination of orbits, I           : {I.to("deg"):.2f}')
        print(f'Inclination of mini-disc, J        : {J.to("deg"):.2f}')
        #print(f'Argument of periapse          : {self.w:.2f}')
        print(f'Semi-major axis of binaries, a     : {self.a.to("AU"):.2f}')
        print(f'Schwarchild radius primary, Rs1    : {self.RS[0].to("R_sun"):.2f}')
        print(f'Schwarchild radius secondary, Rs2  : {self.RS[1].to("R_sun"):.2f}')
        print(f'Max Einstein radius primary, Re1   : {RE1.to("AU"):.2f}')
        print(f'Max Einstein radius secondary, Re2 : {RE2.to("AU"):.2f}')

        N = 1000
        time = np.linspace(0, self.T.value, N)
        phase = time / self.T.value
        t_dur = self.time.to('yr').value[-1]
        
        #M_sun = 1.989e33   # [g]
        q = self.q
        M = (self.M/u.M_sun).value
        t0 = self.t0.to('yr').value
        T = self.T.to('yr').value
        P = self.P.value
        I = self.I.value
        J = self.J.value

        #self.a = self.a.to('cm').value
        self.M = (M * u.M_sun).cgs.value
        self.T = self.T.to('yr').value
        self.a = self.a.cgs.value
        self.Mq = self.reduced_mass()
        self.wvl = self.wvl.cgs.value
        self.I = self.I.value
        self.RS_p = self.RS[0].cgs.value
        self.RS_s = self.RS[1].cgs.value
        self.J = self.J.value
        
        self.G = c.G.cgs.value
        self.cc = c.c.cgs.value
        self.h = c.h.cgs.value
        self.k_B = c.k_B.cgs.value
        self.sigma = 5.67 * 10 **-5 # erg cm^-2s^-1K^-4
        
        # RUN MODEL
        M_point, M_finite = self.run_model(time, wvl, z, T, I, J, M, q, u_max, u_grid, v_grid)

    
    def _true_anomaly(self, E, e):
        """Determine the true anomaly, f [rad].        
        """
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E/2))

    
    def _radial_vector(self, E, e, a):
        """Radial vector of motion, r [cm].
        """        
        return a * (1 - e * np.cos(E))

        
    def _rv_semiamplitude(self, P, M1, M, q, a, i, e):
        """The RV semi-amplitude of secondary
        """
        K2 = (2 * np.pi / P) * (M1 / M) * a * np.sin(i) / np.sqrt(1 - e**2)
        K1 = q * K2
        return K1, K2

    
    def _rv_vector(self, v_z, K1, K2, f, e, w):
        """Projection of the velocity vector on to the line of sight.

        Equation from (Murray & Correria, 2010). Minus sign is introduced
        here as the RV is defined to be positive when object is moving away
        from the observed
        """
        vr1 = v_z + K1 * (np.cos(w + f) + e * np.cos(w))
        vr2 = v_z - K2 * (np.cos(w + f) + e * np.cos(w))    
        return vr1, vr2

    def _xyz_orbital_plane(self, a1, q, i, e, w, E, f, omega=np.pi/2):
        """Cartesian 3D position as function of time.
        """
        # Radial vector of peimary
        r1 = self._radial_vector(E, e, a1)
        # Cartesian positions of primary and secondary 
        sini  = np.sin(i)
        cosi  = np.cos(i)
        sino  = np.sin(omega)
        coso  = np.cos(omega)
        sinwf = np.sin(w + f)
        coswf = np.cos(w + f)
        x1 = r1 * (coso * coswf - sino * sinwf * cosi)
        y1 = r1 * (sino * coswf + coso * sinwf * cosi)
        z1 = r1 * (sinwf * sini)
        x2 = -x1 / q
        y2 = -y1 / q
        z2 = -z1 / q
        return x1, y1, z1, x2, y2, z2
        
    #----- Self-lensing: point source -----#
    
    def _radius_schwarzchild(self, M, q):
        """Schwarzchild radius of primary and secondary [cm].
        """
        RS1 = 2 * c.G.cgs * M     / ((1 + q) * c.c.cgs**2)
        RS2 = 2 * c.G.cgs * M * q / ((1 + q) * c.c.cgs**2)
        return RS1, RS2

    
    # def einstein_radius(self, phi1, phi2, I):
    #     """Einstein radius of primary and secondary [cm].
    #     """        
    #     RS1, RS2 = self.RS
    #     const = 2 * self.a.value * np.cos(I)
    #     RE1 = np.sqrt(const * RS1.value * np.sin(phi1))
    #     RE2 = np.sqrt(const * RS2.value * np.sin(phi2))        
    #     return RE1, RE2


    def _angular_separation_xy(self, x1, x2, y1, y2):
        """Angular separation between lens and source in cartesian coordinates, delta.
        """
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
        
    def _angular_einstein_radius(self, z1, z2, M1, M2):
        """Einstein radius of primary and secondary [cm].
        """
        M_l = np.full(z1.shape, M1)
        D_l = -z1
        D_s = -z2
        M_l[self.flip] = M2
        D_l[self.flip] = -z2[self.flip]
        D_s[self.flip] = -z1[self.flip]
        D_rel = D_s - D_l
        return np.sqrt(4 * c.G.cgs.value * M_l * D_rel / c.c.cgs.value**2)
        
    
    def _magnification_point(self, u):
        """Magnification of point source limit.
        """
        return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    #--------------------------------------------------- Lensing: finite source

    def _radius_isco(self, M_s):
        """Radius of innermost stable circular orbit (ISCO) [cm].
        """
        return 6 * c.G.cgs.value * M_s / c.c.cgs.value**2


    def _radius_tidal(self, a, q):
        """Tidal truncation radius of disc [cm].
        """
        power = np.full(self.t.shape, 0.3) # Secondary
        power[self.flip] = -0.3            # Primary
        return 0.27 * a * q**power
    
    
    def _accretion_rate(self, M_s, radiative_efficiency=0.1):
        """Accretion rate of source [g/s].
        """
        M_s = (M_s * u.g).to('M_sun').value
        M_dot = 2.26e-2 * (radiative_efficiency/0.1)**-1 * (M_s/1e6) / ut.year()
        return (M_dot * u.M_sun / u.yr).cgs.value


    def _temperature_disc(self, r, q, M_s):
        """Temperature profile of accretion disc [K].
        """
        const  = 3 * c.G.cgs.value / (8 * np.pi * c.sigma_sb.cgs.value)
        M_acc  = self._accretion_rate(M_s)
        r_isco = self._radius_isco(M_s)
        return (const * M_s * M_acc * (1 - np.sqrt(r_isco/r)) / r**3)**(1/4)  

    
    def _planck_function(self, wvl, T):
        """Planck function for given temperature [cm].
        """
        numerator   = 2 * c.h.cgs.value * c.c.cgs.value**2
        exponent    = c.h.cgs.value * c.c.cgs.value / (wvl * c.k_B.cgs.value * T)
        denominator = wvl**5 * (np.exp(exponent) - 1)
        return numerator / denominator


    def _radius_disc(self, u, v, u0, v0, theta_E, J):
        """Radius of source disc in lens-centered polar coordinates.
        """
        r_star = np.sqrt(u0**2 + u**2 - 2 * u0 * u * np.cos(v - v0)) * theta_E
        u_sinv = u * np.sin(v) - u0 * np.sin(v0)
        u_cosv = u * np.cos(v) - u0 * np.cos(v0)
        theta  = np.arcsin(u_sinv / np.sqrt(u_sinv**2 + u_cosv**2))
        r = r_star * np.sqrt(np.cos(theta)**2 + np.sin(theta)**2 / np.cos(np.pi/2 - J)**2)
        return r

    
    def _flux_disc(self, r, z1, wvl, a, M, q):
        """Flux from accretion disc [erg/s].
        """

        # Set source mass
        if z1 < 0:
            M_s  = M / (1 + q) # Primary
        else:
            M_s  = q * M / (1 + q) # Secondary

        #M_s = M2_s #np.full(self.t.shape, M2_s)
        #M_s[self.flip] = M1_s
        r_isco  = self._radius_isco(M_s)
        r_tidal = self._radius_tidal(a, q)
        T_disc  = self._temperature_disc(r, q, M_s)
        condition1 = (r_isco < r) & (r < r_tidal)
        condition2 = np.pi * self._planck_function(wvl, T_disc)
        return np.where(condition1, condition2, 0)


    def _magnification_finite(self, flux, M_ps_u, u_grid, delta_u, delta_v):
        """Magnification for finite source limit.
        """
        numer = np.sum(flux * M_ps_u * delta_u * delta_v) + self.floor
        denom = np.sum(flux * u_grid * delta_u * delta_v) + self.floor
        return numer / denom        

    
    # def _position_uv(self, phi1, phi2, a, i, e, fe):
    #     """Relative position of primary and secondary in uv plane.
    #     """
        
    #     if 0 < phi1 < np.pi:
    #         phase_u = np.sqrt(np.cos(phi1)**2 + np.sin(i)**2 * np.sin(phi1)**2)
    #         u_0 = a * phase_u / (self.einstein_radius(phi1, phi2, I)[0])
    #     else:
    #         phase_u = np.sqrt(np.cos(phi2)**2 + np.sin(i)**2 * np.sin(phi2)**2)
    #         u_0 = a * phase_u / (self.einstein_radius(phi1, phi2, I)[1])
    #     v_0 = np.arctan(np.sin(I) * np.tan(phi1))
    #     return u_0, v_0 
    

    def lensing(self, time, wvl_cm, z, T, I, e, w, J, M, q, a, u_max, u_grid, v_grid):
        """Function to evaluate gravitation self-lensing model.
        """
        # Grid for accretion disc 
        u_array = np.linspace(0, u_max, u_grid)
        delta_u = u_max / u_grid
        v_array = np.linspace(0, 2 * np.pi, v_grid)
        delta_v = 2 * np.pi / v_grid
        u_grid, v_grid = np.meshgrid(u_array, v_array)

        # Convert to phase space
        phase = time / T
        phi1  = 2 * np.pi * phase
        phi2  = phi1 - np.pi
        
        # Placeholders for magnifications
        N = len(time)
        M1_ps = np.ones(N)
        M2_ps = np.ones(N)
        M_fs  = np.ones(N)

    
        #for i in tqdm(range(N), bar_format=ut.tqdmBar()):
        for i in range(N):

            # Find position in uv plane
            u_0, v_0 = self.position_uv(phi1, phi2, I, e, E[i])

            # Einstein radii
            RE1, RE2  = self.einstein_radius(phi1, phi2, I)

            # Compute magnification
            if 0 < phi1 < np.pi:
                # Point source
                M2_ps[i] = self.magnification_point(u_0)

            else:
                # Point source
                M1_ps[i] = self.magnification_point(u_0)                
                
            # Finite source magnification
            #r    = self.radius(u_grid, v_grid, u_0, v_0, RE1, J)
            #flux = self.flux(wvl_cm, r, q, M, T)
            #M_fs[i] = self.magnification_finite(flux, u_grid, delta_u, delta_v)

        return M1_ps, M2_ps

#==============================================================#
#                         EXOPLANETS                           #
#==============================================================#
    
class Exoplanet(object):
    """Class for modelling exoplanets.
    """

    def __init__(self, seed=False):
        # Random number generator
        self.rng = ut.rng(seed)

        
    def ldc(self, ):
        """Compute the Limb Darkening (LD) coefficients.

        This module uses the software LDTk:        
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

        # Estimate quadratic law coefficients
        # Take care of occations when LDTk fails
        try:
            u, _ = ps.coeffs_qd(do_mc=True)
        except:
            self.ldc = [0.430, 0.170]
            errorcode('warning', 'LD coefficients failed for ' +
                      f'(Teff, logg, Z) = ({self.Teff}, {self.logg}, {self.Z}')
        else:
            self.ldc = u[0]

        return self.ldc
    
    
class LimbDarkening(funcFit.OneDFit):
    """Class for fitting the Limb Darkening Coefficients.
    Integrated into the BATMAN and SPIDERMAN packages.
    """

    def __init__(self):
        funcFit.OneDFit.__init__(self, ['u1', 'u2'])

    def evaluate(self, x):

        model = 'quadratic'

        if model == 'linear':
            coef = ['u1']
        else:
            coef = ['u1', 'u2']

        y = 1. - self['u1']*(1. - x)  # Linear model
        if model == 'quadratic':   y = y - self['u2']*(1. - x)**2
        if model == 'square-root': y = y - self['u2']*(1. - np.sqrt(x))
        if model == 'logarithmic': y = y - self['u2']*x*np.log(x)
        if model == 'exponential': y = y - self['u2']/(1 - np.exp(x))
        if model == 'nonlinear':   y = 1. - self['u1']*(1 - np.sqrt(x)) - self['u2']*(1. - x)

        return y



    def limb_darkening_coefficients(self, bandpass, star_source, plot=False):
 
        """Calculate the Limb Darkening (LD) coefficient.

        To compute our custom limb-darkening transit duration coefficients that meet
        the PLATO transmission response function. We used the angle-dependent ("MU")
        Specific Intensity Spectra (SIS) from PHOENIX (Goettingen 2018), which exactly
        as above is a library of the stellar effective temperature, surface gravity,
        and metallicity. The limb darkening are naturally calclated for the exact same
        stellar parameter as used for the granulation and oscialltions. 

        See links:
        Webpage : https://phoenix.astro.physik.uni-goettingen.de/
        Download: http://phoenix.astro.physik.uni-goettingen.de/data/

        NOTE not used anymore py VarSim.
        """

        #  Convert input parameters
        wvl_tele  = self.wvl_tele.to('Å').value
        tran_tele = self.tra_tele

        # Interpolate (piecewise cubic) into higher resolution grid
        grid_no  = 1000
        wvl_int  = np.linspace(wvl_tele[0], wvl_tele[-1], grid_no)
        passband = make_interp_spline(wvl_tele, tran_tele, k=3)
        tran_int = passband(wvl_int)

        # Limb darkening model options:
        if args.ldm: limbDarkModel = args.lmd
        else: limbDarkModel = 'quadratic'

        # ANGLE DEPENDENT SIS FROM PHOENIX
        # TODO use proper PHOENIX LD model

        # Get low resolution angle dependent SIS spectra:
        # - Each row in data is a spectrum
        # - mu is then the corresponding values for each spectrum
        phoenix = Phoenix()
        wvl_data, mu, data = phoenix.getSpecIntFITS(self.Teff.value, self.logg, self.Z)

        # Convert data from Vacuum-To-Air (VTA)
        data_int     = np.zeros(grid_no*len(mu)).reshape(grid_no, len(mu))
        data_int_VTA = np.zeros(grid_no*len(mu)).reshape(grid_no, len(mu))
        for jj in range(len(mu)):
            data_int[:,jj] = np.interp(wvl_int, wvl_data, data[jj])
            data_int_VTA[:,jj], vind = pyasl.specAirVacConvert(wvl_int, data_int[:,jj],
                                                               direction="vactoair")

        # Colvolve data with transmission to find the VTA intensity
        intensity_VTA = [np.nansum(data_int_VTA[:,i]*tran_int) for i in range(len(mu))]
        intensity_VTA = intensity_VTA/max(intensity_VTA)

        # Interpolate to a finer grid
        mu_int = np.linspace(0, 1, grid_no)
        intensity_VTA_int = np.interp(mu_int, mu, intensity_VTA)

        # Get rid of the lowest mu's to do 2nd order fitting
        # NOTE The stellar limb causes a jump for mu<0.15
        ind = np.where(mu_int >= 0.15)
        mu_trunc = mu_int[ind]
        intensity_VTA_trunc = intensity_VTA_int[ind]

        # Initialize the class that will do the fitting, and prepare for the fit
        model_ldc = LimbDarkening()
        model_ldc.thaw(['u1', 'u2'])
        model_ldc.assignValue({'u1':0.2, 'u2':0.2})
        model_ldc.fit(mu_trunc, intensity_VTA_trunc, xtol=1.e-7, ftol=1.e-7, disp=0)
        ldc = [model_ldc["u1"], model_ldc["u2"]]

        # Plot results
        if args.plot:
            pt.plot_passband_ldc(wvl_int, tran_int, grid_no, mu_trunc,
                                 intensity_VTA_trunc, model_ldc, ldc)

        # Finito!
        return ldc


    


class DopplerBeaming(funcFit.OneDFit):

    """Model Doppler Beaming Effect (also called Boosting).

    This utility models the beaming effect descriped in Shporer (2017).
    The effect is composed by: 1) Doppler shift, 2) Time dialation
    3) Light abberation. The calculated beaming amplitude asssumes that
    v_RV << c.

    Resources
    ---------
    Sphorer (2019)         : https://arxiv.org/abs/1703.00496
    Murray & Correia (2011): https://arxiv.org/abs/1009.1738v2
    Loeb & Gaudi (2003)    : https://arxiv.org/abs/astro-ph/0303212

    Parameters
    ----------
    Keplerian model from PyAstronomy using parameter space from the code
    Batman, hence, see documentation for input parameters.

    Return
    ------
    Beaming model (light curve) and Amplitude in relative flux [ppm].
    """

    def __init__(self):
        funcFit.OneDFit.__init__(self, ['wvl_c', 'Teff',
                                        't0', 'P', 'a',
                                        'e', 'i', 'w',
                                        'Mp', 'Ms'])

    def evaluate(self, x, nu):

        # Convert to SI units

        x     = x.to('s')
        nu    = nu * u.rad
        e     = self['e']
        i     = self['i'].to('rad')
        w     = self['w'].to('rad')
        t0    = self['t0'].to('s')
        P     = self['P'].to('s')
        a     = self['a'].to('m')
        wvl_c = self['wvl_c'].to('m')
        Ms    = self['Ms'].to('kg')
        Mp    = self['Mp'].to('kg')
        Teff  = self['Teff'].to('K')

        # Correction factor (alpha) between true bolmetric flux and finite flux:
        # We use Sphorer+2017 Eq.5 analytical expression obtained approximating
        # a blackbody spectrum. In bolometric light, alpha=1, but otherwise
        # deviating due to finite bandpass measurement.
        xx = c.h * c.c / (wvl_c * c.k_B * Teff)
        alpha = 1/4. * xx*np.exp(xx)/(np.exp(xx) - 1)

        # Amplitude: Sphorer (2019) Eq. 3:
        A = (0.0028 * alpha * P.to('d')**(-1/3) * (Ms.to('M_sun') + Mp.to('M_sun'))**(-2/3) *
             Mp.to('M_sun') * np.sin(i)) * 1e6

        # RV signal as function of nu: Murray & Correia (2011) Eq. 61, 65 and 66
        # NOTE "astar" in the following is the reduced semimajor axies due to the common
        # center-of-mass and "K" is the relative RV semi-amplitude of the star.
        phi = (x.value - t0.value) / P.value * u.rad
        astar = Mp/(Mp + Ms) * a
        K     = 2.*np.pi/P * astar * np.sin(i)/np.sqrt(1. - e**2)
        #RV    = K * (np.cos(w + nu) - e*np.cos(w))
        RV    = K * (np.cos(2*np.pi*phi + w) - e*np.cos(w))

        # Final beaming effect [ppm]
        # Second term in Eq.1 from Sphorer+2017 but normalized
        y = 4 * alpha * RV/c.c * 1e6

        return y.value, A.value





class EllipsoidalDistortion(funcFit.OneDFit):

    """Class for the calculation of the ellipsoidal effect.

    The model presented here is a fusion between the simple analytical description
    presented by Sphorer (2019) and adding a parametization solving the elliptical
    model. Higher order terms from Morris & Nafilan (1993) may be desired to have
    a look at for future improvemenets.

    Resources:
    ----------
    Sphorer (2019)          : https://arxiv.org/abs/1703.00496
    Murray & Correia (2011) : https://arxiv.org/abs/1009.1738v2
    Morris & Nafilan (1993) : http://cdsads.u-strasbg.fr/pdf/1993ApJ...419..344M
    Claret & Bloemen (2011) : TODO this is for TESS but we don't have g fror PLATO
    NOTE Only w=90 produces the expected beaming effect with max and min between
    inferior and superior conjunction.
    """

    def __init__(self):
        funcFit.OneDFit.__init__(self, ['g', 'u',
                                        't0', 'P',
                                        'e', 'i', 'w',
                                        'a', 'Rs',
                                        'Mp', 'Ms'])

    def evaluate(self, x, nu):

        # Convert to SI units
        x  = x.to('s')
        nu = nu * u.rad
        e  = self['e']
        i  = self['i'].to('rad')
        w  = self['w'].to('rad')
        t0 = self['t0'].to('s')
        P  = self['P'].to('s')
        a  = self['a'].to('m')
        Rs = self['Rs'].to('m')
        Ms = self['Ms'].to('kg')
        Mp = self['Mp'].to('kg')

        # Orbital phase
        phi = (x.value - t0.value) / P.value

        # Coefficient accounting for the stellar LD and GD: Sphorer (2019) Eq.8
        alpha = 0.15 * (15. + self["u"]) * (1. + self["g"])/(3. - self["u"])

        # Amplitude of the ellipsoidal variation: Sphorer (2019) Eq.7
        A = alpha * Mp/Ms * (Rs/a)**3 * np.sin(i)**2 * 1e6

        # Add parameterization of elliptical orbits: Murray & Correia (2011) Eq.20
        # This is just 1 for e=0
        a1 = (1 + e*np.cos(nu)) / (1. - e**2)

        # Add parameterization of the ellipsoidal distortion itself
        a2 = - np.cos(4 * np.pi * phi)

        # Final ellipsoidal distortion [ppm]
        y = A * a1 * a2

        return y.value, A.value


    


class PlanetMRforecast():

    """Class to forecast the mass from a planets radius.
    """

    def __init__(self):
        
        # constant
        self.mearth2mjup = 317.828
        self.mearth2msun = 333060.4
        self.rearth2rjup = 11.21
        self.rearth2rsun = 109.2

        # Boundary
        mlower = 3e-4
        mupper = 3e5

        # Number of different populations
        self.n_pop = 4

        # read parameter file
        filepath = 'inputfiles/data_varsim/varsim_exomass_fitting_parameters.h5' 
        hyper_file = Path(os.getenv("PLATO_PROJECT_HOME")) / filepath

        # Fetch PIC catalogue from FTP server
        try:
            h5 = h5py.File(hyper_file, 'r')
        except:
            errorcode('message', 'Inuaguration: Welcome to the PLATO variability simulator!')
            print(f"Downloading mass-radius parameterisation file...")
            ut.downloadFromFTP(hyper_file.name, hyper_file.parents[0], server='plato')

        # Open file
        h5 = h5py.File(hyper_file, 'r')
        self.all_hyper = h5['hyper_posterior'][:]
        h5.close()


    def indicate(self, M, trans, i):

        """Indicate which M belongs to population i given transition parameter.
        """
        
        ts = np.insert(np.insert(trans, self.n_pop-1, np.inf), 0, -np.inf)
        ind = (M>=ts[i]) & (M<ts[i+1])

        return ind


    
    def split_hyper_linear(self, hyper):

        """Split hyper and derive c.
        """
        
        c0    = hyper[0]
        slope = hyper[1:1+self.n_pop] 
        sigma = hyper[1+self.n_pop:1+2*self.n_pop], 
        trans = hyper[1+2*self.n_pop:]
        
        c = np.zeros_like(slope)
        c[0] = c0
        for i in range(1, self.n_pop):
            c[i] = c[i-1] + trans[i-1]*(slope[i-1]-slope[i])

        return c, slope, sigma, trans


    
    def piece_linear(self, hyper, M, prob_R):

        """model: straight line
        """
        
        c, slope, sigma, trans = self.split_hyper_linear(hyper)
        R = np.zeros_like(M)
        for i in range(4):
            ind = self.indicate(M, trans, i)
            mu = c[i] + M[ind]*slope[i]
            R[ind] = ss.norm.ppf(prob_R[ind], mu, sigma[i])

        return R

    

    def ProbRGivenM(self, radii, M, hyper):


        """Probability of R given M: p(radii|M)
        """

        c, slope, sigma, trans = self.split_hyper_linear(hyper)
        prob = np.zeros_like(M)

        for i in range(4):
            ind = self.indicate(M, trans, i)
            mu = c[i] + M[ind]*slope[i]
            sig = sigma[0][i]
            prob[ind] = ss.norm.pdf(radii, mu, sig)

        prob = prob / np.sum(prob)

        return prob


    
    def classification(self, logm, trans):

        """Classify as four worlds.
        """
        
        count = np.zeros(4)
        sample_size = len(logm)

        for iclass in range(4):
                for isample in range(sample_size):
                        ind = self.indicate(logm[isample], trans[isample], iclass)
                        count[iclass] = count[iclass] + ind

        prob = count / np.sum(count) * 100.
        print('Terran %(T).1f %%, Neptunian %(N).1f %%, Jovian %(J).1f %%, Star %(S).1f %%' \
                        % {'T': prob[0], 'N': prob[1], 'J': prob[2], 'S': prob[3]})

        return None


    
    def Mpost2R(self, mass, unit='Earth', classify='No'):

        """Forecast the Radius distribution given the mass distribution.

        Parameters
        ----------
        mass: one dimensional array
                The mass distribution.
        unit: string (optional)
                Unit of the mass. 
                Options are 'Earth' and 'Jupiter'. Default is 'Earth'.
        classify: string (optional)
                If you want the object to be classifed. 
                Options are 'Yes' and 'No'. Default is 'No'.
                Result will be printed, not returned.

        Returns
        -------
        radius: one dimensional array
                Predicted radius distribution in the input unit.
        """

        # mass input
        mass = np.array(mass)
        assert len(mass.shape) == 1, "Input mass must be 1-D."

        # unit input
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            mass = mass * self.mearth2mjup
        else:
            print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

        # mass range
        if np.min(mass) < 3e-4 or np.max(mass) > 3e5:
            print('Mass range out of model expectation. Returning None.')
            return None

        # convert to radius
        sample_size = len(mass)
        logm = np.log10(mass)
        prob = np.random.random(sample_size)
        logr = np.ones_like(logm)

        hyper_ind = np.random.randint(low=0, high=np.shape(self.all_hyper)[0],
                                      size=sample_size)
        hyper = self.all_hyper[hyper_ind,:]

        if classify == 'Yes':
            self.classification(logm, hyper[:,-3:])


        for i in range(sample_size):
            logr[i] = piece_linear(hyper[i], logm[i], prob[i])

        radius_sample = 10.** logr

        # convert to right unit
        if unit == 'Jupiter':
            radius = radius_sample / self.rearth2rjup
        else:
            radius = radius_sample 

        return radius



    def Mstat2R(self, mean, std, unit='Earth', sample_size=1000, classify='No'):	

        """Forecast the mean and standard deviation of radius.

        Forecast the mean and standard deviation of radius given the mean
        and standard deviation of the mass. Assuming normal distribution
        with the mean and standard deviation truncated at the mass range
        limit of the model.

        Parameters
        ----------
        mean: float
                Mean (average) of mass.
        std: float
                Standard deviation of mass.
        unit: string (optional)
                Unit of the mass. Options are 'Earth' and 'Jupiter'.
        sample_size: int (optional)
                Number of mass samples to draw with the mean and std provided.

        Returns
        -------
        mean: float
                Predicted mean of radius in the input unit.
        std: float
                Predicted standard deviation of radius.
        """

        # unit
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            mean = mean * self.mearth2mjup
            std  = std  * self.mearth2mjup
        else:
            print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

        # draw samples
        mass = ss.truncnorm.rvs((mlower-mean)/std, (mupper-mean)/std,
                                loc=mean, scale=std, size=sample_size)
        
        if classify == 'Yes':	
            radius = self.Mpost2R(mass, unit='Earth', classify='Yes')
        else:
            radius = self.Mpost2R(mass, unit='Earth')

        if unit == 'Jupiter':
            radius = radius / self.rearth2rjup

        r_med = np.median(radius)
        onesigma = 34.1
        r_up   = np.percentile(radius, 50.+onesigma, interpolation='nearest')
        r_down = np.percentile(radius, 50.-onesigma, interpolation='nearest')

        return r_med, r_up - r_med, r_med - r_down



    def Rpost2M(self, radius, unit='Earth', grid_size = 1e3, classify = 'No'):

        """Forecast the mass distribution given the radius distribution.

        Parameters
        ----------
        radius: one dimensional array
                The radius distribution.
        unit: string (optional)
                Unit of the mass. Options are 'Earth' and 'Jupiter'.
        grid_size: int (optional)
                Number of grid in the mass axis when sampling mass from radius.
                The more the better results, but slower process.
        classify: string (optional)
                If you want the object to be classifed. 
                Options are 'Yes' and 'No'. Default is 'No'.
                Result will be printed, not returned.

        Returns
        -------
        mass: one dimensional array
                Predicted mass distribution in the input unit.
        """

        # unit
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            radius = radius * self.rearth2rjup
        else:
            print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")


        # radius range
        if np.min(radius) < 1e-1 or np.max(radius) > 1e2:
            print('Radius range out of model expectation. Returning None.')
            return None

        # sample_grid
        if grid_size < 10:
            print('The sample grid is too sparse. Using 10 sample grid instead.')
            grid_size = 10

        ## convert to mass
        sample_size = len(radius)
        logr = np.log10(radius)
        logm = np.ones_like(logr)

        hyper_ind = np.random.randint(low=0, high=np.shape(self.all_hyper)[0],
                                      size=sample_size)
        hyper = self.all_hyper[hyper_ind,:]
        
        logm_grid = np.linspace(-3.522, 5.477, 1000)

        for i in range(sample_size):
            prob = self.ProbRGivenM(logr[i], logm_grid, hyper[i,:])
            logm[i] = np.random.choice(logm_grid, size=1, p = prob)

        mass_sample = 10.** logm

        if classify == 'Yes':
            self.classification(logm, hyper[:,-3:])

        ## convert to right unit
        if unit == 'Jupiter':
            mass = mass_sample / self.mearth2mjup
        else:
            mass = mass_sample

        return mass



    def Rstat2M(self, mean, std, unit='Earth', sample_size=1e3, grid_size=1e3, classify='No'):

        """Forecast the mean and standard deviation of mass given
        the mean and standard deviation of the radius.

        Parameters
        ----------
        mean: float
                Mean (average) of radius.
        std: float
                Standard deviation of radius.
        unit: string (optional)
                Unit of the radius. Options are 'Earth' and 'Jupiter'.
        sample_size: int (optional)
                Number of radius samples to draw with the mean and std provided.
        grid_size: int (optional)
                Number of grid in the mass axis when sampling mass from radius.
                The more the better results, but slower process.
        
        Returns
        -------
        mean: float
                Predicted mean of mass in the input unit.
        std: float
                Predicted standard deviation of mass.
        """
        
        # unit
        if unit == 'Earth':
            pass
        elif unit == 'Jupiter':
            mean = mean * self.rearth2rjup
            std  = std  * self.rearth2rjup
        else:
            print("Input unit must be 'Earth' or 'Jupiter'. Using 'Earth' as default.")

        # draw samples
        radius = ss.truncnorm.rvs((0-mean)/std, np.inf, loc=mean, scale=std, size=sample_size)
        if classify == 'Yes':
            mass = self.Rpost2M(radius, 'Earth', grid_size, classify='Yes')
        else:
            mass = self.Rpost2M(radius, 'Earth', grid_size)

        if mass is None:
            return None

        if unit=='Jupiter':
            mass = mass / self.mearth2mjup

        m_med = np.median(mass)
        onesigma = 34.1
        m_up   = np.percentile(mass, 50.+onesigma, interpolation='nearest')
        m_down = np.percentile(mass, 50.-onesigma, interpolation='nearest')

        return m_med, m_up - m_med, m_med - m_down        

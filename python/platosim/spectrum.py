#!/usr/bin/env python

"""
This class contains functions relevant for downloading SEDs from 
the PHOENIX and ATLAS9 library of synthetic spectra.
"""

# Built-in
import os
import zipfile
import urllib.request
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd
from astropy.io import fits

# PlatoSim functions
from platosim.utilities import errorcode, findNearestIndex


#==============================================================#
#                           SED CLASS                          #
#==============================================================#

class Spectrum(object):

    """This class provides a convenient handling of model spectra.
    
    For a given set of Teff, logg, Z, and alpha, it downloads the
    spectrum from the servers mentioned below.

    PHOENIX references:
    Webpage : https://phoenix.astro.physik.uni-goettingen.de/
    Download: http://phoenix.astro.physik.uni-goettingen.de/data/
    Paper   :
    NOTE Phoenix NextGen gas phase models are only valid for Teff>2700K

    ATLAS9 references:
    Webpage : 
    Download: https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/
    Paper   : 
    """

    def __init__(self, verbose=1):

        """Initialize and prepare data structure
        """
        
        # Create data directories if they do not exist

        path    = Path(__file__).parent.resolve()
        dataDir = os.getenv("PLATO_PROJECT_HOME") + '/inputfiles/data_varsim'
        self.path = path.joinpath(dataDir)
        self.path.mkdir(parents=False, exist_ok=True)
        self.verbose = verbose




    def parameter_space(self, library):

        """Return parameter space.
        """

        if library == 'PhoenixAtmos':
            self.valid_t = np.array([*list(range(2300, 7000, 100)),
                                     *list((range(7000, 12200, 200)))])
            self.valid_g = np.array([*list(np.arange(0, 6, 0.5))])
            self.valid_z = np.array([*list(np.arange(-4, -2, 1)),
                                     *list(np.arange(-2.0, 1.5, 0.5))])
            self.valid_a = np.array([-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

        if library == 'PhoenixHiRes':
            self.valid_t = np.array([*list(range(2300, 7000, 100)),
                                     *list((range(7000, 12200, 200)))])
            self.valid_g = np.array([*list(np.arange(0, 6, 0.5))])
            self.valid_z = np.array([*list(np.arange(-4, -2, 1)),
                                     *list(np.arange(-2.0, 1.5, 0.5))])
            self.valid_a = np.array([-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

        if library == 'PhoenixMedRes':
            self.valid_t = np.array([*list(range(2300, 7000, 100)),
                                     *list((range(7000, 12200, 200)))])
            self.valid_g = np.array([*list(np.arange(-5.0, 1.0, 0.5))])
            self.valid_z = np.array([*list(np.arange(-6, 0.5, 0.5))])
            self.valid_a = np.array([0])

        if library == 'PhoenixSpecInt':
            self.valid_t = np.array([*list(range(2300, 5000, 100)),
                                     *list(range(5100, 7000, 100)),
                                     *list(range(7000, 12200, 200))])
            self.valid_g = np.array([*list(np.arange(0, 6.5, 0.5))])
            self.valid_z = np.array([-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
            self.valid_a = np.array([0])

        if library == 'Atlas9':
            self.valid_t = np.array([*list(range( 3000, 13000,  250)),
                                     *list(range(13000, 50000, 1000))])
            self.valid_g = np.array([*list(np.arange(0.0, 5.0, 0.5))])
            self.valid_z = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5])
            self.valid_a = np.array([0])




            
    def nearest_parameters(self, Teff, logg, Z, alpha, library):

        """Return compliant parameter space.
        """

        # Find indices for closest valid parameter
        self.parameter_space(library)
        dex_t = findNearestIndex(self.valid_t, Teff)
        dex_g = findNearestIndex(self.valid_g, logg)
        dex_z = findNearestIndex(self.valid_z, Z)
        dex_a = findNearestIndex(self.valid_a, alpha)

        # Select valid parameter
        Teff  = self.valid_t[dex_t]
        logg  = self.valid_g[dex_g]
        Z     = self.valid_z[dex_z]
        alpha = self.valid_a[dex_a]

        return Teff, logg, Z, alpha

        


    
    def destructor(self):

        """Destructor.
        """
        
        string = ('Invalid parameter space! Valid values are:' +
                  f'\nTeff: {self.valid_t} ' +
                  f'\nlogg: {self.valid_g} ' +
                  f'\nZ: {self.valid_z}\n')
                    
        errorcode('error', string)
        





    #--------------------------------------------------------------#
    #                        PHOENIX MODELS                        #
    #--------------------------------------------------------------#    

        
    def getPhoenixAtmosFITS(self, Teff, logg, Z, alpha):

        """Download and load PHOENIX atmospheric SED models.
        """

        # Create data folder if it do not exist
        dataDir = self.path / 'PHOENIX_AtmosFITS'
        Path(self.path.joinpath(dataDir)).mkdir(parents=False, exist_ok=True)

        # Make sure code do not crash
        Teff, logg, Z, alpha = self.nearest_parameters(Teff, logg, Z, alpha, 'PhoenixAtmos')

        # Accept only valid parameter space

        if Teff in self.valid_t and logg in self.valid_g \
           and Z in self.valid_z and alpha in self.valid_a:

            # Fetch zip url
            url_zip = (
                "ftp://phoenix.astro.physik.uni-goettingen.de/"
                "AtmosFITS/PHOENIX-ACES-AGSS-COND-2011_AtmosFITS_"
                "Z-{0:{1}2.1f}{2}{3}.zip".format(
                    Z,
                    "+" if Z > 0 else "-",
                    "" if alpha == 0 else ".Alpha=",
                    "" if alpha == 0 else "{:+2.2f}".format(alpha),
                )
            )

            # Select proper file name
            lte_file = (
                "lte{0:05}-{1:2.2f}-{2:{3}2.1f}{4}{5}."
                "PHOENIX-ACES-AGSS-COND-2011.ATMOS.fits".format(
                    Teff,
                    logg,
                    Z,
                    "+" if Z > 0 else "-",
                    "" if alpha == 0 else ".Alpha=",
                    "" if alpha == 0 else "{:+2.2f}".format(alpha),
                )
            )

            # Correct urls
            url_zip  = url_zip.replace("--", "-")
            lte_file = lte_file.replace("--", "-")

            # Download zip file
            phoenix_zip_file = self.path.joinpath(dataDir).joinpath(url_zip.split('/')[-1])
            if not phoenix_zip_file.is_file():
                if self.verbose > 0: print(f"Downloading Phoenix spectrum from {url_zip}")
                with urllib.request.urlopen(url_zip) as response, open(phoenix_zip_file, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)

            # Extract only requested spectrum from zip
            phoenix_file = self.path.joinpath(dataDir).joinpath(lte_file)
            if not phoenix_file.is_file():
                if self.verbose > 0: print(f'Extracting file from zip {lte_file}')
                with zipfile.ZipFile(phoenix_zip_file, 'r') as zip:
                    # Extract all files
                    zip.extractall(self.path.joinpath(dataDir))
                    # Remove zip file
                    phoenix_zip_file.unlink(missing_ok=False)

            # Fetch spectrum
            spectrum = fits.getdata(phoenix_file)

        # Raise an error if parameters is outside limits
        else: self.destructor()

        # Finito!
        return spectrum




    

    def getPhoenixHiResFITS(self, Teff, logg, Z, alpha):

        """Download and load PHOENIX High Resolution SED models.
        """

        # Prepare data structure
        dataDir = self.path / 'PHOENIX_HiResFITS'
        Path(self.path.joinpath(dataDir)).mkdir(parents=False, exist_ok=True)
        phoenix_path = self.path.joinpath(dataDir).joinpath('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

        # Accept only valid parameter space
        if Teff in self.valid_t and logg in self.valid_g \
           and Z in self.valid_z and alpha in self.valid_a:

            # FETCH WAVELENGTH DATA

            if not phoenix_path.is_file():
                if self.verbose > 0: print("Downloading PHOENIX wavelength file")
                url = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
                with urllib.request.urlopen(url) as response, open(phoenix_path, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)
            wvl = fits.getdata(phoenix_path)  # [AA]

            # SECURE PARAMETER SPACE
            
            if Z == 1.0:
                if   Teff >= 5300 and Teff < 6200 and logg < 0.5:
                    logg = 0.5
                elif Teff >= 6200 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 8800 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 8800 and Teff < 9600 and logg < 2.0:
                    logg = 2.0
                elif Teff >= 9600 and Teff < 11600 and logg < 2.5:
                    logg = 2.0
                elif Teff >= 11600 and logg < 3.0:
                    logg = 3.0
            
            elif Z == 0.5:
                if   Teff >= 5800 and Teff < 6200 and logg < 0.5:
                    logg = 0.5
                elif Teff >= 6200 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 9400 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 9400 and logg < 2.0:
                    logg = 2.0
            
            elif Z == 0.0:
                if   Teff >= 5700 and Teff < 6000 and logg < 0.5:
                    logg = 0.5
                elif Teff >= 6000 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 9400 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 9400 and logg < 2.0:
                    logg = 2.0
                    
            elif Z == -0.5:
                if   Teff >= 5800 and Teff < 6100 and logg < 0.5:
                    logg = 0.5
                elif Teff >= 6100 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 9600 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 9600 and logg < 2.0:
                    logg = 2.0                    

            elif Z == -1.0:
                if   Teff == 5900 and logg < 0.5:
                    logg = 0.0
                elif Teff == 5900 and logg >= 0.5 and logg <= 1.0:
                    logg = 1.0
                elif Teff >= 6100 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 9600 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 9600 and logg < 2.0:
                    logg = 2.0

            elif Z in [-1.5, -2.0, -3.0, -4.0]:
                if   Teff == 7800 and Z == -4.0 and logg == 3.0:
                    logg = 3.5
                elif Teff >= 6100 and Teff < 6200 and logg < 0.5:
                    logg = 0.5
                elif Teff >= 6200 and Teff < 8400 and logg < 1.0:
                    logg = 1.0
                elif Teff >= 8400 and Teff < 9600 and logg < 1.5:
                    logg = 1.5
                elif Teff >= 9600 and logg < 2.0:
                    logg = 2.0                    
                    
            # FETCH SPECTRUM
                    
            baseurl = (
                "ftp://phoenix.astro.physik.uni-goettingen.de/"
                "HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"
                "Z-{0:{1}2.1f}{2}{3}/".format(
                    Z,
                    "+" if Z > 0 else "-",
                    "" if alpha == 0 else ".Alpha=",
                    "" if alpha == 0 else "{:+2.2f}".format(alpha),
                )
            )

            url = (
                baseurl + "lte{0:05}-{1:2.2f}-{2:{3}2.1f}{4}{5}."
                "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
                    Teff,
                    logg,
                    Z,
                    "+" if Z > 0 else "-",
                    "" if alpha == 0 else ".Alpha=",
                    "" if alpha == 0 else "{:+2.2f}".format(alpha),
                )
            )

            # Fix url
            url = url.replace("--", "-")
            url = url.replace("-+", "+")

            # Download spectrum
            spectrum_path = self.path.joinpath(dataDir).joinpath(url.split("/")[-1])
            if not spectrum_path.is_file():
                if self.verbose > 0: print(f"Downloading Phoenix spectrum: {url}")
                with urllib.request.urlopen(url) as response, open(spectrum_path, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)

            # Fetch spectrum from file [ergs/s/cm^2/cm] -> convert with *0.1 to [uW/m^2/um]
            flux = fits.getdata(spectrum_path) / 1e8  # [ergs/s/cm2/AA]  

        # Raise an error if parameters is outside limits
        else: self.destructor()

        # Finito!
        return wvl, flux



    

    def getPhoenixSpecIntFITS(self, Teff, logg, Z):

        """Download and load PHOENIX spectral intensities.
        """

        # Create data folder if it do not exist
        dataDir = self.path / 'PHOENIX_SpecIntFITS'
        Path(self.path.joinpath(dataDir)).mkdir(parents=False, exist_ok=True)

        # Make sure that all paramters are valid
        Teff, logg, Z, alpha = self.nearest_parameters(Teff, logg, Z, 0, 'PhoenixSpecInt')

        if Teff in self.valid_t and logg in self.valid_g and Z in self.valid_z:

            # DOWNLOAD SPECTRUM

            baseurl = (
                "ftp://phoenix.astro.physik.uni-goettingen.de/"
                "SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/"
                "Z{0}{1:2.1f}/".format(
                    "+" if Z > 0 else "-",
                    Z,
                )
            )

            url = (
                baseurl + "lte{0:05}{1}{2:2.2f}{3}{4:2.1f}."
                "PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits".format(
                    Teff,
                    "+" if logg == 0 else "-",
                    logg,
                    "+" if Z > 0 else "-",
                    Z
                )
            )

            # Fix url
            url = url.replace("--", "-")
            url = url.replace("-+", "+")
            
            # Download spectrum
            phoenix_file = self.path.joinpath(dataDir).joinpath(url.split("/")[-1])

            if not phoenix_file.is_file():

                if self.verbose > 0: print(f"Downloading Phoenix spectrum: {url}")
                with urllib.request.urlopen(url) as response, open(phoenix_file, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)

            # FETCH WAVELENGTH AND INTENSITIES

            data, hdr = fits.getdata(phoenix_file, header=True)
            mu  = fits.getdata(phoenix_file, 'MU')
            wvl = np.arange(hdr["CRVAL1"], hdr["CRVAL1"]+hdr["NAXIS1"]*hdr["CDELT1"], hdr["CDELT1"])

        # Raise an error if parameters is outside limits

        else: self.destructor()

        # Finito!

        return wvl, mu, data




    
    #--------------------------------------------------------------#
    #                         ATLAS9 MODELS                        #
    #--------------------------------------------------------------#    

    
    def getAtlasFITS(self, Teff, logg, Z, alpha):

        """Download and load an ATLAS9 LTE SED model.

        These LTE models are from Castelli & Kurucz (2003). The models
        do not include convective overshooting, but have improved 
        opacities and abundances upon previously used by Kurucz (1990).
        """

        # Prepare data structure        
        dataDir = self.path / 'ATLAS9_FITS'
        Path(self.path.joinpath(dataDir)).mkdir(parents=False, exist_ok=True)

        # Accept only valid parameter space or raise an error
        
        if (Teff in self.valid_t and logg in self.valid_g 
            and Z in self.valid_z and alpha in self.valid_a):

            # Sort filename string
            if Z >= 0:
                sign = 'p'
            else:
                sign = 'm'
                Z    = np.abs(Z)
            string_z = str(Z)[0]    + str(Z)[2]
            string_g = str(logg)[0] + str(logg)[2]

            # Full download link
            server = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models"
            url_name = f'ck{sign}{string_z}'
            url = f"{server}/{url_name}/{url_name}_{Teff}.fits"

            # Download spectrum
            spectrum_path = self.path.joinpath(dataDir).joinpath(url.split("/")[-1])
            if not spectrum_path.is_file():
                if self.verbose > 0: print(f"Downloading ATLAS9 spectrum: \n{url}")
                with urllib.request.urlopen(url) as response, open(spectrum_path, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)

            # Fetch spectrum from file
            hdul = fits.open(spectrum_path)
            df   = pd.DataFrame(hdul[1].data)
            wvl  = df.WAVELENGTH.to_numpy()

            # Select column of closest logg
            flux = df[f'g{string_g}'].to_numpy()
            dex  = df.columns.tolist().index(f'g{string_g}')

            # Find nearest logg value
            if flux.sum() == 0:
                for i in range(dex, len(df.iloc[0,:])):
                    flux += df.iloc[:,i].to_numpy()
                    if flux.sum() > 0:
                        break

        else:
            self.destructor()
        
        # Finito!
        return wvl, flux

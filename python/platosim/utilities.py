#!/usr/bin/env python3
"""
Python modules that contain some general utilities that are commonly
used by the PlatoSim and PLATOnium.
"""

# Built-in
import os
import sys
import math
import ftplib
import shutil
import inspect
import fnmatch
from pathlib import Path
from zipfile import ZipFile

# PlatoSim standard
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
from prettytable import PrettyTable
import scipy
from scipy.stats import gaussian_kde
from scipy import constants as c

# Backward compatability
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    # Versions of scipy < 1.6 use a different name
    from scipy.integrate import cumtrapz as cumulative_trapezoid

#--------------------------------------------------------------#
#                        UNIT FUNCTIONS                        #
#--------------------------------------------------------------#

def day():
    """Return 1 day in seconds.
    """
    return 86400


def year():
    """Return 1 year in seconds.
    """
    return 31556926


def quarter():
    """Return 1 mission quarter in days.
    """
    return year() / (4 * day())


def rng(seed=None):
    """Choose seed for randomness
    """
    if seed is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(seed=seed)        

    
def getHomeDir(path=''):
    """Fetch PlatoSim home and working directory.
    """    
    return Path(os.getenv('PLATO_PROJECT_HOME')) / path


def getWorkDir(path=''):
    """Fetch PlatoSim home and working directory.
    """    
    return Path(os.getenv('PLATO_WORKDIR')) / path


#--------------------------------------------------------------#
#                        BASH FUNCTIONS                        #
#--------------------------------------------------------------#

def errorcode(API, message):
    """Function to colour code error messages within a code.

    Parameters
    ----------
    API : str
       Which API to use: [software, module, message, warning, error]
    message : str
       Message to add to API

    Return
    ------
    Error message written to bash
    """
    from colorama import Fore, Style
    
    if API == 'software':
        print(Style.BRIGHT + Fore.BLUE + message + Style.RESET_ALL)
    if API == 'module':
        print(Style.BRIGHT + Fore.GREEN + message + Style.RESET_ALL)
    if API == 'message':
        print(Style.BRIGHT + message + Style.RESET_ALL)
    if API == 'warning':
        print(Style.BRIGHT + Fore.YELLOW + '[Warning]: ' + message + Style.RESET_ALL)
    if API == 'error':
        print(Style.BRIGHT + Fore.RED + '[Error]: ' + message + Style.RESET_ALL)
        sys.exit()


def fileMatch(fileList, stringList):
    """
    fileMatch(fileList, stringList)
    
    returns the list of files from fileList in which all strings from stringList 
    can be found, in any order Case sensitive.
    """

    while (len(stringList)>1):
        fileList = fileMatch(fileList,[stringList.pop()])
    return [file for file in fileList if fnmatch.fnmatch(file,'*'+stringList[0]+"*")]





def fileSelect(stringList, location="./", listOrder=0):

    """fileSelect(stringList, location="./", listOrder=0)

    Returns a list of all files in 'location' with name matching every string in the list
    If listOrder is True, the ordering is forced to be identical to the one in stringList
    """
    
    allfiles = os.listdir(location)
    if listOrder:
        pattern = "*"
        i = 0
        while i < len(stringList): 
            pattern += stringList[i] + "*"
            i += 1
        return [file for file in allfiles if fnmatch.fnmatch(file,pattern)]
    else:
        return fileMatch(allfiles, stringList)




    
def getFunctions(script):

    """Fetch names of class functions.
    """
    
    names = inspect.getmembers(script, inspect.isfunction)
    funcs = [item[0] for item in names]
    t = PrettyTable()
    t.add_column(f"{script.__name__} functions", funcs)

    return t
    




def tqdmBar():

    """Add-on function to be used in a for statement with tqdm library.

    Example:
    >>> from tqdm import tqdm
    >>> for i in tqdm(range(<number of loops>), bar_format=ut.tqdmBar()):
            <loop over something>
    """

    return "{l_bar}{bar:50}{r_bar}{bar:-50b}"





def compilation(i, i_max, text=''):

    """Custum function to print out a compilation-time-bar in the terminal.

    Parameters
    ----------
    i : int
        Index of the current running job-loop
    i_max : int
        Index of the last running job-loop
    text : str
        Optional text written next to compilation-bar

    Return
    ------
        No parameters only a nice compilation bar to bash

    Example
    -------
    for i in range(10):
        compilation(i, len(ra), 'Computing pixel positions')
    print; print('')
    """

    # Running percentage calculation

    percent = (i + 1) / (i_max * 1.0) * 100

    # We here divide by 2 as the length of the bar is only 50 characters:

    bar = "[" + "-" * int(percent / 2) + '>' + " " * (50 - int(percent / 2)) + "] {}% {}" \
        .format(int(percent), text)

    # Print and clean-up

    sys.stdout.write(u"\u001b[1000D" + bar)
    sys.stdout.flush()


    


def downloadFromFTP(filename, outputDir=False, server='plato'):

    """Function to download file from KUL FTP.
    
    Parameters
    ----------
    filename : str
        Filename of file on server
    outputDir : str
        Output directory to save file ()
    https://stackoverflow.com/questions/67300881/how-do-i-keep-a-ftp-connection-alive

    Return
    ------
    File with <filename> is saved in <outputDir>
    """

    # Assume that no suffix means a folder of data
    # If true then download folder and it entire content
    # If flase simply download the requested file
    
    ftp_filename = Path(filename)

    if ftp_filename.suffix:
        ftp_subpath = ftp_filename.parents[0]
        permission  = False
    else:
        ftp_subpath = ftp_filename
        permission  = True
        
    # If file on FTP is within a folder, create folder locally

    if not outputDir:
        outputDir = os.getenv('PLATO_PROJECT_HOME') + '/inputfiles'
        
    outputDir = outputDir / ftp_subpath
    outputDir.mkdir(parents=True, exist_ok=True)
        
    # Login to server
    # For plato: Download a single file
    # For platodata: Download all files in a folder
    #ftp = 'ftp://plato:miSotalP@ftp.ster.kuleuven.be'
    #ftp = 'ftp://platodata:i9Pidw1bXIFShGYb0jI8@ftp.ster.kuleuven.be/PLATOSIM'
    
    ftp = ftplib.FTP('ftp.ster.kuleuven.be')
    
    if server == 'plato':
        # Single file download
        ftp.login(user=server, passwd='miSotalP')
        ftp.cwd(f'{ftp_subpath}')
        files = [filename]
    else:
        errorcode('error', f'Server name {server} is not valid!')
            
    # Fetch all the files
        
    for filename in files:
        
        # Only try to save file if is do not exist

        local_file = Path(outputDir) / filename

        if not local_file.is_file():

            # Download file
            ftp_file   = open(local_file, 'wb')
            ftp.retrbinary(f'RETR {filename}', ftp_file.write)
            ftp_file.close()

            # Check if zip file
            if ftp_filename.suffix == '.zip':
                # Decompress files
                with ZipFile(local_file, 'r') as f:
                    f.extractall(path=outputDir)
                # Remove zip file
                local_file.unlink()
                
            # Give read and write permission
            if permission:
                local_file.chmod(755)

        # Close connection
    
        ftp.quit()

        # Login each time for download due to timeout
        
        if server == 'platodata' and not filename == files[0]:
            ftp = ftplib.FTP('ftp.ster.kuleuven.be')
            ftp.login(user=server, passwd='i9Pidw1bXIFShGYb0jI8')
            ftp.cwd(f'PLATOSIM/{ftp_subpath}')




def get_passband(passband='plato', response='absolute', interpolate=False, k=3, n=1000):

    """Fetch normalised passband data.

    Passbands available: [plato, cheops, tess, kepler]
    """

    # Import passband
    path = Path(os.getenv("PLATO_PROJECT_HOME")) / 'inputfiles/data_varsim'
    df  = pd.read_csv(f"{path}/passband_{passband}.txt", comment='#')
    wvl = df.wavelength  # [nm]

    # Fetch response function
    if response == 'absolute':
        tra = df.absolute
    elif response == 'relative':
        tra = df.relative
    else:
        errorcode('error', 'Not valid response! Use either [absolute, relative]')
        
    # Interpolate data if requested
    if interpolate:
        passband = scipy.interpolate.make_interp_spline(wvl, tra, k=3)
        wvl = np.linspace(wvl.iloc[0], wvl.iloc[-1], n)
        tra = passband(wvl)

    return wvl, tra

    


    
#--------------------------------------------------------------#
#                      PANDAS OPERATIONS                       #
#--------------------------------------------------------------#


def pdAddColumn(df, newCol, name):

    """Add a column to an exisiting pandas data frame as first entry.
    """

    df[name] = newCol
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]

    return df[cols]





def pdMoveRowToFirst(df, target_row, reset_index=True):

    """ Move target row to first element of list.

    NOTE: Only works for df with a reset index.
    """

    dex = [target_row] + [i for i in range(df.shape[0]) if i != target_row]

    if reset_index:
        return df.iloc[dex].reset_index(drop=True)
    else:
        return df.iloc[dex]

    



def pdMergeRows(df0, df1, identical=True):

    """Merge two data frames and keep (non)identical rows.
    """
    
    dex = df1.set_index(list(df1.columns)).index
    if identical:
        return df0.loc[df0.set_index(list(df0.columns)).index.isin(dex)]
    else:
        return df0.loc[~df0.set_index(list(df0.columns)).index.isin(dex)]





def votable2pandas(votable):

    """Function to convert a votable to a pandas data frame.

    Resource: https://gist.github.com/icshih/52ca49eb218a2d5b660ee4a653301b2b
    """

    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()


def pdWeightedAverage(df, value, weight):

    """Function to calculate a weighted average
    """    

    val = df[value]
    wt  = df[weight]
    return (val * wt).sum() / wt.sum()


#--------------------------------------------------------------#
#                       NUMPY OPERATIONS                       #
#--------------------------------------------------------------#

    
def findNearestIndex(array, value):

    """Find the nearest value within an numpy array.
    """

    return (np.abs(np.asarray(array) - value)).argmin()





def medianAbsoluteDeviation(array):

    """Calculate the Median Abosolute Deviation (MAD) of an array.
    """
    
    return np.sum(np.abs(array - np.median(array))) / len(np.ravel(array))





def rootMeanSquare(array):

    """Calculate the Root Mean Square (RMS) of an array.
    """
    
    return np.sqrt(np.mean(array ** 2))





def convolve(data0, filtertype, n): 

    """Function to apply filter convolution.

    The utility takes the flatten data, a string with the desired filter, 
    and n number of points is should smooth the data with. Compared to the
    bottleneck package this function do not leave an offset.
    """

    # Constants:
    data  = data0.copy()           # Avoid overwritting data:
    data_new = np.zeros(len(data)) # To pick up new data
    nzero = np.zeros(2*n+1)        # optimization constant

    # Available filters:
    if filtertype=='mean':   moving_filter = np.mean
    if filtertype=='median': moving_filter = np.median
    if filtertype=='sum':    moving_filter = np.sum
    if filtertype=='std':    moving_filter = np.std

    # Interval: d[n, 1+n, ... , N-1, N-n]
    for i in range(len(data)-2*n):   
        data_new[n+i] = moving_filter(data[range((n+i)-n, (n+i)+n+1)])
    for i in range(n):
        # Interval: d[-n, -(n-1), ... , n-1, n] - Low end of data
        low = nzero
        low[range(n-i)] = data[0]*np.ones(n-i)
        low[-(n+1+i):]  = data[range(0, n+1+i)]
        data_new[i]     = moving_filter(low)
        # Interval: d[N-n, N-(n-1), ... , N+(n-1), N+n] - High end of data
        high = nzero
        high[range(n+1+i)] = data[range(len(data)-(n+i+1), len(data))]
        high[-(n-i):]      = data[-1]*np.ones(n-i)
        data_new[len(data)-1-i] = moving_filter(high)

    return data_new





def normalize(signal, factor=1e6, length=-1):

    """Normalize a signal with the option to factorize it.

    Parameters
    ----------
    signal : narray
        Signal array that needs to be turned into relative signal.
    factor : int, float
        Factor to scale signal. Default is in ppm.
    length : int, float
        Option to normalize signal using only a smaller initial signal segment.
        Default is to use the entire signal sequence.

    Return
    ------
    relative_signal : narray
        Normalized relative signal is returned. Default unit in [ppm].
    """

    return (signal / np.nanmedian(signal[:int(length)]) - 1) * factor






def evalLinReg(x, y, x0):

    """Evaluate a simple linear regresion in point x0.
    """
    
    coeff = np.polyfit(x, y, 1)

    return coeff[0] * x0 + coeff[1]





def sortAfterDensity(x, y): 

    """Enable the usage of a slider to show multiple images.

    Parameters
    ----------
    x, y : ndarray
        (x,y) variables to generate density map from.
    
    Return
    ------
    x', y', z' : ndarray
        Sorted arrays after density map, z.

    Example
    -------
    >> plt.scatter(x, y, c=z, cmap='jet')
    """

    # Calculate the point density
    
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density with densest points last

    dex = z.argsort()

    return x[dex], y[dex], z[dex], dex





def imageClip(inputArray, norm="percentile", sigma=2):

    """Performs custom scaling of the input numpy array.

    Parameters
    ----------
    inputArray : ndarray
        Input image array to normalize.
    norm : str
        Normalization method. 
        Options: ['linear', 'log', 'sqrt', 'asinh'] 
    sigma : float
        Scaling factor corresponding to the std of the image.
    scale_min : float
        Minimum data value.
    scale_max : float
        Maximum data value.
    
    Return
    ------
    image : ndarray
        Normalized image array.

    NOTE: This is a help function for simfile.showImage()
    """
    
    # Input image array

    image = np.array(inputArray, copy=True)

    # Methods that return array values

    if norm == "percentile":
        clipPercentile = sigma
        vmin = np.percentile(image, clipPercentile).astype(int)
        vmax = np.percentile(image, 100-clipPercentile).astype(int)
        norm = Normalize(vmin, vmax)
        
    elif norm == "auto":
        image = imageNorm(image, "linear", sigma)
        vmin  = image.min()
        vmax  = image.max()
        norm  = None

    elif norm == "minmax":
        vmin = image.min()
        vmax = image.max()
        norm = Normalize(vmin, vmax)

    elif norm == 'log':
        vmin = image.min()
        vmax = image.max()
        norm = LogNorm(vmin, vmax)
            
    else:
        print('ERROR: imageClip(): Not a valid scaling!')

    # That's it!

    return image, norm, vmin, vmax





def imageNorm(inputArray, norm="linear", sigma=2, scale_min=None, scale_max=None):

    """Performs custom scaling of the input numpy array.

    Parameters
    ----------
    inputArray : ndarray
        Input image array to normalize.
    norm : str
        Normalization method. 
        Options: ['linear', 'log', 'sqrt', 'asinh'] 
    sigma : float
        Scaling factor corresponding to the std of the image.
    scale_min : float
        Minimum data value.
    scale_max : float
        Maximum data value.
    
    Return
    ------
    image : ndarray
        Normalized image array.
    """
    
    # Input image array

    image = np.array(inputArray, copy=True)

    # Extract image information

    image_min  = image.min()
    image_max  = image.max()
    image_mean = image.mean()
    image_std  = image.std()

    # Default scaling is 2 sigma

    if scale_min is None:
        scale_min = image_mean - sigma * image_std
    if scale_max is None:
        scale_max = image_mean + sigma * image_std

    # Clip data

    image = image.clip(min=scale_min, max=scale_max)

    # Function below return normalized image arrat -> [0,1]

    if norm == "linear":
        image = (image - scale_min) / (scale_max - scale_min)
        indices = np.where(image < 0)
        image[indices] = 0.0
        indices = np.where(image > 1)
        image[indices] = 1.0

    elif norm == "log":
        factor = np.log10(scale_max - scale_min)
        indices0 = np.where(image < scale_min)
        indices1 = np.where((image >= scale_min) & (image <= scale_max))
        indices2 = np.where(image > scale_max)
        image[indices0] = 0.0
        image[indices2] = 1.0
        image[indices1] = np.log10(image[indices1]) / factor

    elif norm == "sqrt":
        image = image - scale_min
        indices = np.where(image < 0)
        image[indices] = 0.0
        image = np.sqrt(image)
        image = image / math.sqrt(scale_max - scale_min)
        image = np.sqrt(image)
        image = image / np.sqrt(scale_max - scale_min)

    elif norm == "asinh":
        non_linear = 2.0
        factor = np.arcsinh((scale_max - scale_min) / non_linear)
        indices0 = np.where(image < scale_min)
        indices1 = np.where((image >= scale_min) & (image <= scale_max))
        indices2 = np.where(image > scale_max)
        image[indices0] = 0.0
        image[indices2] = 1.0
        image[indices1] = np.arcsinh((image[indices1] - scale_min) / non_linear) / factor
            
    else:
        print("ERROR: Not valid scaling for 'imgScale'")

    # That's it!

    return image





#--------------------------------------------------------------#
#                       GENERAL ASTRONOMY                      #
#--------------------------------------------------------------#


def massLuminosityRelation(R, Teff):

    """Calculate mass using M-L relation.

    Using the Teff in the mass-luminosity relation, one can find
    the stellar mass for a main sequence dwarf star. Method valid
    for (0.43 < M/Msun < 2)

    Notes
    -----
    Reference from:
    https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
    """
    return R**(1/2) * Teff/5777.



def radialDistance(alpha1, delta1, alpha2, delta2):

    """Radial distance between two equatorial coordinates.

    The shortest angular distance between two points on the 
    celestial sphere is measured along a great circle that passes
    through both of them.

    Parameters
    ----------
    (alpha1, delta1) : ndarray, pdframe
        Equatorial coordinates of point 1 [deg]
    (alpha2, delta2) : ndarray, pdframe
        Equatorial coordinates of point 2 [deg]
    
    Return
    ------
    Radial distance between coordinates [deg]
    """
    
    alpha1 = np.deg2rad(alpha1)
    alpha2 = np.deg2rad(alpha2)
    delta1 = np.deg2rad(delta1)
    delta2 = np.deg2rad(delta2)
    
    cosR = (np.sin(delta1) * np.sin(delta2) +
            np.cos(delta1) * np.cos(delta2) * np.cos(alpha1-alpha2))
    
    return np.rad2deg(np.arccos(cosR))



def getContaminants(dt, dc, column='PIC', radius=45):
    """Match contaminating sources from dc to targets of dt.
    """
    from tqdm import tqdm
    
    # Query radial distance [arcsec]
    x = radius/3600.

    # Loop over each target
    
    for i in tqdm(range(dt.shape[0]), bar_format=tqdmBar()):
        # Select target star
        dt_i = dt.iloc[i]

        # Fetch smaller region around target
        dc_i = dc[(dc.ra  > dt_i.ra  - x) & (dc.ra  < dt_i.ra  + x) &
                  (dc.dec > dt_i.dec - x) & (dc.dec < dt_i.dec + x)]
        dc_i = dc_i.reset_index(drop=True)
        
        # Remove target star if present
        dc_i = dc_i.drop(dc_i[dc_i[column] == dt_i[column]].index)

        # Find radial distance [arcsec] 
        dc_i['dis'] = radialDistance(dt_i.ra, dt_i.dec, dc_i.ra, dc_i.dec).to_numpy() * 3600.
        dc_i = dc_i.sort_values(by=['dis'])
        dc_i = dc_i.reset_index(drop=True)

        # Set contaminant ID to target ID
        dc_i[column] = dt_i[column]

        # Save to a new df
        if i == 0:
            df = dc_i
        else:
            df = pd.concat([df, dc_i])

    # Save to feather files
    return df.reset_index(drop=True)



def cart2pol(x, y):
    """Transformation from cartesian to polar coordinates.

    Parameters
    ----------
    x,y : float, ndarray
        Cartesian coordinates.

    Return
    ------
    phi,rho : float, ndarray
        Polar coordinates.
    """    
    phi = np.rad2deg(np.arctan2(y, x)) - 180
    rho = np.sqrt(x**2 + y**2)
    return (phi, rho)



def mm2pixels(distanceMm, focalLength, plateScale):

    """Conversion from millimeters to pixels.

    Parameters
    ----------
    distanceMm : float
    Distance [mm]
    focalLength : float
    Focal lenght of telescope [mm]
    plateScale : float
    The plate scale of the telescope [mm/pixel]
    
    Returns
    -------
    distancePixels : float
    Distance [pixels]
    """
    
    distancePixels = (np.degrees( np.arctan(distanceMm / focalLength)) /
                      plateScale * c.degree / c.arcsec)
    return distancePixels





def cpd2muhz(freq):

    """Convert cycles/day to muHz.

    Parameters
    ----------
    freq : float, ndarray
        Frequency [cpd]

    Return
    ------
    freq : float, ndarray
        Frequency [muHz]
    """    
    return 1e6/86400 * freq




def muhz2cpd(freq):

    """Convert muHz to cycles/day.

    Parameters
    ----------
    freq : float, ndarray
        Frequency [muHz]

    Return
    ------
    freq : float, ndarray
        Frequency [cpd]
    """    
    return 86400/1e6 * freq




def ppt2mmag(flux):

    """Convert relative flux [ppt] to relative magnitude [mmag].

    Parameters
    ----------
    flux : float
        Input flux [ppt]

    Return
    ------
    mag : ndarray
        Relative magnitude [mmag]
    """    
    return 2.5 * np.log10(flux/1e3 + 1) * 1e3





def mmag2ppt(dmag):

    """Convert relative magnitude [mmag] to relative flux [ppt].

    Parameters
    ----------
    dmag : float
        Input magnitude [mmag]

    Return
    ------
    flux : ndarray
        Relative flux [ppt]
    """
    return (10**(dmag / 2.5 / 1e3) - 1) * 1e3





def fromMagToFlux(mag):

    """Convert relative magnitude to relative flux.

    Parameters
    ----------
    mag : float
        Input magnitude

    Return
    ------
    flux : ndarray
        Relative flux
    """

    return 10**(-0.4*mag)





def fromMagToRelativeFlux(mag, norm=1e6):

    """Convert magnitude to relative flux

    Parameters
    ----------
    mag : float
        Input magnitude
    norm : float
        Normalisation contant for relative flux

    Return
    ------
    flux : ndarray
        Relative flux scaled after the normalisation constant.
    """
    
    flux = fromMagToFlux(mag)

    return (flux / np.nanmedian(flux) - 1) * norm





def normFlux(flux, norm=1e3):

    """Convert magnitude to relative flux

    Parameters
    ----------
    flux : float
        Relative flux around unity
    norm : float
        Normalisation contant for relative flux

    Return
    ------
    flux : ndarray
        Relative flux scaled after the normalisation constant.
    """

    return (flux / np.nanmedian(flux) - 1) * norm


#--------------------------------------------------------------#
#                        PLATO SPECIFIC                        #
#--------------------------------------------------------------#


def stellarFlux(Vmag, exposureTime, fluxm0=1.00238e8,
                throughputBandwidth=400, transmissionEfficiency=0.76,
                lightCollectingArea=0.01131, quantumEfficiency=0.87):

    """Compute the stellar flux given the instrumental characteristics.

    Parameters
    ----------
    Vmag : float
        Johnson-Cousin V magnitude
    exposureTime : float
        Exposure time (without the readout) [s]
    fluxm0 : float
        Photon flux of a V=0 G2V star [phot/s/m^2/nm]
    throughputBandwidth : float 
        Throughput FWHM value in the passband [nm]
    transmissionEfficiency : 
        Transmission efficiency in the passband [0, 1]
    lightCollectingArea : float
        The aperture of the cameras [m^2]
    quantumEfficiency : float
        Quantum efficiency of the detector [0, 1]

    Return
    ------
    flux : float
        Instrumental stellar flux [e-/exposure]
    """

    photonFlux = (fluxm0 * throughputBandwidth * transmissionEfficiency *
                  lightCollectingArea * pow(10.0, -0.4 * Vmag) * exposureTime)

    return photonFlux * quantumEfficiency





def passbandConversionV2P(mag, Teff, inverse=False, method='fialho'):

    """Coversion between Johnson-Cousin V magnitude and the PLATO passband P.
    
    Parameters
    ----------
    V : float, ndarray
        Johnson-Cousin magnitude of star(s).
    Teff : float narray
        Effective temperature of star(s).
    inverse : bool
        Convert from P to V (assuming that mag = V).
    method : str
        Which conversion method to use ['fialho', 'marchiori'].

    Return
    ------
    mag : float, ndarray
        P mag (if inverse = False) or V mag (if inverse = True).

    Notes
    -----
    - 
    - Marchiori et al. (2019) conversion (Eq. 5 and 6) is extracted 
      using synthetic stellar spectra from the POLLOX database. The
      conversion is valid for Teff = 4000-15000 K (not for M-dwarfs).

    """

    # Bolometric scaling relation

    if method == 'fialho':
        # Fialho et al. (2024)
        c   = [-2.366e-12, 8.126e-8, -9.279e-4, 3.499]
    elif method == 'marchiori':
        # Machiori et al. (2019)
        c = [-1.184e-12, 4.526e-8, -5.805e-4, 2.449]

    bol = c[0]*Teff**3 + c[1]*Teff**2 + c[2]*Teff + c[3]

    # From V to P (or P to V if inverse it True)

    if inverse:
        return mag + bol
    else:
        return mag - bol





def passbandConversionG2P(mag, BP_RP, inverse=False, camera='normal', stage='dwarf'):

    """Conversion from Gaia G_0 magnitude to the PLATO passband.
    
    The calibration relation is derived in PLATO-UPD-SCI-TN-0019, Sect. 6.
    NOTE only valid for (4000K < Teff < 15,000K), hence, not for M-dwarfs.

    Parameters
    ----------
    mag : float, narray
        Gaia mean dereddened G magnitude of star(s).
    BP_RP : float narray
        Gaia dereddened color of star(s).

    Return
    ------
    P : float, narray
        The PLATO passband magnitude of star(s).
    """

    # Define coefficient to transform from G to P
    
    if camera == 'normal':
        if stage == 'dwarf': 
            coeff = [-0.3613390, 0.0632494, 0.0301607, -0.0163962, 0.0027984, -0.0001679]
        elif stage == 'giant':
            coeff = [-0.3586933, 0.0598219, 0.0244786, -0.0119261, 0.0017487, -0.0000870]
    elif camera == 'fast_blue':
        coeff = [-0.1386193, 0.1103836, 0.0582385, -0.0144120, 0.0006554, 0.0000251]
    elif camera == 'fast_red':
        coeff = [-0.6795686, 0.0539941, 0.0331913, -0.0123407, 0.0019006, -0.0001174]

    color = np.sum([coeff[i-1] * BP_RP**i for i in range(1,7)], axis=0)

    # From G to P (or P to G if inverse=True)
    
    if inverse:
        return mag - color 
    else:
        return mag + color
 
    


    
def getPointingField(name, unit='deg', system='icrs'):

    """Function to fetch pointing field coordinates.

    Small function that takes a string of numbers (here of magnitudes)
    and split it up into readable float values used as real number
    ranges. If a single number is given, a selection of 1 mag around
    the imput int/float is returned as a magnitude range.
    
    Used in: PLATOnium/simulator-pic.py

    Parameters
    ----------
    name : str
        Name of the requested pointing field.

    Return
    ------
    Sky coordinates (alpha, delta, kappa) [deg]
    """

    PF = {'NPF':   [265.08002279,  39.5836954,  -10.0000],  # PIC 1.1.0
          'SPF':   [ 86.79870508, -46.39594703,  10.0000],  # PIC 1.1.0
          'LOPN1': [277.18023,     52.85952,    -13.9947],  # PIC 2.0.0
          'LOPS2': [ 95.31043,    -47.88693,     13.9947],  # PIC >= 2.0.0
          'KUL20': [ 86.79870508, -46.39594703,  0.0]}      # TN of KUL20

    # Check data field exists
    
    try:
        p = PF[name]
    except KeyError:
        errorcode('error', 'Not valid Plato field! Use {LOPS2, LOPN1, SPF, NPF}')

    # Convert units and return
    
    if unit == 'deg':
        return p[0], p[1], p[2]
    elif unit == 'rad':
        return np.deg2rad(p[0]), np.deg2rad(p[1]), np.deg2rad(p[2])
    else:
        errorcode('error', 'Unit do not exist! Use either "deg" or "rad"')




        
def getSolarPanelOrientation(kappa, quarter):

    """Fetch solar panel orientation for specific mission quarter.

    Parameters
    ----------
    quarter : int
        Mission quarter number (starting from 1)
    kappa : float
        Orientation of the solar panels (i.e. roll angle) [deg]

    Return
    ------
    The corrected roll angle of the spacecraft [deg]
    """

    return math.fmod(quarter * 90, 360) - 90 + kappa        





def getJitterNoiseLimitNSR(rms, tdur=3600, camType='normal'):

    """NSR estimate of the systematic jitter noise component.

    Parameters
    ----------
    rms : float
        RMS of jitter signal [ppm]
    tdur : float, narray
        Time duration over which the NSR is estimated. E.g., 3600s for 1h precision.
    camType : str
        Either the normal (N) or fast (F) cameras. Default is normal.

    Return
    ------
    NSR : float
    """

    # Amplitude Spectral Density of the jitter [ppm/sqrt(muHz)]
    
    ASD = 2 * rms / np.sqrt(1e-6)

    # Choose cycle and exposure time [s] for either the normal (N) or fast (F) cameras

    if camType == 'normal':
        tcyc = 25.
    elif camType == 'fast':
        tcyc = 2.5

    # Calculate the jitter noise in time domain

    return ASD * np.sqrt(tcyc / tdur) 

    # PSD domain from Boner+2024:
    # ASD_jitter = 0.54   # [ppm / sqrt(muHz)]
    # x1 = 25     # [s]
    # x2 = 3600   # [s]
    # y1 = 40000  # [muHz]
    # y2 = 278    # [muHz]
    # a = (y2-y1)/(x2-x1)
    # f = a * (tdur - x1) + y1
    # nsr = ASD_jitter * np.sqrt(f)




        
def getPhotonNoiseLimitNSR(mag, passband='P', camType='normal', ncam=1, ntra=1, tdur=3600):

    """NSR estimate in the photon noise limit of bright stars.

    The stellar flux are calculated from the PLATO passband found by 
    Marchiori et al. (2019).

    Parameters
    ----------
    mag : float, ndarray
        The PLATO passband magnitude.
    camType : str
        Either the normal (N) or fast (F) cameras. Default is normal.
    ncam : float, narray
        Number of telescope visibility. N-Cams (6, 12. 18, 24) or F-Cams (2).
    ntra : float, ndarray
        Number of transits that can be co-added by phase-folding.
    tdur : float, ndarray
        Time duration over which the NSR is estimated. E.g., 3600s for 1h precision.

    Return
    ------
    NSR : float, narray
        NSR only valid for the photon noise limit.
    
    FIXME These limits do not seem to match Borner+2024
    TODO update gain values for F-CAMs
    """

    # Choose cycle and exposure time [s] for either the normal or fast cameras

    if camType == 'normal':
        texp = 21.
        tcyc = 25.
    else:
        texp = 2.1
        tcyc = 2.5

    # FEE + CCD gains [ADU/e-]
    gain = 2.18 * 0.0186 
        
    # Flux of stars [e-/s]
    
    if passband == "V":
        f0 = 1.00179e8
        f = 10**(-0.4 * mag) * f0

    elif passband == 'P':
        # The P passband zero-point
        if camType == 'normal':
            zp = 20.77
        if camType == 'fastblue':
            zp = 20.18
        if camType == 'fastred':
            zp = 19.81
        # Calculate flux
        f = 10**(-0.4 * (mag - zp))

    else:
        errorcode('error', f'Wrong {camType} name!')

    # Observed total flux [ADU/exp]

    
    # SNR from pure photon noise and NSR from uncorrelated noise.
    # Gaussian statistic gives sigma --> sigma/sqrt(N)
    #tdur = (tdur / tcyc) * texp
    flux = f * tdur * gain * texp
    nsr  = 1e6 / np.sqrt(flux * ncam * ntra)
        
    return nsr 





def getBackgroundNoiseLimitNSR(mag, passband='P', camType='normal',
                               tdur=3600, bg=124, ncam=1):

    """NSR estimate in the photon noise limit of bright stars.

    The stellar flux are calculated from the PLATO passband found by 
    Marchiori et al. (2019).

    Parameters
    ----------
    mag : float, narray
        Magnitude of stars
    camType : str
        Either the normal (N) or fast (F) cameras. Default is normal.
    Ncam : float, narray
        Number of telescope visibility. N-Cams (6, 12. 18, 24) or F-Cams (2).
    Ntra : float, narray
        Number of transits that can be co-added by phase-folding.
    tdur : float, narray
        Time duration over which the NSR is estimated. E.g., 3600s for 1h precision.
    bg : float
        Constant background noise level [e-/s/pixel] 

    Return
    ------
    NSR : float, narray
        NSR only valid for the photon noise limit.

    FIXME doesn't seem to be correct!
    """

    # The passband zero-point flux
    
    if passband == "V":
        f0 = 1.00179e8
    elif passband == 'P':
        if camType == 'normal':
            f0 = 7.324509159344043e7
        if camType == 'blue':
            f0 = 3.808715439431968e7
        if camType == 'red':
            f0 = 2.7591704260173317e7
    else:
        errorcode('error', f'Wrong {camType} name!')
        
    # Method cf. Matuszewskic+2023 
    gain = 2.5 * 0.0185
    tcyc = 25
    texp = 21
    tdur = (tdur / tcyc) * texp

    f = 10**(-0.4 * mag) * f0 * gain

    return np.sqrt((bg/f)**2 * tdur) / np.sqrt(ncam)




#--------------------------------------------------------------#
#                       PLATONIUM FUNCTIONS                    #
#--------------------------------------------------------------#


def convertQuarterRange(dQ):

    """Function to sort a quarter ranges.
    
    Small function that takes a string of numbers (here quarters)
    and split it up into readable float values used as real number
    ranges. If a single number is given, an quarter integer is 
    returned.
    """
    
    quarters = []
    for part in dQ.split(','):

        if '-' in part:

            # If a range in mag is provided

            q1, q2 = part.split('-')
            q1, q2 = int(q1), int(q2)
            quarters.append(q1)
            quarters.append(q2)
            
        else:

            # If only one mag-value is given select 1 mag around it

            q1 = int(part)
            quarters.append(q1)

    # That's it!
            
    return quarters





def convertMagnitudeRange(dm):

    """Function to sort magnitudes ranges.

    Small function that takes a string of numbers (here of magnitudes)
    and split it up into readable float values used as real number
    ranges. If a single number is given, a selection of 1 mag around
    the imput int/float is returned as a magnitude range.
    
    Used in: PLATOnium/simulator-pic.py

    Parameters
    ----------
    dm : str
        Magnitude value or range (e.g. '10.5' or '10.0-11.5')

    Return
    ------
    magRange : range()
        Corresponding magnitude range using the range() object.
    """
    
    magRange = []
    for part in dm.split(','):
        
        if '-' in part:

            # If a range in mag is provided

            m1, m2 = part.split('-')
            m1, m2 = float(m1), float(m2)
            
        else:
            
            # If only one mag-value is given select 1 mag around it
            
            m1 = float(part) - 0.5
            m2 = float(part) + 0.5
            
        magRange.append(m1)
        magRange.append(m2)

    # That's it!
        
    return magRange        





def getMainSequenceLimit(Teff):
    
    """Function defined the devision between dwarfs and sub-giants.

    We use the limit defined by: Pecaut and Mamajek (2013)
    """

    return 1 + 2 * 1e-7 * (Teff - 4000)**2





def diff(new, old):

    """Find the index where a value is nearest the array values. 
    """

    return (new - old) / old





def superLorentzian(nu, b, sigma):

    """Calculate the super Lorentzian function.
    """
    
    xi = 2. * np.sqrt(2) / np.pi
    
    return (xi * sigma**2. / b) / (1+(nu/b)**4.)





def rebin3(x, xp, fp):

    """Rebinning method by S. Sarkar.
    """
    
    if np.diff(xp).min() < np.diff(x).min():

        # Binning
        x_cum = xp[1:]
        c =  cumulative_trapezoid(fp,xp)
        x_diff =  np.diff(x)
        b = x[:-1] + x_diff/2.

        # Deal with edge points - estimate x diff in the outer directions
        b = np.hstack((x[0] - x_diff[0]/2. , b, x[-1] + x_diff[-1]/2. ))
        c_new = np.interp(b, x_cum, c)
        d = 0.5*(x_diff[:-1] + x_diff[1:])

        # Deal with edge points - estimate x diff in the outer directions
        d = np.hstack((x_diff[0] , d, x_diff[-1]))
        new_f = (c_new[1:] - c_new[:-1] ) / d
    else:
        # Interpolate!
        new_f = np.interp(x, xp, fp, left=0.0, right=0.0)
        
    return x, new_f






def copyInputYAML(field, odir):

    """Function to copy and adjust a yaml ready to launch.

    Parameters
    ----------
    field : str
        Observational PLATO field (e.g. SPF, NPF, LOPS2, LOPN1)
    odir : str, pathlib object
        Absolute output directory (pathlib object)

    Notes
    -----
    The zero-point flux of a P=0 G2V-star [phot/s/m^2/nm] is 
    converted to the PLATO passband since PlatoSim uses the
    V magnitude as a standard.
    """

    # Get files names of YAML files
    yaml_old = Path(os.getenv("PLATO_PROJECT_HOME") + "/inputfiles/inputfile.yaml")
    yaml_new = odir / "inputfile.yaml"

    # Copy YAML if it doesn't exist already
    if not yaml_new.is_file():

        shutil.copy(yaml_old, yaml_new)

        # Find and replace a few strings:
        with open(yaml_new, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('inputfiles/starcatalog.txt', field)
            filedata = filedata.replace('1.00179e8       #', '0.73244782244e8 #')
            filedata = filedata.replace( 'NumColumns:                      100',
                                        f'NumColumns:                      8  ')
            filedata = filedata.replace( 'NumRows:                         100',
                                        f'NumRows:                         8  ')
            filedata = filedata.replace('IncludePhotometry:               no ',
                                        'IncludePhotometry:               yes')
            filedata = filedata.replace('MaskUpdateInterval:              14.0',
                                        'MaskUpdateInterval:              30.5')
            filedata = filedata.replace('GroupByExposure:                 yes',
                                        'GroupByExposure:                 no ')
            filedata = filedata.replace('WriteBiasMaps:                   yes',
                                        'WriteBiasMaps:                   no ')
            filedata = filedata.replace('WriteSmearingMaps:               yes',
                                        'WriteSmearingMaps:               no ')
            filedata = filedata.replace('WriteFlatfieldMap:               yes',
                                        'WriteFlatfieldMap:               no ')
            filedata = filedata.replace('WriteThroughputMaps:             yes',
                                        'WriteThroughputMaps:             no ')
            filedata = filedata.replace('WriteTransmissionEfficiency:     yes',
                                        'WriteTransmissionEfficiency:     no ')
            filedata = filedata.replace('WriteBackgroundMap:              yes',
                                        'WriteBackgroundMap:              no ')
            filedata = filedata.replace('WriteCTI:                        yes',
                                        'WriteCTI:                        no ')
            filedata = filedata.replace('WriteACS:                        yes',
                                        'WriteACS:                        no ')
            filedata = filedata.replace('WriteTelescopeACS:               yes',
                                        'WriteTelescopeACS:               no ')
            filedata = filedata.replace('WriteStarCatalog:                yes',
                                        'WriteStarCatalog:                no ')
            filedata = filedata.replace('WriteStarPositions:              yes',
                                        'WriteStarPositions:              no ')
            filedata = filedata.replace('WriteGhostPositions:             yes',
                                        'WriteGhostPositions:             no ')
            filedata = filedata.replace('WriteCosmics:                    yes',
                                        'WriteCosmics:                    no ')
            # Write the file out again
            with open(yaml_new, 'w') as file:
                file.write(filedata)



def copyVizierInputYAML(field, odir):

    """Function to copy and adjust a yaml ready to launch.

    Parameters
    ----------
    field : str
        Observational PLATO field (e.g. SPF, NPF, LOPS2, LOPN1)
    odir : str, pathlib object
        Absolute output directory (pathlib object)

    Notes
    -----
    The zero-point flux of a P=0 G2V-star [phot/s/m^2/nm] is 
    converted to the PLATO passband since PlatoSim uses the
    V magnitude as a standard.
    """

    # Get files names of YAML files
    yaml_old = Path(os.getenv("PLATO_PROJECT_HOME") + "/inputfiles/inputfile.yaml")
    yaml_new = odir / "inputfile_vizier.yaml"

    # Copy YAML if it doesn't exist already
    if not yaml_new.is_file():

        shutil.copy(yaml_old, yaml_new)

        # Find and replace a few strings:
        with open(yaml_new, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('inputfiles/starcatalog.txt', field)
            filedata = filedata.replace('1.00179e8       #', '0.73244782244e8 #')
            filedata = filedata.replace('BackgroundValue:               -1',
                                        'BackgroundValue:               0')
            filedata = filedata.replace('IncludeCosmicsInSubField:        yes',
                                        'IncludeCosmicsInSubField:        no')
            filedata = filedata.replace('IncludeCosmicsInSmearingMap:     yes',
                                        'IncludeCosmicsInSmearingMap:     no')
            filedata = filedata.replace('IncludeCosmicsInBiasMap:         yes',
                                        'IncludeCosmicsInBiasMap:         no') 
            filedata = filedata.replace('IncludeStrayLight:             yes',
                                        'IncludeStrayLight:             no')           
            filedata = filedata.replace('UseJitter:                       yes',
                                        'UseJitter:                       no')
            filedata = filedata.replace('IncludeAberrationCorrection:     yes',
                                        'IncludeAberrationCorrection:     no')
            filedata = filedata.replace('IncludePointLikeGhosts:          yes',
                                        'IncludePointLikeGhosts:          no')
            filedata = filedata.replace('IncludeChargeDiffusion:      yes',
                                        'IncludeChargeDiffusion:      no')
            filedata = filedata.replace('IncludeFlatfield:                yes',
                                        'IncludeFlatfield:                no')
            filedata = filedata.replace('IncludeDarkSignal:               yes',
                                        'IncludeDarkSignal:               no')
            filedata = filedata.replace('IncludeBFE:                      yes',
                                        'IncludeBFE:                      no')
            filedata = filedata.replace('IncludePhotonNoise:              yes',
                                        'IncludePhotonNoise:              no')
            filedata = filedata.replace('IncludeReadoutNoise:             yes',
                                        'IncludeReadoutNoise:             no')
            filedata = filedata.replace('IncludeCTIeffects:               yes',
                                        'IncludeCTIeffects:               no')
            filedata = filedata.replace('IncludeOpenShutterSmearing:      yes',
                                        'IncludeOpenShutterSmearing:      no')
            filedata = filedata.replace('IncludeQuantumEfficiency:        yes',
                                        'IncludeQuantumEfficiency:        no')
            filedata = filedata.replace('IncludeRelativeTransmissivity:   yes',
                                        'IncludeRelativeTransmissivity:   no')
            filedata = filedata.replace('IncludePolarization:             yes',
                                        'IncludePolarization:             no')
            filedata = filedata.replace('IncludeParticulateContamination: yes',
                                        'IncludeParticulateContamination: no')
            filedata = filedata.replace('IncludeMolecularContamination:   yes',
                                        'IncludeMolecularContamination:   no')
            filedata = filedata.replace('IncludeConvolution:              yes',
                                        'IncludeConvolution:              no')
            filedata = filedata.replace('IncludeFullWellSaturation:       yes',
                                        'IncludeFullWellSaturation:       no')
            filedata = filedata.replace('IncludeDigitalSaturation:        yes',
                                        'IncludeDigitalSaturation:        no')
            filedata = filedata.replace('IncludeQuantisation:             yes',
                                        'IncludeQuantisation:             no')
            filedata = filedata.replace('IncludeGainNonlinearity:         yes',
                                        'IncludeGainNonlinearity:         no')
            filedata = filedata.replace('WritePixelMaps:                  yes',
                                        'WritePixelMaps:                  no')
            filedata = filedata.replace('WriteBiasMaps:                   yes',
                                        'WriteBiasMaps:                   no ')
            filedata = filedata.replace('WriteSmearingMaps:               yes',
                                        'WriteSmearingMaps:               no ')
            filedata = filedata.replace('WriteFlatfieldMap:               yes',
                                        'WriteFlatfieldMap:               no ')
            filedata = filedata.replace('WriteThroughputMaps:             yes',
                                        'WriteThroughputMaps:             no ')
            filedata = filedata.replace('WriteTransmissionEfficiency:     yes',
                                        'WriteTransmissionEfficiency:     no ')
            filedata = filedata.replace('WriteBackgroundMap:              yes',
                                        'WriteBackgroundMap:              no ')
            filedata = filedata.replace('WriteCTI:                        yes',
                                        'WriteCTI:                        no ')
            filedata = filedata.replace('WriteACS:                        yes',
                                        'WriteACS:                        no ')
            filedata = filedata.replace('WriteTelescopeACS:               yes',
                                        'WriteTelescopeACS:               no ')
            filedata = filedata.replace('WriteStarCatalog:                yes',
                                        'WriteStarCatalog:                no ')
            filedata = filedata.replace('WriteGhostPositions:             yes',
                                        'WriteGhostPositions:             no ')
            filedata = filedata.replace('WriteCosmics:                    yes',
                                        'WriteCosmics:                    no ')

            # Write the file out again
            with open(yaml_new, 'w') as file:
                file.write(filedata)

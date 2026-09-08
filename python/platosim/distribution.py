#!/usr/bin/env python3

"""
Create distributions for several input aspects for PlatoSim.

This class provides a catalogue of distributions that can be
used to generate more realistic simulation with PlatoSim.

For usage, see the Jupyter tutorial notebooks available at:
"PlatoSim/docs/tutorials"
"""

# Built-in
import random

# PlatoSim standard
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.random import default_rng


#==============================================================#
#                         BEGIN CLASS                          #
#==============================================================#


class Distribution(object):

    """
    This class provides the Python interface to different distribution models.
    """


    def __init__(self, filename):

        """Open the HDF5 output file
        """
        
        self.filename = filename




    def __del__(self):

        """Destructor
        """

        pass


    #--------------------------------------------------------------#
    #                      MATHEMATICAL MODELS                     #
    #--------------------------------------------------------------#


    def expFunction(self, x, a, b, c):

        """
        Evaluation of the function a * EXP(-b * x) + c, in variable x.

        INPUT: x: Variable for which the evaluate the exponential function.
        INPUT: a: Scale factor for the exponential component.
        INPUT: b: Scale factor for the variable in the argument of the exponential function.
        INPUT: c: Constant to add to the exponential function.

        OUTPUT: Result of evaluating the function y = a * EXP(-b * x) + c, for x.
        """

        return a * np.exp(-b * x) + c




    
    def inverseExpFunction(self, y, a, b, c):

        """
        Evaluation of the inverse of the function y = a * EXP(-b * x) + c, in variable x,
        which is x = -LN((y - c) / a) / b.

        INPUT: y: Result of the evaluation of the function, for variable x.
        INPUT: x: Variable for which the evaluate the exponential function.
        INPUT: a: Scale factor for the exponential component.
        INPUT: b: Scale factor for the variable in the argument of the exponential function.
        INPUT: c: Constant to add to the exponential function.

        OUTPUT: Result of evaluating the inverse of function y = a * EXP(-b * x) + c, for y.
        """
        return -np.log((y - c) / a) / b


    #--------------------------------------------------------------#
    #                       STELLAR CATALOGUES                     #
    #--------------------------------------------------------------#


    def readMagnitudes(self):
        """
        Reads the magnitudes from the star catalogues with the given filename of the HDF5.  
        This function assumed is that this file comprises three (space-separated) columns:
            (1) right ascension of the stars, expressed in degrees
            (2) declination of the stars, expressed in degrees
            (3) magnitude of the stars
        """

        # Open the star catalogue

        catalog = open(self.filename, 'r')

        # Used to store the magnitude values from the catalogue

        magnitudes = []

        # Read the file line-by-line and store the 3rd column in the array of magnitudes

        for star in catalog:

            magnitude = star.split()[2]     # 3rd column
            magnitudes.append(magnitude)

        magnitudes = np.asarray(magnitudes, dtype = np.double)  # Conversion to numpy array

        return magnitudes



    

    def getRandomPosition(self, numRows, numColumns):

        """
        Generates random position (row, column) in the sub-field with the given dimensions, 
        using a uniform distribution for row and column.  This does not take the position of the
        sub-field on the CCD into account.

        INPUT: numRows: Number of rows in the sub-field, expressed in pixels.
        INPUT: numColumns: Number of columns in the sub-field, expressed in pixels.

        OUTPUT: Randomly generated position (row, column) in the sub-field.  Note that these shoud NOT
                be integer values (i.e. stars don't necessarily fall in the centre of a pixel).
        """

        randomRow    = random.uniform(0, numRows    - 1)  # Not necessarily an integer value
        randomColumn = random.uniform(0, numColumns - 1)  # Not necessarily an integer value

        return randomRow, randomColumn


    




    def getRandomMagnitude(self, minMagnitude, maxMagnitude, a, b, c):
        """
        Generates random magnitude in the given magnitude range, using the (fit to the)
        distribution of the magnitudes in a real catalogue (a * EXP(-b * magnitude) + c).

        INPUT: minMagnitude: Lowest magnitude (i.e. of the brightest star).
        INPUT: maxMagnitude: Highest magnitude (i.e. of the faintest star).
        INPUT: a: Scale factor for the exponential component.
        INPUT: b: Scale factor for the variable in the argument of the exponential function.
        INPUT: c: Constant to add to the exponential function.
        """

        # Inverse transform sampling method

        y = 1e3 + 1e4*random.random()
        magnitude = self.inverseExpFunction(y, a, b, c)

        # Make sure the generated magnitude is withing the allowed range
        # (keep on generating new values until it is)
        
        while (magnitude < minMagnitude) or (magnitude > maxMagnitude):     
            magnitude = self.getRandomMagnitude(minMagnitude, maxMagnitude, a, b, c)

        return magnitude

    


    

    

    def getMagnitudeDistribution(self, minMag=2, maxMag=15, plot=False):
        """
        Extracting the magnitude values from a star catalogue (containing right ascension, 
        declination, and magnitude), plotting a histogram, fitting a function
        a * EXP(-b * magnitude) + c to it, and showing the resulting fit.

        - readMagnitudes: Reading the magnitude values from a catalogue
        - expFunction: Function f(x) = a * EXP(-b * x) + c
        - inverseExpFunction: Inverse of function f(x) = a * EXP(-b * x + c)
        - getMagnitudeDistribution: Plotting the magnitude histogram and the fit.
        Reads the magnitudes from a given star catalogue, plots a histogram and fits the
        magnitude distribution with a function a * EXP(-b * magnitude) + c.

        INPUT: filename: Filename (full path) of the star catalogue from which to read the magnitudes.

        OUTPUT: Optimised parameters (a, b, c) of fitting function a * EXP(-b * magnitude) + c 
                to the magnitude distribution in the given star catalogue.
        """

        # Read the magnitudes from a real catalogue

        import pandas as pd
        df = pd.DataFrame(self.readMagnitudes(), columns=['mag'])
        counts, bins = np.histogram(df.mag, bins=100)
        
        # Find magnitude limits

        df0 = df[(df.mag > minMag) & (df.mag < maxMag)]
        counts0, bins0 = np.histogram(df0.mag, bins=100)
        
        # Fit function: a * EXP(-b * magnitude) + c, to the magnitude distribution
        
        x = (bins0[1:] + bins0[:-1]) / 2   # Centre of bins
        fitMag, cov = curve_fit(self.expFunction, x, counts0, p0=(1, 1e-6, 1))
        
        # Plot if requested
        
        if plot:
            plt.figure(figsize=(8,5))
            # Plot input data
            plt.hist(df.mag, bins=100,
                     facecolor='b', edgecolor='b',
                     fill=True, log=False, alpha=0.3,
                     label="Stellar input catalogue")
            # Plot fit
            xx = np.linspace(minMag, maxMag, 100)
            yy = self.expFunction(xx, *fitMag)
            plt.plot(xx, yy, 'm', lw=2, label="Best fit to magnitude range")
            plt.xlabel("Magnitude")
            plt.ylabel("Number of stars")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

        # Finito!
            
        return fitMag






    def getStarCatalog(self, numStars, minMag, maxMag, subfieldNumRows, subfieldNumCols,
                       subfieldZeroRow=False, subfieldZeroCol=False, plot=False):

        # Fetch coefficient for the exponential distribution fitted to the SPF:
        
        a, b, c = self.getMagnitudeDistribution(minMag, maxMag, plot=plot)

        # Loop over each star to get random row, col, and mag

        star = np.arange(numStars)
        rows = np.zeros(numStars)
        cols = np.zeros(numStars)
        mags = np.zeros(numStars)

        for i in range(numStars):

            # Random position in the subfield (uniform distribution)
            
            randomRow, randomCol = self.getRandomPosition(subfieldNumRows, subfieldNumCols)

            # Account for subfield zero-point if requested

            if subfieldZeroRow and subfieldZeroCol:
                randomRow += subfieldZeroRow
                randomCol += subfieldZeroCol
                
            # Random magnitude between choices (exponential distribution)
            
            randomMag = self.getRandomMagnitude(minMag, maxMag, a, b, c)

            # Append to arrays
            
            rows[i] = randomRow
            cols[i] = randomCol
            mags[i] = randomMag
            
        # Add star which shows blooming

        # Random position in the sub-field (uniform distribution)
        # bloomingRow, bloomingColumn = 25, 25
        # #bloomingRow, bloomingColumn = getRandomPosition(subFieldDimensions, subFieldDimensions)
        # rows.append(bloomingRow)
        # columns.append(bloomingColumn)
        # magnitudes.append(7.0)

        return rows, cols, mags, star

    




# def picOfDestiny(distribution, prange):
#     """
#     This function randomly picks a value from any gievn distribution and returns it.
#     The distribution must consist of values between 0 and 1 with its peak at 1. This function picks a
#     random value from the allowed range and then uses a distribution to get a P number
#     between 0 and 1, it then rolls a dice and chekcs wheter the dice roll is under the
#     P number. If it is, then the picked value is returned. This ensures a recration of
#     the distribution shape over thousands of picks
#     """

#     pick = random.random()*(prange[1]-prange[0]) + prange[0]
#     p = distribution(pick)
#     roll = random.random()
#     if roll < p:
#         return pick
#     else:
#         return distribution_pick(distribution, range)   

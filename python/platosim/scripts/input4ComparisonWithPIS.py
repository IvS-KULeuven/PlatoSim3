"""
Simulation of different (100 x 100 pixels) sub-fields on CCD 2 with fake stars:
- the position of the stars is sampled using a uniform distribution over the sub-field
- the magnitude of the stars is sampled using the (fitted) distribution of magnitudes
  in a real catalogue
- four sub-fields are defined:
    (1) "lowerLeft": in the lower left corner of the CCD (i.e. close to the readout register,
        small PSF)
    (2) "lowerRight": in the lower right corner of the CCD (i.e. close to the readout register,
        large PSF)
    (3) "upperLeft": in the upper left corner of the CCD (i.e. far from the readout register,
        large PSF)
    (4) "middle": in the centre of the CCD (sub-field overlaps with both detector halves)
    (5) "upperRight": in the upper right corner of the CCD (i.e. far from the readout register,
         large PSF)
"""





#########
# Imports
#########

from imp import reload
import platosim.simulation
from pyparsing import nums
reload(platosim.simulation)
from platosim.simulation import Simulation
import platosim.referenceFrames
reload(platosim.referenceFrames)
from platosim.referenceFrames import pixelToSkyCoordinates, skyToPixelCoordinates
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from platosim.magnitudeDistribution import *





###################
# Auxiliary methods
###################

# def getRandomPosition(numRows, numColumns, centerRow, centerColumn):
#     
#     """
#     Generates random position (row, column) in the sub-field with the given dimensions,
#     using a uniform distribution for row and column.
#     
#     INPUT: numRows: Number of rows in the sub-field, expressed in pixels.
#     INPUT: numColumns: Number of columns in the sub-field, expressed in pixels.
#     INPUT: centerRow: Row index of the centre of the sub-field.
#     INPUT: centerColumn: Column index of the centre of the sub-field.
#     
#     OUTPUT: Randomly generated position (row, column) in the sub-field.  Note that these shoud NOT
#             be integer values (i.e. stars don't necessarily fall in the centre of a pixel).
#     """
#     
#     randomRow = random.uniform(0, numRows - 1)          # Not necessarily an integer value
#     randomColumn = random.uniform(0, numColumns - 1)    # Not necessarily an integer value
#     
#     randomRow = randomRow + (centerRow - numRows / 2)
#     randomColumn = randomColumn + (centerColumn - numColumns/ 2)
#     
#     return randomRow, randomColumn





def getRandomPosition(numRows, numColumns):
    
    """
    Generates random position (row, column) in the sub-field with the given dimensions, 
    using a uniform distribution for row and column.  This does not take the position of the
    sub-field on the CCD into account.
    
    INPUT: numRows: Number of rows in the sub-field, expressed in pixels.
    INPUT: numColumns: Number of columns in the sub-field, expressed in pixels.
    
    OUTPUT: Randomly generated position (row, column) in the sub-field.  Note that these shoud NOT
            be integer values (i.e. stars don't necessarily fall in the centre of a pixel).
    """
    
    randomRow = random.uniform(0, numRows - 1)          # Not necessarily an integer value
    randomColumn = random.uniform(0, numColumns - 1)    # Not necessarily an integer value
    
    return randomRow, randomColumn



    
def getRandomMagnitude(minMagnitude, maxMagnitude, a, b, c):
    
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
    magnitude = inverseExpFunction(random.random(), a, b, c)
    print(a, b, c)
    
    
    # Make sure the generated magnitude is withing the allowed range
    # (keep on generating new values until it is)
    
    while (magnitude < minMagnitude) or (magnitude > maxMagnitude):
        
        magnitude = getRandomMagnitude(minMagnitude, maxMagnitude, a, b, c)
        
    return magnitude





# def testRandomMagnitude():
#     
#     m = [getRandomMagnitude(min(x), max(x), a, b, c) for i in range(1000)]
#     plt.hist(m, bins = 100, normed = 'True')
#     plt.show()
# 
# 
# 
# 
# 
# def testRandomPosition():
#     
#     rows = []
#     columns = []
#     
#     for i in range(1000):
#         
#         row, column = getRandomPosition(100, 100)
#         
#         rows.append(row)
#         columns.append(column)
#     
#     rows = np.asarray(rows, dtype = np.double)
#     columns = np.asarray(columns, dtype = np.double)
#     
#     plt.scatter(columns, rows)
#     plt.xlabel("Row")
#     plt.ylabel("Column")
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.show()





def getStarCatalog(numStars, subFieldDimensions):
    
    filename = os.environ["PLATO_PROJECT_HOME"] + "/inputfiles/starField_RA180Dec-70.txt"
    a, b, c = getMagnitudeDistribution(filename)
    
    rows = []
    columns = []
    magnitudes = []
    
    for starIndex in range(numStars - 1):
    
        randomRow, randomColumn = getRandomPosition(subFieldDimensions, subFieldDimensions)     # Random position in the sub-field (uniform distribution)
        randomMagnitude = getRandomMagnitude(9, 15, a, b, c)
    
        rows.append(randomRow)
        columns.append(randomColumn)
        magnitudes.append(randomMagnitude)
    
    # Add star which shows blooming
    
    #bloomingRow, bloomingColumn = getRandomPosition(subFieldDimensions, subFieldDimensions)     # Random position in the sub-field (uniform distribution)
    bloomingRow, bloomingColumn = 25, 25
    rows.append(bloomingRow)
    columns.append(bloomingColumn)
    magnitudes.append(7.0)
    
    return np.array(rows), np.array(columns), np.array(magnitudes)

    



##########################
# Configuration parameters
##########################

# Working directory

workDir = os.environ["PLATO_WORKDIR"]

# CCD related configuration parameters

ccdDimensions = 4510

# Sub-field related configuration parameters
# (1) "lowerLeft": in the lower left corner of the CCD (i.e. close to the readout register,
#     small PSF)
# (2) "lowerRight": in the lower right corner of the CCD (i.e. close to the readout register,
#     large PSF)
# (3) "upperLeft": in the upper left corner of the CCD (i.e. far from the readout register,
#     large PSF)
# (4) "middle": in the centre of the CCD (sub-field overlaps with both detector halves)
# (5) "upperRight": in the upper right corner of the CCD (i.e. far from the readout register,
#      large PSF)

numStars = 200

numSubfields = 5
subFieldDimensions = 100

subFieldNames = ["lowerLeft", "lowerRight", "upperLeft", "middle", "upperRight"]
subFieldCenterRows = [subFieldDimensions / 2, subFieldDimensions / 2, ccdDimensions - subFieldDimensions / 2, ccdDimensions / 2, ccdDimensions - subFieldDimensions / 2]
subFieldCenterColumns = [subFieldDimensions / 2, ccdDimensions - subFieldDimensions / 2, subFieldDimensions / 2, ccdDimensions / 2, ccdDimensions - subFieldDimensions / 2]

# PSF related configuration parameters

psfModel = "MappedFromFile"

# Star catalogue

rowsCatalog, columnsCatalog, magnitudesCatalog = getStarCatalog(numStars, subFieldDimensions)





for simulationIndex in range(numSubfields):
    
    sim = Simulation(subFieldNames[simulationIndex], outputDir = workDir)
    
    # Platform related configuration parameters
    
    sim["Platform/UseJitterFromFile"] = 1
    sim["Platform/JitterFileName"] = "inputfiles/AOCS_NM_pointing_7h.txt"
    
    # PSF related configuration parameters

    sim["PSF/Model"] = psfModel
    
    # CCD related configuration parameters    
    
    ccdCode = "4"
    sim["CCD/Position"] = ccdCode

    # Camera related configuration parameters
    
    sim["Camera/IncludeFieldDistortion"] = 0
    sim["Camera/FocalLength/Source"] = "ConstantValue"
    
    # Observing parameters    

    sim["ObservingParameters/NumExposures"] = 500 # 500 * (22s + 3s) ~ 3.5h #570 # 570 * (22s + 3s) ~ 4h
    sim["ObservingParameters/RApointing"] = 10.0
    sim["ObservingParameters/DecPointing"] = 10.0
    
    #sim["Platform/UseJitter"] = 0
    #sim["Telescope/UseDrift"] = 0
    #sim["Camera/IncludeAberrationCorrection"] = 0
    #sim["Camera/IncludeFieldDistortion"] = 0
    #sim["CCD/IncludeFlatfield"] = 0
    #sim["CCD/IncludePhotonNoise"] = 0
    #sim["CCD/IncludeReadoutNoise"] = 0
    #sim["CCD/IncludeCTIeffects"] = 0
    #sim["CCD/IncludeOpenShutterSmearing"] = 0
    #sim["CCD/IncludeQuantumEfficiency"] = 0
    #sim["CCD/IncludeNaturalVignetting"] = 0
    #sim["CCD/IncludeMechanicalVignetting"] = 0
    #sim["CCD/IncludePolarization"] = 0
    #sim["CCD/IncludeParticulateContamination"] = 0
    #sim["CCD/IncludeMolecularContamination"] = 0
    #sim["CCD/IncludeConvolution"] = 1
    #sim["CCD/IncludeFullWellSaturation"] = 1
    #sim["CCD/IncludeDigitalSaturation"] = 0
    #sim["CCD/IncludeQuantisation"] = 0
    
    # Sub-field related configuration parameters
    
    subFieldCenterRa, subFieldCenterDec = pixelToSkyCoordinates(sim, ccdCode, subFieldCenterColumns[simulationIndex], subFieldCenterRows[simulationIndex])                  # (x, y) = (column, row) -> (RA, Dec) [radians]
    sim.setSubfieldAroundPixelCoordinates(ccdCode, subFieldCenterColumns[simulationIndex], subFieldCenterRows[simulationIndex], subFieldDimensions, subFieldDimensions)     # (x, y)
    sim["SubField/SubPixels"] = 64 

    # Generate star catalogue
    
    starCatalogFilename = workDir + "catalog4" +  subFieldNames[simulationIndex] + ".starcat"
    
    raCatalog = []
    decCatalog = []
    
    for starIndex in range(numStars):
        
        #randomRow, randomColumn = getRandomPosition(subFieldDimensions, subFieldDimensions, subFieldCenterRows[simulationIndex], subFieldCenterColumns[simulationIndex])     # Random position in the sub-field (uniform distribution)
        randomRow = rowsCatalog[starIndex] + subFieldCenterRows[simulationIndex] - subFieldDimensions / 2
        randomColumn = columnsCatalog[starIndex] + subFieldCenterColumns[simulationIndex] - subFieldDimensions / 2
        randomRa, randomDec = pixelToSkyCoordinates(sim, ccdCode, randomColumn, randomRow)                                                                                   # (x, y) -> (RA, Dec) [radians]
        raCatalog.append(randomRa)
        decCatalog.append(randomDec)
    
    raCatalog = np.array(raCatalog)                     # Conversion to numpy array
    decCatalog = np.array(decCatalog)                   # Conversion to numpy array
    
    raCatalog[raCatalog > math.pi] = raCatalog[raCatalog > math.pi] - 2 * math.pi       # RA should be in range [-PI, PI]
    raCatalog = np.rad2deg(raCatalog)                                                   # RA [radians] -> [degrees]
    decCatalog = np.rad2deg(decCatalog)                                                 # Dec [radians] -> [degrees]
    
    np.savetxt(starCatalogFilename, np.transpose([raCatalog, decCatalog, magnitudesCatalog]), fmt=['%11.6f', '%11.6f', '%8.4f'])     # Store the catalogue
    sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename
    
    configurationFilename = workDir + "config4" + subFieldNames[simulationIndex] + ".yaml"
    sim.writeYamlConfigurationFile(configurationFilename)
    
    simFile = sim.run()
    simFile.showImage(1)

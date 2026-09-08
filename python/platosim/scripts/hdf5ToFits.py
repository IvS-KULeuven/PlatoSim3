from platosim.simfile import SimFile
from astropy.io import fits
import platosim.referenceFrames as rf
import math
from pathlib import Path
from datetime import datetime, timedelta
import os

def hdf5ToFits(inputFilename, outputFilename):

    """
    PURPOSE: Convert the given HDF5 file (PlatoSim output file) to a
             FITS file with a header compring the sub-field information.
    
    INPUT:
        - inputFilename: Filename of the PlatoSim HDF5 file.
        - outputFilename: Filename of the FITS file comprising the given exposure 
                          and a header that contains the sub-field information.
    """

    cwd = os.getcwd()
    outputDir = os.path.dirname(outputFilename)

    # If you don't do this, appending will fail when the output directory is
    # not the same as the directory you are running the script from

    os.chdir(outputDir)

    simFile = SimFile(inputFilename)
    outputFilePath = Path(outputFilename)

    # The primary HDU contains only a header and no image data
    # (if the output filename is already in use, an exception will be thrown)

    primaryHDU = fits.PrimaryHDU()                             # Primary HDU
    primaryHDU.header, timestamp = getPrimaryHeader(simFile)   # Header of the primary HDU, timestamp of the start of the conversion

    primaryHDU.writeto(outputFilePath)              # Creation of the file

    # Write the exposures, bias register maps (serial pre-scan), and smearing maps (parallel over-scan)
    # to individual layers in the FITS file
    # Make sure to adapt the timestamp!

    imageHeader = getImageHeader(simFile)
    biasHeader = getSerialPreScanHeader(simFile)    
    smearingHeader = getParallelOverScanHeader(simFile)

    beginExposureNumber = simFile.getInputParameter("ObservingParameters", "BeginExposureNr")
    numExposures = simFile.getInputParameter("ObservingParameters", "NumExposures")

    for exposure in range(beginExposureNumber, beginExposureNumber + numExposures):

        formattedTimestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Image data

        imageHeader["DATE-OBS"] = formattedTimestamp
        imageHeader["EXTVER"] = exposure

        image = simFile.getImage(exposure)
        fits.append(outputFilePath, image, imageHeader)

        # Serial pre-scan (bias register maps)
        try:
            readoutMode = simFile.getInputParameter("CCD/ReadoutMode", "ReadoutMode").decode("utf-8")
        except:
            readoutMode = simFile.getInputParameter("CCD/ReadoutMode", "ReadoutMode")
        

        if readoutMode == "Nominal":

            biasHeader["DATE-OBS"] = formattedTimestamp

            biasMapLeft = simFile.getBiasMapLeft(exposure)
            biasHeader["EXTNAME"] = f"SPRE_{biasHeader['CCD_ID']}_E"
            biasHeader["EXTVER"] = exposure
            fits.append(outputFilePath, biasMapLeft, biasHeader)
            
            biasMapRight = simFile.getBiasMapRight(exposure)
            biasHeader["EXTNAME"] = f"SPRE_{biasHeader['CCD_ID']}_F"
            fits.append(outputFilePath, biasMapRight, biasHeader)

        # Parallel over-scan (smearing map)

        smearingHeader["DATE-OBS"] = formattedTimestamp
        smearingHeader["EXTVER"] = exposure

        try:

            smearingMap = simFile.getSmearingMap(exposure)
            fits.append(outputFilePath, smearingMap, smearingHeader)

        except:

            pass

        # Update timestamp for the next exposure

        cycleTime = simFile.getInputParameter("ObservingParameters", "CycleTime") 
        timestamp += timedelta(seconds=cycleTime)

    # Only one window is stored in the FITS file

    fits.setval(outputFilePath, "NWINDOWS", value=1)

    # Go back to the directory you were at the start

    os.chdir(cwd)


def getImageHeader(simFile: SimFile):

    """
    PURPOSE: Creates and returns a FITS header with the information on the given simulation.

    INPUT:
        - simFile: File with the PlatoSim simulation.

    OUTPUT:
        - FITS header with the information on the given simulation.
    """

    header = fits.Header()

    header["SIMPLE"] = "T"

    # Dimensionality of the sub-field

    header["NAXIS"] = (2, "Dimensionality of the sub-field")

    # Focal length (this is needed for the conversion to field angles)

    header["FOCALLEN"] = (simFile.getInputParameter("Camera/FocalLength", "ConstantValue") * 1000.0, "Focal length [mm]")   # Focal length [mm], "Focal length [mm]")

    # Linear coordinate transformation from sub-field to focal-plane coordinates

    header["ctype1"] = ("LINEAR", "Linear coordinate transformation")
    header["ctype2"] = ("LINEAR", "Linear coordinate transformation")

    # Focal-plane coordinates are expressed in mm

    header["CUNIT1"] = ("MM", "Target unit in the column direction [mm]")
    header["CUNIT2"] = ("MM", "Target unit in the row direction [mm]")

    # Pixel size

    pixelSize = simFile.getInputParameter("CCD", "PixelSize")    # Pixel size [µm]
    cdelt = pixelSize / 1000.0                                   # Pixel size [mm]

    header["CDELT1"] = (cdelt, "Pixel size in the x-direction [mm]")
    header["CDELT2"] = (cdelt, "Pixel size in the y-direction [mm]")

    # Dimensions of the sub-field

    numRows = simFile.getInputParameter("SubField", "NumRows")           # Number of rows in the sub-field
    numColumns = simFile.getInputParameter("SubField", "NumColumns")     # Number of columns in the sub-field

    header["NAXIS1"] = (numColumns, "Number of columns in the sub-field")
    header["NAXIS2"] = (numRows, "Number of rows in the sub-field")

    # CCD origin + corresponding focal-plane coordinates

    # For the CCD origin, we know the coordinates in the source coordinate system
    # (i.e. the sub-field reference frame), from the sub-field zeropoint, and
    # can calculate the coordinates in the target coordinate system (i.e. in the
    # focal-plane reference frame).  Hence, we use the CCD origin to be the
    # reference point of the coordinate transformation (from the sub-field
    # reference frame to the focal-plane reference frame):
    #   - the coordinates of the reference point in the source reference frame
    #     (i.e. in the sub-field reference frame) need to go in the CRPIXi
    #     keywords;
    #   - the coordinates of the reference point in the target reference frame
    #     (i.e. in the focal-plane reference frame) need to go in the CRVALi
    #     keywords.

    ccdCode = simFile.getInputParameter("CCD", "Position")
    subfieldZeropointRow = int(simFile.getInputParameter("SubField", "ZeroPointRow"))           # Sub-field zeropoint row [pixels]
    subfieldZeropointColumn = int(simFile.getInputParameter("SubField", "ZeroPointColumn"))     # Sub-field zeropoint column [pixels]

    header["CRPIX1"] = (-subfieldZeropointColumn, "Sub-field column of the CCD origin [pixels]")
    header["CRPIX2"] = (-subfieldZeropointRow, "Sub-field row of the CCD origin [pixels]")

    try:
        
        ccdCode = int(ccdCode)

        header["CCD_ID"] = (ccdCode, "CCD code")
        
        ccdOffsetX = simFile.getInputParameter("CCDPositions", "OriginOffsetX")[ccdCode - 1]
        ccdOffsetY = simFile.getInputParameter("CCDPositions", "OriginOffsetY")[ccdCode - 1]
        ccdOrientationAngleDegrees = simFile.getInputParameter("CCDPositions", "Orientation")[ccdCode - 1]

    except ValueError:

        header["CCD_ID"] = ("Custom", "CCD code")

        ccdOffsetX = simFile.getInputParameter("CCD", "OriginOffsetX")
        ccdOffsetY = simFile.getInputParameter("CCD", "OriginOffsetY")
        ccdOrientationAngleDegrees = simFile.getInputParameter("CCD", "Orientation")

    ccdOrientationAngleRadians = math.radians(ccdOrientationAngleDegrees)

    crval1, crval2 = rf.pixelToFocalPlaneCoordinates(0, 0, pixelSize, ccdOffsetX, ccdOffsetY, ccdOrientationAngleRadians)

    header["CRVAL1"] = (crval1, "FP x-coordinate of the CCD origin [mm]")
    header["CRVAL2"] = (crval2, "FP y-coordinate of the CCD origin [mm]")

    # Orientation angle of the CCD
    
    header["crota2"] = (ccdOrientationAngleDegrees, "CCD orientation angle [degrees]",)

    header["cd1_1"] = (cdelt * math.cos(ccdOrientationAngleRadians), "Pixel size x cos(CCD orientation angle)",)
    header["cd1_2"] = (-cdelt * math.sin(ccdOrientationAngleRadians), "-Pixel size x sin(CCD orientation angle)",)
    header["cd2_1"] = (cdelt * math.sin(ccdOrientationAngleRadians), "Pixel size x sin(CCD orientation angle)",)
    header["cd2_2"] = (cdelt * math.cos(ccdOrientationAngleRadians), "Pixel size x cos(CCD orientation angle)",)

    # Additional keywords

    # header["TELESCOP"] = (setup["camera_id"], "Camera ID")
    # header["INSTRUME"] = (setup["camera_id"], "Camera ID")
    # header["SITENAME"] = (setup["site_id"], "Name of the test site")
    # header["EXPOSURE"] = (exposureTime, "Exposure time [s]")

    # Using this keyword, the image will end up in the correct extension

    header["EXTNAME"] = f"IMAGE_{header['CCD_ID']}"

    return header


def getSerialPreScanHeader(simFile: SimFile):

    """
    PURPOSE: Creates and returns a FITS header with the information on the serial pre-scan
             of the given simulation. 

    INPUT:
        - simFile: File with the PlatoSim simulation.

    OUTPUT:
        - FITS header with the information on the serial pre-scan of the given simulation.
    """

    header = fits.Header()

    header["SIMPLE"] = "T"

    # Dimensionality of the serial pre-scan

    header["NAXIS"] = (2, "Dimensionality of the sub-field")

    header["NAXIS1"] = (simFile.getInputParameter("SubField", "NumBiasPrescanColumns"), "Number of columns in the serial pre-scan")
    header["NAXIS2"] = (simFile.getInputParameter("SubField", "NumBiasPrescanRows"), "Number of rows in the serial pre-scan")

    # CCD

    try:
        
        ccdCode = int(simFile.getInputParameter("CCD", "Position"))
        header["CCD_ID"] = (ccdCode, "CCD code")

    except ValueError:

        header["CCD_ID"] = ("Custom", "CCD code")

    return header


def getParallelOverScanHeader(simFile: SimFile):

    """
    PURPOSE: Creates and returns a FITS header with the information on the parallel over-scan
             of the given simulation.  Note that this will only be included in case of
             nominal readout mode.

    INPUT:
        - simFile: File with the PlatoSim simulation.

    OUTPUT:
        - FITS header with the information on the parallel over-scan of the given simulation.
    """

    header = fits.Header()

    header["SIMPLE"] = "T"

    # Dimensionality of the parallel over-scan

    header["NAXIS"] = (2, "Dimensionality of the sub-field")

    header["NAXIS1"] = (simFile.getInputParameter("SubField", "NumColumns"), "Number of columns in the parallel over-scan")
    header["NAXIS2"] = (simFile.getInputParameter("SubField", "NumSmearingOverscanRows"), "Number of rows in the parallel over-scan")

    # CCD

    try:
        
        ccdCode = int(simFile.getInputParameter("CCD", "Position"))
        header["CCD_ID"] = (ccdCode, "CCD code")

    except ValueError:

        header["CCD_ID"] = ("Custom", "CCD code")

    # Using this keyword, the parallel over-scan will end up in the correct extension


    header["EXTNAME"] = f"POVER_{header['CCD_ID']}"

    return header


def getPrimaryHeader(simFile: SimFile):

    """
    PURPOSE: Creates and returns the primary header (i.e. the header of the primary HDU).  This contains 
             the information that is specific for the camera.
    
    INPUT:
        - simFile: File with the PlatoSim simulation.

    OUTPUT:
        - FITS header with the camera-specific information on the given simulation.
    """

    primaryHeader = fits.Header()

    primaryHeader["SIMPLE"] = "T"

    # Focal length [mm] (this is needed for the conversion to field angles)

    primaryHeader["FOCALLEN"] = (simFile.getInputParameter("Camera/FocalLength", "ConstantValue") * 1000.0, "Focal length [mm]")   # Focal length [mm], "Focal length [mm]")

    # Number of windows

    primaryHeader["NWINDOWS"] = (0, "Number of windows")

    timestamp = datetime.now()
    primaryHeader["DATE-OBS"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return primaryHeader, timestamp

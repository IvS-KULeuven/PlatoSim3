#!/usr/bin/env python3

"""
Extract information from the PlatoSim HDF5 output file.

This class provides a direct interface to extract and inspect
the HDF5 output files that PlatoSim produce upon execution.

For usage see the Jupyter tutorial notebooks available at:
"PlatoSim/docs/tutorials"
"""

# Built-in
import os

# PlatoSim standard
import h5py
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

# PlatoSim functions
from platosim.utilities import imageClip, normalize


#==============================================================#
#                         SIMFILE CLASS                        #
#==============================================================#


class SimFile (object):

    """Class that provides the Python interface to an HDF5 file.

    Usage example:
    >> f = SimFile("output.hdf5")
    """

    def __init__(self, filename):

        """Open the HDF5 file.

        Parameters
        ----------
        filename : str
            String containing the full path of the HDF5 file
        """

        self.filename = filename
        self.hdf5file = h5py.File(filename, "r")




        
    def __del__(self):

        """Destructor.
        """

        pass




    
    def reload(self):

        """Close the file, and reload it.

        Useful when the file may have changed meanwhile.
        """

        self.hdf5file.close()
        self.hdf5file = h5py.File(self.filename, "r")





    def checkPhotometry(self):

        """Check that photometry has been saved.
        """

        if "Photometry" not in self.hdf5file["/"].keys():
            print("ERROR: No photometry present in the HDF5 file!")




    def getVersion(self):

        """Check that photometry has been saved.
        """

        application = self.hdf5file['Version'].attrs['Application']
        gitVersion  = self.hdf5file['Version'].attrs['GitVersion']

        return application, gitVersion




    
    #--------------------------------------------------------------#
    #                       INPUT PARAMETERS                       #
    #--------------------------------------------------------------#


    def getInputParameter(self, groupName, parameterName):

        """Get an input parameters that was read and copied from the YAML file.
        
        Parameters
        ----------
        groupName : str
            A parameter group as present in the Yaml input file (e.g. "Platform")
        parameterName : str
            Name of the input parameter as present in the YAML file (e.g. "CycleTime")

        Return
        ------
        Requested input paramter(s) (str, int, float, ndarray)
        """

        return self.hdf5file["InputParameters/" + groupName].attrs[parameterName]





    def getTime(self, df=False):

        """Get the time points according to the CCD cadence [s].
        """

        cadence       = self.getInputParameter("ObservingParameters", "CycleTime")
        numExposures  = self.getInputParameter("ObservingParameters", "NumExposures")
        beginExposure = self.getInputParameter("ObservingParameters", "BeginExposureNr")
        ccdCode       = self.getInputParameter("CCD", "Position")

        # TODO fix this in future!
        if ccdCode == "Custom": 
            ccdCode = "1"
        timeShift = self.getInputParameter("CCDPositions", "TimeShift")[int(ccdCode)-1]
        timeArray = np.arange(beginExposure, beginExposure+numExposures) * cadence+timeShift

        # Return time points

        if df:
            return pd.DataFrame({"time": timeArray})
        else:
            return timeArray

    



    def getExposureTime(self):

        """Get the exposure time.

        This function calculation the exposure time which is defined as the cadence
        minus the readout time before the next exposure starts [s]
        """

        cycleTime = self.getInputParameter("ObservingParameters", "CycleTime")
        readoutTimeBeforeNextExposure = self.getReadoutTime()[0]

        return cycleTime - readoutTimeBeforeNextExposure





    def getTimeOffset(self):

        """Get time offset value of the F-CAMs.
        """

        return np.array(self.hdf5file["/Offset/OffsetValueFastCadence"])





    def getReadoutTime(self):

        """Get CCD readout times.

        Return
        ------
        - Readout time before the next exposure starts [s]
        - Readout time during the next exposure [s]
        """

        cameraGroup = self.getInputParameter("Telescope", "GroupID")
        ccdPosition = self.getInputParameter("CCD", "Position")

        # Fetch information about the subfield CCD position
        
        if ccdPosition == "Custom":
            numRows         = self.getInputParameter("CCD", "NumRows")
            numColumns      = self.getInputParameter("CCD", "NumColumns")
            firstRowExposed = self.getInputParameter("CCD", "FirstRowExposed")

        else:
            if cameraGroup == 'Fast':
                index = int(ccdPosition[0]) - 1
                firstRowExposed = self.getInputParameter("CCDPositions",
                                                         "FirstRowForFastCamera")[index]
            else:
                index = int(ccdPosition) - 1
                firstRowExposed = self.getInputParameter("CCDPositions",
                                                         "FirstRowForNormalCamera")[index]
                
            numRows    = self.getInputParameter("CCDPositions", "NumRows")[index]
            numColumns = self.getInputParameter("CCDPositions", "NumColumns")[index]

        readoutMode = self.getInputParameter("CCD/ReadoutMode", "ReadoutMode")

        # Unit conversion for serial tranfer    [ns -> s]
        # Unit conversion for parallel transfer [µs -> s]
        
        serialTransferTime   = self.getInputParameter("CCD", "SerialTransferTime")   * 1e-9
        parallelTransferTime = self.getInputParameter("CCD", "ParallelTransferTime") * 1e-6
        parallelTransferTimeFast = self.getInputParameter("CCD", "ParallelTransferTimeFast") * 1e-6

        numColumnsBiasMap  = self.getInputParameter("SubField", "NumBiasPrescanColumns")
        numBiasPrescanRows = self.getInputParameter("SubField", "NumBiasPrescanRows")
        numRowsSmearingMap = self.getInputParameter("SubField", "NumSmearingOverscanRows")

        # Both detector halves are read out simultaneously -> columns read out by the FEE:
        # - half of the CCD
        # - serial prescan
        # - parallel prescan

        numColumnsReadout = numColumns / 2 + numColumnsBiasMap + numBiasPrescanRows

        # How many rows will be actually read out by the FEE?
        # - Nominal mode: image area + parallel over-scan
        #   Normal camera: image area = whole CCD
        # - Fast camera: image area = lower half of the CCD
        #   Partial readout: configurable
        # The rest of the image area will be dumped

        numRowsReadout = 0
        numRowsDump    = 0
        
        # FAST CAMERA

        if cameraGroup == 'Fast':

            # Move the upper half of the CCD down to the lower half, row-by-row

            numRowsFrameTransfer = numRows - firstRowExposed

            readoutTimeBeforeNextExposure = numRowsFrameTransfer * parallelTransferTimeFast

            # Nominal mode:
            # The actual readout of the lower half of the CCD (after frame transfer)
            # is done while the next exposure has already started

            if readoutMode == "Nominal":

                numRowsReadout = firstRowExposed + numRowsSmearingMap
                numRowsDump = 0

            # Partial mode:
	    # Rows read out by the FEE: rows in the block (other rows in image area are dumped)
	    # Note: no parallel over-scan

            elif readoutMode == "Partial":

                numRowsReadout = self.getInputParameter("CCD/ReadoutMode/Partial",
                                                        "NumRowsReadout")
                numRowsDump = firstRowExposed - numRowsReadout

            readoutTimeDuringNextExposure = (numRowsDump * parallelTransferTimeFast +
                                             numRowsReadout * (parallelTransferTime +
                                                               numColumnsReadout *
                                                               serialTransferTime))

        # NORMAL CAMERA

        else:
            
            # Nominal mode (full-frame readout)

            if readoutMode == "Nominal":

                # Rows read out by the FEE:
                # - Rows of image area
                # - Parallel over-scan

                numRowsReadout = numRows + numRowsSmearingMap

                # No rows dumped

                numRowsDump = 0

            # Partial readout

            elif readoutMode == "Partial":

                # Rows read out by the FEE: rows in the block
                # (other rows in image area are dumped)
                # Note: no parallel over-scan

                numRowsReadout = self.getInputParameter("CCD/ReadoutMode/Partial",
                                                        "NumRowsReadout")
                numRowsDump = numRows - numRowsReadout

            readoutTimeBeforeNextExposure = (numRowsDump * parallelTransferTimeFast +
                                             numRowsReadout * (numColumnsReadout *
                                                               serialTransferTime +
                                                               parallelTransferTime))
            readoutTimeDuringNextExposure = 0

        return readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure





    def getGain(self):

        """Get the gain for both CCD halves.

        NOTE: Left (F) and Right (E) seen from readout register.
        """

        gainCCD_F = self.getInputParameter("CCD/Gain", "RefValueLeft")
        gainFEE_F = self.getInputParameter("FEE/Gain", "RefValueLeft")

        gainCCD_E = self.getInputParameter("CCD/Gain", "RefValueRight")
        gainFEE_E = self.getInputParameter("FEE/Gain", "RefValueRight")

        gain_F = 1. / (gainCCD_F * gainFEE_F)
        gain_E = 1. / (gainCCD_E * gainFEE_E)
        
        return gain_F, gain_E


    

    
    def getReadoutNoise(self):

        """Get the total readout noise (RON) of CCD and FEE [e-/exp/pixel].
        """

        ronCCD = self.getInputParameter("CCD", "ReadoutNoise")
        ronFEE = self.getInputParameter("FEE", "ReadoutNoise")
        
        return np.sqrt(ronCCD**2 + ronFEE**2)



    
    
    #--------------------------------------------------------------#
    #                          IMAGE MAPS                          #
    #--------------------------------------------------------------#


    def getMap(self, imageMap, imageNr=False):

        """Get the a simulated image map from the HDF5 file.

        Parameters
        ----------
        imageMap : str
            Requested image map saved to the HDF5 file.
            Options: PSF

        imageNr : int
            The exposureNr of the map that we are interested in (if sensible).

        Return
        ------
        ndarray : 2D numpy array containing the image map.
        """

        data= {"highResPSF":      "highResPSF",
                "diffusedPSF":     "diffusedPSF",
                "PRNU":            "PRNU",
                "BackgroundMap":   "skyBackground",
                "ThroughputMaps": f"throughputMap",
                "SmearingMaps":   f"smearingMap",
                "BiasMapsLeft":   f"bias",
                "BiasMapsRight":  f"bias",
                "Images":         f"subfield",
                "SubPixelImages": f"images"}

        datasetName = data[imageMap]

        # Correct naming conventions

        if imageMap in ["highResPSF", "diffusedPSF"]: 
            imageMap = "PSF"
        if imageMap in ["PRNU", "IRNU"]:  
            imageMap = "Flatfield"

        # Check if the image is in the file

        if self.hdf5file[imageMap].keys() is None:
            print(f"ERROR: {datasetName} not in HDF5 file")
            return

        # Cases when on or more images are requested

        if imageNr is False:
            
            # Fetch images

            image = self.hdf5file[imageMap][datasetName]
            return image

        else:
            firstNr = self.getInputParameter("ObservingParameters", "BeginExposureNr")
            return self.hdf5file[imageMap][datasetName][imageNr-firstNr,:,:]




        
    def getPSF(self, psfName):

        """Get the PSF from the HDF5 file.

        INPUT: psfName: either "highResPSF" or "diffusedPSF"

        """

        return self.getMap(psfName)





    def getPRNU(self):

        """Get the normalized flat-field map.
        """

        return self.getMap("PRNU")




    
    def getBackground(self):

        """Get the sky background map [photons/pixel/exposure]
        """

        return self.getMap("BackgroundMap", imageNr=False)





    def getSmearingMap(self, imageNr=False):

        """Get the smearing map from the HDF5 file [ADU/exposure].
        """

        return self.getMap("SmearingMaps", imageNr=imageNr)





    def getBiasMapLeft(self, imageNr=False):

        """Get bias map from for the left detector half (F side) [ADU/exp].
        """

        return self.getMap("BiasMapsLeft", imageNr=imageNr)




    
    def getBiasMapRight(self, imageNr=False):

        """Get the bias map of for the right detector half (E side) [ADU/exposure].
        """

        return self.getMap("BiasMapsRight", imageNr=imageNr)





    def getThroughputMap(self, imageNr=False):

        """Get the throughput map (normalised) for the given exposure [0,1].
        """

        return self.getMap("ThroughputMaps", imageNr=imageNr)




    
    def getStraylight(self):

        """Get the straylight.
        """

        return self.hdf5file["Straylight"]["Moon"][:]

    


    
    def getImage(self, imageNr=False):

        """Get the pixel image.
        """

        return self.getMap("Images", imageNr=imageNr)

        


    
    def getSubPixelImage(self, imageNr=False):

        """Get the sub-pixel image.
        """

        return self.getMap("SubPixelImages", imageNr=imageNr)




    
    def getImagette(self, starID, imageNr, radius=2):

        """Get small square imagette around a requested star in a requested image.

        Parameters
        ----------
        starID : int
            Integer star identifier. Equals the line number of the star in the
            input star catalog (counting from 0).
        imageNr : int
            Integer sequential number of the image in the HDF5 file
        radius : int
            Output imagette has size 2*radius+1 pixels in both x and y direction. [pixels]

        Return
        ------
        imagette : ndarray
            Rectangular imagette. If pixel coordinates of the star are x and y,
            then imagette is extracted as: image[x-radius:x+radius+1, y-radius:y+radius+1]
        """

        # Construct the image name that was used to store the image

        imageName = f"image{imageNr:07d}"

        # Check if the image is in the file

        image = self.getMap("Images", imageNr=imageNr)

        # Get a list of the stars visible in the subimage and their coordinates
        # Check if our star is in this list. If not, complain.

        IDs, row, col, Xmm, Ymm, flux = self.getStarCoordinates(imageNr)
        if starID not in IDs:
            print("Error: SimfFile.getImagette(): " +
                  f"star {starID} not in image {imageName}")
            return None

        # Get the pixel coordinates of our star, and use it to construct the imagette borders

        rowStar     = row[IDs == starID][0]
        columnStar  = col[IDs == starID][0]
        rowBegin    = max(0, int(np.round(rowStar - radius)))
        columnBegin = max(0, int(np.round(columnStar - radius)))
        rowEnd      = min(image.shape[0], int(np.round(rowStar + radius + 1)))
        columnEnd   = min(image.shape[1], int(np.round(columnStar + radius + 1)))

        # Extract the imagette

        return image[rowBegin:rowEnd, columnBegin:columnEnd]


    

    
    #--------------------------------------------------------------#
    #                         SAVE PIXEL MAPS                      #
    #--------------------------------------------------------------#


    def saveHighResPSFtoFITS(self, FITSfileName):

        """Save the high resolution PSF to a FITS file.

        In case of the analytic non-Gaussian PSF, a high-resolution PSF corresponding
        to the center of the subfield is stored in the HDF5 file. Extract this image
        and save it to a FITS file.
        """

        imageName = "HighResPSFmapCenterSubfield"
        if imageName not in self.hdf5file["PSF"].keys():
            print("Error: Could not find PSF/{0} in HDF5 file")
            return None
        else:
            dataset = self.hdf5file["PSF"][imageName]
            image = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(image)
            hduList = [fits.PrimaryHDU(image)]
            myFits = fits.HDUList(hduList)
            myFits.writeto(FITSfileName)





    def saveImagesToFITS(self, fileName):

        """Save the image maps to a FITS file.

        Save all subfield images in the HDF5 file to a FITS file with the given file name.
        This will go horribly wrong when the number of images is too large or when the images
        themselves are too large.
        """

        hduList = []
        Nimages = self.getInputParameter("ObservingParameters", "NumExposures")
        for imageNr in range(0, Nimages):
            
            imageName = "image{0:07d}".format(imageNr)
            image = self.getImage(imageNr)
            
            if imageNr == 0:
                hdu = fits.PrimaryHDU(image)
            else:
                hdu = fits.ImageHDU(image, name=imageName)
            hduList.append(hdu)

        myFits = fits.HDUList(hduList)
        myFits.writeto(fileName, overwrite=True)





    def saveSmearingMapsToFITS(self, fileName):

        """Save the smearing maps to a fits file.

        Save all smearing maps in the HDF5 file to a FITS file with the given file name.
        This will go horribly wrong when the number of exposures is too large or when the maps
        themselves are too large.
        """

        hduList = []
        Nimages = self.getInputParameter("ObservingParameters", "NumExposures")
        for imageNr in range(Nimages):
            image = self.getSmearingMap(imageNr)
            imageName = "smearingMap{0:07d}".format(imageNr)
            if imageNr == 0:
                hdu = fits.PrimaryHDU(image)
            else:
                hdu = fits.ImageHDU(image, name=imageName)
            hduList.append(hdu)

        myFits = fits.HDUList(hduList)
        myFits.writeto(fileName)





    def saveBiasMapsLeftToFITS(self, fileName):

        """Save the left bias maps to a FITS file.

        Save all bias maps in the HDF5 file to a FITS file with the given file name.
        This will go horribly wrong when the number of exposures is too large or when the maps
        themselves are too large.
        """

        hduList = []
        Nimages = self.getInputParameter("ObservingParameters", "NumExposures")
        for imageNr in range(Nimages):
            image = self.getBiasMapLeft(imageNr)
            imageName = "biasMap{0:07d}".format(imageNr)
            if imageNr == 0:
                hdu = fits.PrimaryHDU(image)
            else:
                hdu = fits.ImageHDU(image, name=imageName)
            hduList.append(hdu)

        myFits = fits.HDUList(hduList)
        myFits.writeto(fileName)





    def saveBiasMapRightToFITS(self, fileName):

        """Save the right bias maps to a FITS file.

        Save all bias maps in the HDF5 file to a FITS file with the given file name.
        This will go horribly wrong when the number of exposures is too large or when the maps
        themselves are too large.
        """

        hduList = []
        Nimages = self.getInputParameter("ObservingParameters", "NumExposures")
        for imageNr in range(Nimages):
            image = self.getBiasMapRight(imageNr)
            imageName = "biasMap{0:07d}".format(imageNr)
            if imageNr == 0:
                hdu = fits.PrimaryHDU(image)
            else:
                hdu = fits.ImageHDU(image, name=imageName)
            hduList.append(hdu)

        myFits = fits.HDUList(hduList)
        myFits.writeto(fileName)




        
    #--------------------------------------------------------------#
    #                     PLATFORM & TELESCOPE                     #
    #--------------------------------------------------------------#


    def getPayloadInfo(self, groupName, subGroup, getTime, df):

        """Get all columns saved to the telescope entry.
        
        The telescope information contains information about the telescope 
        jitter and pointing. This implies: time, (yaw, pitch, roll) angles,
        and (RA, Dec) of the telescope pointing. The cadence is equal to the
        time scale set in the YAML configuration file.

        Parameters
        ----------
        getTime : bool
            If "True" also the time points are returned

        Returns
        -------
        time : ndarray
            Time point [s]
        yaw : ndarray
            Yaw of the telescope pointing [arcsec]
        pitch : ndarray
            Pitch of the telescope pointing [arcsec]
        roll : ndarray
            Roll of the telescope pointing [arcsec]
        alpha : ndarray
            Right acsension of the telescope pointing [deg]
        delta : ndarray
            Declination of the telescope pointing [deg]
        """

        # TODO change code to be exact the sam entries!

        if groupName == 'ACS':
            stringRA    = 'platformRA'
            stringDec   = 'platformDec'
            stringYaw   = 'yaw'
            stringPitch = 'pitch'
            stringRoll  = 'roll'
        elif groupName == 'Telescope':
            stringRA    = 'telescopeRA'
            stringDec   = 'telescopeDec'
            stringYaw   = 'telescopeYaw'
            stringPitch = 'telescopePitch'
            stringRoll  = 'telescopeRoll'

        # Extract arrays

        if getTime:
            time  = self.hdf5file[groupName]["time"][:]

        if subGroup in ['info', 'pointing']:
            alpha = self.hdf5file[groupName][stringRA][:]
            delta = self.hdf5file[groupName][stringDec][:]
            
        if subGroup in ['info', 'angles']:
            pitch = self.hdf5file[groupName][stringYaw][:]
            roll  = self.hdf5file[groupName][stringPitch][:]
            yaw   = self.hdf5file[groupName][stringRoll][:]
            
        # Return requested columns

        if subGroup == 'info':
            if df:
                return pd.DataFrame({'time':time, 'alpha':alpha, 'delta':delta,
                                     'yaw':yaw, 'pitch':pitch, 'roll':roll})
            else:
                return time, alpha, delta, yaw, pitch, roll

        elif subGroup == 'pointing':
            if getTime:
                if df:
                    return pd.DataFrame({'time':time, 'alpha':alpha, 'delta':delta})
                else:
                    return time, alpha, delta
            else:
                if df:
                    return pd.DataFrame({'alpha':alpha, 'delta':delta})
                else:
                    return alpha, delta
            
        elif subGroup == 'angles':
            if getTime:
                if df:
                    return pd.DataFrame({'time':time, 'yaw':yaw, 'pitch':pitch, 'roll':roll})
                else:
                    return time, yaw, pitch, roll
            else:
                if df:
                    return pd.DataFrame({'yaw':yaw, 'pitch':pitch, 'roll':roll})
                else:
                    return yaw, pitch, roll

            


                
    def getPlatformInfo(self, df=False):

        """Get platform information.

        This function use the general 'getPayloadInfo' function to fetch
        the information contained in the 'ACS' HDF5 group.
        """

        return self.getPayloadInfo("ACS", "info", True, df)





    def getPlatformPointingCoordinates(self, getTime=False, df=False):

        """Get platform pointing coordinates.

        This function use the general 'getPayloadInfo' function to fetch
        the pointing coordinates contained in the 'Platform' HDF5 group.
        """

        return self.getPayloadInfo("ACS", "pointing", getTime, df)





    def getPlatformYawPitchRoll(self, getTime=False, df=False):

        """Get platform jitter information.

        This function use the general 'getPayloadInfo' function to fetch
        the jitter information contained in the 'Platform' HDF5 group.
        """

        return self.getPayloadInfo("ACS", "angles", getTime, df)





    def getTelescopeInfo(self, df=False):

        """Get telescope information.

        This function use the general 'getPayloadInfo' function to fetch
        the information contained in the 'Telescope' HDF5 group.
        """

        return self.getPayloadInfo("Telescope", "info", True, df)





    def getTelescopePointingCoordinates(self, getTime=False, df=False):

        """Get telescope pointing coordinates.

        This function use the general 'getPayloadInfo' function to fetch
        the information contained in the 'Telescope' HDF5 group.
        """

        return self.getPayloadInfo("Telescope", "pointing", getTime, df)





    def getTelescopeYawPitchRoll(self, getTime=False, df=False):

        """Get telescope drift information.

        This function use the general 'getPayloadInfo' function to fetch
        the pointing information contained in the 'Telescope' HDF5 group.
        """

        return self.getPayloadInfo("Telescope", "angles", getTime, df)
    



        
    #--------------------------------------------------------------#
    #                       STELLAR INFORMATION                    #
    #--------------------------------------------------------------#


    def getStarCatalog(self, df=False):

        """Get the stellar catalogue.

        Retrieve a catalog of stars that were observed during the time
        series. Most stars are observed during every exposure, but at
        the edge of the subfield some stars may be observed during a
        few exposures only, depending on the ACS. The source of the
        information is the Star Input Catalog that the user specified
        before running the simulation.

        The initial planar focal plane and pixel coordinates are the
        coordinates of the stars on the CCD before any Jitter or thermal
        distortions took place. Field distortion is however taken into
        account when requested by the user in the YAML input file.

        Parameters
        ----------
        df : bool
            Flag to save stellar catalogue to a pandas data frame

        Returns
        -------
        starIDs : ndarray
            Sequential number of those stars in the input catalog that
            were detected on the subfield in one or more exposures.
        RA : ndarray
            Right ascension of stars [deg]
        dec : ndarray
            Declination of stars [deg]
        mag : ndarray
            V magnitude of stars
        xFPmm : ndarray
            Initial planar X focal plane coordinates of the stars [mm]
        yFPmm: ndarray
            Initial planar Y focal plane coordinates of the stars [mm]
        rowPix : ndarray
            Initial CCD (not subfield) X pixel coordinates of the stars [pix]
        colPix : ndarray 
            Initial CCD (not subfield) Y pixel coordinates of the stars [pix]
        """

        # First some checks. Failure will lead to meaningful error messages

        if "starIDs" not in self.hdf5file["StarCatalog"].keys():
            print("Error: SimfFile.getStarCatalog(): not starIDs found in hdf5 file")
            return None, None, None, None, None, None, None, None

        elif "RA" not in self.hdf5file["StarCatalog"].keys():
            print("Error: SimfFile.getStarCatalog(): not RA values found in hdf5 file")
            return None, None, None, None, None, None, None, None

        elif "Dec" not in self.hdf5file["StarCatalog"].keys():
            print("Error: SimfFile.getStarCatalog(): no declination values found in hdf5 file")
            return None, None, None, None, None, None, None, None

        elif "Vmag" not in self.hdf5file["StarCatalog"].keys():
            print("Error: SimfFile.getStarCatalog(): no V magnitudes found in hdf5 file")
            return None, None, None, None, None, None, None, None

        # Extract the data from the HDF5 file

        ID  = self.hdf5file["StarCatalog"]["starIDs"][:]
        ra  = self.hdf5file["StarCatalog"]["RA"][:]
        dec = self.hdf5file["StarCatalog"]["Dec"][:]
        mag = self.hdf5file["StarCatalog"]["Vmag"][:]

        # xFPmm, yFPmm, rowPix, and colPix were all introduced with the same commit
        # So, testing if xFPmm is present in the StarCatalog group is sufficient

        if "xFPmm" in self.hdf5file["StarCatalog"].keys():
            
            xFP  = self.hdf5file["StarCatalog"]["xFPmm"][:]
            yFP  = self.hdf5file["StarCatalog"]["yFPmm"][:]
            xCCD = self.hdf5file["StarCatalog"]["colPix"][:]
            yCCD = self.hdf5file["StarCatalog"]["rowPix"][:]

            if df:
                return pd.DataFrame({'ID':ID, 'ra':ra, 'dec':dec, 'mag':mag,
                                     'xFP':xFP, 'yFP':yFP, 'xPix':xCCD, 'yPix':yCCD})
            else:
                return ID, ra, dec, mag, xFP, yFP, xCCD, yCCD

        else:
            if df:
                return pd.DataFrame({'ID':ID, 'ra':ra, 'dec':dec, 'mag':mag})
            else:
                return ID, ra, dec, mag, None, None, None, None





    def getCoordinates(self, groupName, imageNr, minMag=None, maxMag=None, df=False):

        """General function to get coordinate information.

        This function fetch the star ID, pixel and focal-plane coordinates, and flux
        of all (stars, point-like ghosts, or extended ghosts) in the given image.

        Parameters
        ----------
        groupName: either "StarPositions", "PointLikeGhostPositions", or "ExtendedGhostPositions"  
        imageNr : int 
            Integer sequential number of the image in the HDF5 file.
        minMag : int, float
            Min magnitude of the objects detected on the CCD subfield.
            Only return objects brighter than minMag.
            Should be 'None' if no cut in minimum magnitude should be made.
        maxMag : int, float
            Maximum magnitude of the objects detected on the CCD subfield.
            Only return objects fainter than maxMag.
            Should be 'None' if no cut in maximum magnitude should be made.
        df : bool
            Flag to save stellar catalogue to a pandas data frame

        Return
        ------
        starID : ndarray
            Integer numpy array containing the star IDs of the stars (as mentioned in
            the input catalogue) that cause point-like ghosts in the current image.  
            If no star ID was given in the input star catalogue file, the identifier 
            equals the line number of the star in the input star catalogue (counting from 0). 
        row : ndarray
            Pixel row coordinates of each star in the image (float).
        col : ndarray
            Pixel column coordinates of each star in the image (float).
        Xmm : ndarray
            Focal-plane x-coordinates of each star in the image.
        Ymm : ndarray
            Focal plane y-coordinates of each star in the image.
        flux : ndarray
            Flux of each point-like ghost in the image [photons].

        Remarks
        ------- 
        - The coordinates returned are the time-averaged coordinates of the point-like
          ghosts during the exposure.
        - To get the pixel with the higest flux of star #0, given its (row, col) coordinates:
          >>> im = file.getImage(0)
          >>> ID,row,col,Xmm,Ymm,flux = file.getPointLikeGhostCoordinates(4,minMag=6.0,maxMag=9.0)
          >>> im[int(row[0]), int(col[0])]
        - To use this function to overplot the positions of the point-like ghost on an 
          image plotted by showImage(), use plt.scatter(floor(col), floor(row)) because 
          showImage uses matplotlib.imshow() which switches rows and columns.
        """

        # Check if the point-like ghost info was saved to the HDF5 file

        if groupName not in self.hdf5file["/"].keys():

            print(f"No group '{groupName}' in the HDF5 file.")
            return None, None, None, None, None, None

        # Extract information depending of HDF5 structure
        
        groupByExposure = self.getInputParameter("ControlHDF5Content", "GroupByExposure") 

        if groupByExposure:

            # Construct the exposure name that was used to store the image

            exposureGroupName = "Exposure{0:07d}".format(imageNr)

            # Check if the arrays are in the file:
            # If not: complain, if yes: copy the contents into a numpy array.

            if exposureGroupName not in self.hdf5file[groupName].keys():

                if groupName == "StarPositions":           
                    func = "getStarCoordinates()" 
                if groupName == "PointLikeGhostPositions": 
                    func = "getPointLikeGhostCoordinates()" 
                if groupName == "ExtendedGhostPositions":  
                    func = "getExtendedGhostCoordinates()"
                
                print(f"SimFile.{func}: {exposureGroupName} not in hdf5 file")

                if groupName == "ExtendedGhostPositions":
                    return None, None, None, None, None, None, None
                else:
                    return None, None, None, None, None, None

            else:

                # Extract the arrays from the HDF5 file

                starID = self.hdf5file[groupName][exposureGroupName]["starID"][:]
                row    = self.hdf5file[groupName][exposureGroupName]["rowPix"][:]
                col    = self.hdf5file[groupName][exposureGroupName]["colPix"][:]
                Xmm    = self.hdf5file[groupName][exposureGroupName]["xFPmm"][:]
                Ymm    = self.hdf5file[groupName][exposureGroupName]["yFPmm"][:]
                flux   = self.hdf5file[groupName][exposureGroupName]["flux"][:]
                
                # Make sure that the point-like ghost star IDs are sorted

                sorted = np.argsort(starID)
                starID = starID[sorted]
                row    = row[sorted]
                col    = col[sorted]
                Xmm    = Xmm[sorted]
                Ymm    = Ymm[sorted]
                flux   = flux[sorted]

                # Add radius column for extended ghosts
                
                if groupName == "StarPositions":
                    #print(starID, row, col, Xmm, Ymm, flux)
                    isNotValid = np.all([row != row, col != col, Xmm != Xmm, Ymm != Ymm])
                    if isNotValid:
                        return None, None, None, None, None, None

                if groupName == "ExtendedGhostPositions":
                    
                    radius = self.hdf5file[groupName][exposureGroupName]["radius"]
                    radius = radius[sorted]

        elif not groupByExposure:

            # Or grouped per star which is already sorted

            star = list(self.hdf5file[groupName].keys())
            
            # Check if only a single image is requested and use that automatically

            N = len(self.hdf5file[groupName][star[0]]["rowPix"][:])
            if N == 1: 
                imageNr = 0
            starID = np.array([int(s[-6:]) for s in star])
            row    = np.array([self.hdf5file[groupName][s]["rowPix"][imageNr] for s in star])
            col    = np.array([self.hdf5file[groupName][s]["colPix"][imageNr] for s in star])
            Xmm    = np.array([self.hdf5file[groupName][s]["xFPmm"][imageNr]  for s in star])
            Ymm    = np.array([self.hdf5file[groupName][s]["yFPmm"][imageNr]  for s in star])
            flux   = np.array([self.hdf5file[groupName][s]["flux"][imageNr]   for s in star])

            # Add radius column for extended ghosts

            if groupName == "ExtendedGhostPositions":
                    
                radius = np.array([self.hdf5file[groupName][s]["radius"][imageNr] for s in star])
                
        # If no cut in V magnitude is required, we're finished.

        if (minMag is None) and (maxMag is None):
            
            if groupName == "ExtendedGhostPositions":
                if df:
                    return pd.DataFrame({'ID':starID, 'row':row, 'col':col,
                                         'xFP':Xmm, 'yFP':Ymm, 'flux':flux,
                                         'radius':radius})
                else:
                    return starID, row, col, Xmm, Ymm, flux, radius
            else:
                if df:
                    pd.DataFrame({'ID':starID, 'row':row, 'col':col,
                                  'xFP':Xmm, 'yFP':Ymm, 'flux':flux})
                else:
                    return starID, row, col, Xmm, Ymm, flux
        
        # If a cut in magnitude is required, get the magnitudes from the star input catalogue

        try:
            inputStarIDs, _, _, mag, _, _, _, _ = self.getStarCatalog()
            subFieldMag = mag[np.in1d(inputStarIDs, starID)]

            # If the min or max V magnitude is set to None, use the default values

            if minMag is None:
                minMag = subFieldMag.min()
            if maxMag is None:
                maxMag = subFieldMag.max()

            # Make the magnitude cut

            dex = (subFieldMag >= minMag) & (subFieldMag <= maxMag)

        except:

            # If star position doesn't exist return all
            
            dex = np.arange(starID.shape[0])
                
        # Return after stellar cut

        if groupName == "ExtendedGhostPositions":
            if df:
                return pd.DataFrame({'ID':starID[dex], 'row':row[dex], 'col':col[dex],
                                     'xFP':Xmm[dex], 'yFP':Ymm[dex], 'flux':flux[dex],
                                     'radius':radius[dex]})
            else:
                return (starID[dex], row[dex], col[dex],
                        Xmm[dex], Ymm[dex], flux[dex], radius[dex])
            
        else:
            if df:
                return pd.DataFrame({'ID':starID[dex], 'row':row[dex], 'col':col[dex],
                                     'xFP':Xmm[dex], 'yFP':Ymm[dex], 'flux':flux[dex]})
            else:
                return (starID[dex], row[dex], col[dex], Xmm[dex], Ymm[dex], flux[dex])

    
    

        
    def getStarCoordinates(self, imageNr, minMag=None, maxMag=None, df=False):

        """Get star information.

        This function use the general 'getCoordinates' function to fetch
        the (fractional) pixel coordinates of all stars in the given image.
        See parameters and returns for this function.
        """

        return self.getCoordinates("StarPositions", imageNr, minMag, maxMag, df)



    

    def getPointLikeGhostCoordinates(self, imageNr, minMag=None, maxMag=None, df=False):

        """Get point-like ghost information.
                
        This function use the general 'getCoordinates' function to fetch
        the (fractional) pixel coordinates of all point-like ghosts in the given image.
        See parameters and returns for this function.
        """

        return self.getCoordinates("PointLikeGhostPositions", imageNr, minMag, maxMag, df)




    
    def getExtendedGhostCoordinates(self, imageNr, minMag=None, maxMag=None, df=False):

        """Get point-like ghost information.
                
        This function use the general 'getCoordinates' function to fetch
        the (fractional) pixel coordinates of all point-like ghosts in the given image.
        See parameters and returns for this function.
        """
        
        return self.getCoordinates("ExtendedGhostPositions", imageNr, minMag, maxMag, df)




    
    def getStarPositions(self, starID, getTime=False, df=False):

        """Get stellar pixel positions

        This function fetch the pixel coordinates of a desired 
        star ID from the output file.
        
        Parameters
        ----------
        starID : int 
            Integer sequential number of the star ID.

        Return
        ------
        row : ndarray
            Pixel row coordinates of each star in the image (float).
        col : ndarray
            Pixel column coordinates of each star in the image (float).
        """

        # Get the time column

        if getTime:
            time = self.getTime()

        # Check if the point-like ghost info was saved to the HDF5 file

        groupName = "StarPositions"
        
        if groupName not in self.hdf5file["/"].keys():
            print(f"No group '{groupName}' in the HDF5 file.")
            return None, None

        # Extract information depending of HDF5 structure
        
        groupByExposure = self.getInputParameter("ControlHDF5Content", "GroupByExposure") 
        
        if groupByExposure:

            # Get all Exposure00.. strings and avoid the time array being the last entry
            # TODO the "-1" from Nexp when time column time has its own group!
            
            exp       = np.array(self.hdf5file[groupName])[:]
            Nexp      = len(exp) - 1 
            starIDs   = self.hdf5file[groupName][exp[0]]["starID"][:]
            starIndex = np.where(starIDs == starID)
            star      = starIndex[0]

            # Check if star exist

            if not starIndex:
                print(f"No star with ID '{starID}' in the HDF5 file.")
                return None, None
            
            # Loop over each image to fetch pixel coordinates
            
            row = np.zeros(Nexp)
            col = np.zeros(Nexp)
            for i in range(Nexp):
                row[i] = self.hdf5file[groupName][exp[i]]['rowPix'][star]
                col[i] = self.hdf5file[groupName][exp[i]]['colPix'][star]

        elif not groupByExposure:
            
            star = f"starID{starID:07d}"
            row = self.hdf5file[groupName][star]["rowPix"][:]
            col = self.hdf5file[groupName][star]["colPix"][:]

        # That's it!

        if getTime:
            if df:
                return pd.DataFrame({'time':time, 'row':row, 'col':col})
            else:
                return time, row, col
            
        else:
            if df:
                return pd.DataFrame({'row':row, 'col':col})
            else:
                return row, col




    
    #--------------------------------------------------------------#
    #                        COSMIC PARTICLES                      # 
    #--------------------------------------------------------------#


    def getCosmicsInfo(self, imageNr, field="SubField", df=False):

        """Get information about cosmic rays in the pixel maps.
        
        This function returns for all cosmics that hit the CCD in a given field, 
        the entry rows, the entry columns, the entry angles, the intensities, 
        and the trail lengths.

        Parameters
        ----------
        imageNr : int
            Integer sequential number of the image in the HDF5 file.
        field : str
            String that determines from what field the Cosmics should be returned. 
            Option: ['SubField', 'BiasMapLeft', 'BiasMapRight', 'SmearingMap'].

        Returns
        -------
        entryRows[0..N-1] : ndarray    
            The row where the cosmic hit the CCD [integer pixel]
        entryColumns[0..N-1] : ndarray
            The column where the cosmic hit the CCD [integer pixel]
        entryAngles[0..N-1] : ndarray
            The angle in which the cosmic hit the CCD [rad]
        intensities[0..N-1] : ndarray
            The total number of e- of the cosmic before they were spread out in a trail [e-]
        trailLengths[0..N-1] : ndarray
            The length of the trail caused by the cosmic [pixels]
        
        Notes
        -----
        - If input do not exist or do not match any of these values in the HDF5 file
          the function returns: (None, None, None, None, None).
        - Here N is the total number of cosmics that hit the CCD. If there were no
          hits in the field, the arrays will be empty. The latter is especially likely
          for the bias.
        """

        # Check if the field variables matches the allowed values

        if field not in ["SubField", "BiasMapLeft", "BiasMapRight", "SmearingMap"]:
            print("Error: SimFile.getCosmicsInfo(): " +
                  "field variable doesn't match one of the allowed values: " +
                  "[SubField, BiasMapLeft, BiasMapRight, SmearingMap] ")
            return None, None, None, None, None

        # Check if the Cosmics info was saved to the HDF5 file

        if "Cosmics" not in self.hdf5file["/"].keys():
            print("Error: SimFile.getCosmicsInfo(): " +
                  "no cosmics information was saved in the HDF5 file.")
            return None, None, None, None, None

        # Check if the arrays are in the file:
        # If not: complain
        # If yes: copy the contents into a numpy array

        if field not in self.hdf5file["Cosmics"].keys():
            print(f"Error: SimFile.getCosmicsInfo(): {field} not in hdf5 file")
            return None, None, None, None, None

        # Construct the exposure name that was used to store the image

        exposureGroupName = "Exposure{0:07d}".format(imageNr)
        
        # Extract information depending of HDF5 structure
        
        groupByExposure = self.getInputParameter("ControlHDF5Content", "GroupByExposure") 

        if not groupByExposure:

            # Here cosmics are further grouped per 1000th exposures
            # and we overwrite the "exposureGroupName"

            n = 1000
            imageNrRound = imageNr // n
            exposureGroupName = f"Exposure{imageNrRound:03d}/{exposureGroupName}"

        # Check if this column exists in the HDF5 file

        if exposureGroupName not in self.hdf5file["Cosmics/" + field].keys():
            print(self.hdf5file["Cosmics/" + field].keys())
            print(f"Error: SimFile.getCosmicsInfo(): {exposureGroupName} not in hdf5 file")
            return None, None, None, None, None

        # Extract the arrays from the HDF5 file

        entryRows    = self.hdf5file["Cosmics"][field][exposureGroupName]["entryRows"][:]
        entryColumns = self.hdf5file["Cosmics"][field][exposureGroupName]["entryColumns"][:]
        entryAngles  = self.hdf5file["Cosmics"][field][exposureGroupName]["entryAngles"][:]
        intensities  = self.hdf5file["Cosmics"][field][exposureGroupName]["intensities"][:]
        trailLengths = self.hdf5file["Cosmics"][field][exposureGroupName]["trailLengths"][:]
            
        # That's it!

        if len(intensities) == 1 and intensities[0] == -1.0:
            if df:
                return pd.DataFrame()
            else:
                empty = np.array([])
                return empty, empty, empty, empty, empty
        else:
            if df:
                return pd.DataFrame({'entryRows':entryRows, 'entryColumns':entryColumns,
                                     'entryAngles':entryAngles, 'intensities':intensities,
                                     'trailLengths':trailLengths})
            else:
                return entryRows, entryColumns, entryAngles, intensities, trailLengths


        


    def getCosmicsAffectedPixels(self, imageNr, field="SubField", df=False):

        """Get the pixel affected by cosmic rays.

        Get the pixel coordinates and flux for those pixels that are affected
        by a cosmics, for a particular field (e.g. SubField, SmearingMap, etc).

        Parameters
        ----------
        imageNr : int
            Integer sequential number of the image in the HDF5 file.
        field : str
            String that determines from what field the Cosmics should be returned. 
            Option: ['SubField', 'BiasMapLeft', 'BiasMapRight', 'SmearingMap'].

        Returns
        -------
        row : ndarray
            Pixel row coordinates of each star in the image (integer).
        col : ndarray
            Pixel column coordinates of each star in the image (integer).
        flux : ndarray
            Flux of cosmics [electrons].
        """

        # Check if the field variables matches the allowed values

        if field not in ["SubField", "BiasMapLeft", "BiasMapRight", "SmearingMap"]:
            print("Error: SimFile:getCosmicsAffectedPixels(): " +
                  "field variable doesn't match one of the allowed values: " +
                  "[SubField, BiasMapLeft, BiasMapRight, SmearingMap]")
            return None, None, None

        # Check if the Cosmics info was saved to the HDF5 file

        if "Cosmics" not in self.hdf5file["/"].keys():
            print("Error: SimFile.getCosmicsAffectedPixels(): " +
                  "no cosmics data was saved in the HDF5 file.")
            return None, None, None

        # Check if the arrays are in the HDF5 file
        
        if field not in self.hdf5file["Cosmics"].keys():
            print("Error: SimFile.getCosmicsAffectedPixels(): {field} not in hdf5 file")
            return None, None, None
        
        # Construct the exposure name that was used to store the image

        exposureGroupName = "Exposure{0:07d}".format(imageNr)

        # Extract information depending of HDF5 structure
        
        groupByExposure = self.getInputParameter("ControlHDF5Content", "GroupByExposure") 

        if not groupByExposure:

            # Here cosmics are further grouped per 1000th exposures
            # and we overwrite the "exposureGroupName"
            
            n = 1000
            imageNrRound = imageNr // n
            exposureGroupName = f"Exposure{imageNrRound:03d}/{exposureGroupName}"
            
        # Check if this column exists in the HDF5 file
                
        if exposureGroupName not in self.hdf5file["Cosmics/" + field].keys():
            print(self.hdf5file["Cosmics/" + field].keys())
            print(f"Error: SimFile.getCosmicsAffectedPixels(): {exposureGroupName} " +
                  "not in hdf5 file")
            return None, None, None

        # Extract the arrays from the HDF5 file

        col  = self.hdf5file["Cosmics"][field][exposureGroupName]["columns"][:]
        row  = self.hdf5file["Cosmics"][field][exposureGroupName]["rows"][:]
        flux = self.hdf5file["Cosmics"][field][exposureGroupName]["flux"][:]

        # That's it!

        if df:
            return pd.DataFrame({'row':row, 'col':col, 'flux':flux})
        else:
            return row, col, flux




    
    def getCosmicsAffectedExposures(self, field="SubField"):

        """Get the pixel affected by cosmic rays.

        Get the pixel coordinates and flux for those pixels that are affected
        by a cosmics, for a particular field (e.g. SubField, SmearingMap, etc).

        Parameters
        ----------
        field : str
            String that determines from what field the Cosmics should be returned. 
            Option: ['SubField', 'BiasMapLeft', 'BiasMapRight', 'SmearingMap'].

        Returns
        -------
        Exposure Nr : ndarray
            Indices for exosure affected by cosmic rays.
        """

        # Fetch the image numbers

        # Fetch StarPosition keys (i.e. "Exposure000000", etc)
        # TODO remove time column: -1 because Time is last column
        # Fetch the image names
        imgNames = list(self.hdf5file['Images'].keys())
        N        = len(imgNames)
        
        indices = np.zeros(N)
        
        for i in range(N-1):

            _, _, flux = self.getCosmicsAffectedPixels(i+1, field=field)

            if flux[0] == -1:
                indices[i] = i

        return indices




    
    def getCosmicsWithinApertureMask(self, starID):

        """Get the pixel affected by cosmic rays.

        Get the pixel coordinates and flux for those pixels that are affected
        by a cosmics, for a particular field (e.g. SubField, SmearingMap, etc).

        Parameters
        ----------
        field : str
            String that determines from what field the Cosmics should be returned. 
            Option: ['SubField', 'BiasMapLeft', 'BiasMapRight', 'SmearingMap'].

        Returns
        -------
        Exposure Nr : ndarray
            Indices for exosure affected by cosmic rays.
        """

        # Fetch StarPosition keys (i.e. "Exposure000000", etc)

        imgNames = list(self.hdf5file['Images'].keys())
        N        = len(imgNames)
    
        # Check photometry is present for each star
        
        starIDgroupName = f"starID{starID}"
        if starIDgroupName not in self.hdf5file["Photometry"]["Lightcurves"].keys():
            print(f"ERROR: simfile.getCosmicsWithinApertureMask(): {starIDgroupName} " +
                  "not present in HDF5 group: Photometry/Lightcurves")
        else:
            updates = self.getMaskUpdateEvents()

        # For the exposures with cosmics fetch the aperture mask(s)

        dex  = []
        for i in range(N-1):

            # Get info about cosmics for entire image
            
            row_cos, col_cos, flux_cos = self.getCosmicsAffectedPixels(i+1, field='SubField')
            
            if flux_cos[0] != -1:

                # Get the aperture mask pixels
                
                row_mask, col_mask, _, _, _, _ = self.getApertureMask(starID=starID, imageNr=i)

                # Compare mask pixels to cosmics affected pixels element wise

                try: 
                    (row_cos == row_mask).all()
                except: 
                    rows = True
                else: 
                    rows = False

                try: 
                    (col_cos == col_mask).all()
                except: 
                    cols = True
                else: 
                    cols = False
                
                if rows and cols:
                    dex.append(i)

        return dex
                



    
    #--------------------------------------------------------------#
    #                    PHOTOMETRIC FUNCTIONS                     #
    #--------------------------------------------------------------#

           
    def getFlux(self, starID, fluxType="estimated", df=False):

        """Returns flux points in units [e-/s].
        """

        # Select the proper flux name

        if   fluxType == "estimated": 
            lctype = "estimatedFlux"
        elif fluxType == "input":     
            lctype = "inputFlux"
        else:
            print("ERROR: SimFile.getFlux(): fluxType can only be 'estimated' or 'input'")
            return None
        
        # Query either a single star or multiple stars as requested

        try: 
            len(starID)
        except:
            starID = np.array([starID])
            names = False
        else:
             names = True

        # Add a flux column(s) for the star(s)

        for ID in starID:

            # Check photometry is present for each star
            
            starIDgroupName = f"starID{ID}"
            if starIDgroupName not in self.hdf5file["Photometry"]["Lightcurves"].keys():
                print(f"ERROR: SimFile.getFlux(): {starIDgroupName} not present in " +
                      "Photometry/Lightcurves/ in the HDF5 file")

            # Select correct name convention
            
            if names: 
                string = f"flux_{ID}"
            else:     
                string = "flux"

            # Fetch flux column
            
            flux = np.array(self.hdf5file[f"Photometry/Lightcurves/starID{ID}/{lctype}"])

            # Create data frame and append to it
            
            if ID == starID[0]:
                df0 = pd.DataFrame({string: flux})
            else:
                df0[string] = flux

        # Convert unit [e-/exp] -> [e-/s]
        
        df0 /= self.getExposureTime()
        
        # Finito!

        if df:
            return df0
        else:
            flux = df0.to_numpy().T
            if not names: 
                flux = flux[0]                
            return flux 






    def getLightCurve(self, starID, fluxType='estimated', df=False):

        """Extract the light curve of one or more stars

        This function can be used to extract the light curves for one star
        with the given ID, or for a full list of stars with IDs given by an
        list or array.

        Parameters
        ----------
        starID : int, list/ndarray
            int  : ID of the star as mentioned in the last column of the star catalog
            list : List of star IDs for which the light curve should be extracted
        fluxType : str
            Choose either {estimated, input, both}:
            The estimated one is derived from a binary mask.
            The input is derived from the mean input mag specified in the star catalog
            and (for variable stars) the delta-mag time series given as an input file.

        Return
        ------
        df : pdframe
            time : first column [s]
            flux : consecutive columns [e-/exposure]
        """

        # Fetch time column

        time = self.getTime(df=True)

        # Fetch flux column(s)

        if fluxType == 'both':
            flux       = self.getFlux(starID, fluxType="estimated", df=True)
            flux_input = self.getFlux(starID, fluxType="input",     df=True)
            if df:
                df = pd.concat([time, flux, flux_input], axis=1)
                df.columns = ["time", "flux", "flux_input"]
                return df
            else:
                return time.to_numpy().T, flux.to_numpy().T, flux_input.to_numpy().T
            
        elif fluxType in ("input", "estimated"):
            flux = self.getFlux(starID, fluxType=fluxType, df=True)
            if df:
                df = pd.concat([time, flux], axis=1)
                df.columns = ["time", "flux"]
                return df
            else:
                return time.to_numpy().T, flux.to_numpy().T
        else:
            print("ERROR: no such flux name, use either 'estimated' or 'input'!")





    def getMaskUpdateEvents(self):

        """Exposure number of all mask updates.
        """

        # Fetch mask update events
        
        return np.array(self.hdf5file["Photometry/Masks/exposureNrOfMaskUpdate"])





    def getApertureMask(self, starID, imageNr=None):

        """Fetch all information about the photometric aperture mask used.

        This function returns the subfield row and column indices of the 
        mask that is used to extract the flux of star with the given ID for
        the given exposure number. This only makes sense if the photometry 
        was activated in the configuration yaml file.

        Parameters
        ----------
        starID : int
            ID of the star as mentioned in the last column of the star catalog
        imageNr : int 
            Integer sequential number of the image in the HDF5 file

        Returns
        -------
        rowIndices : ndarray
            Subfiled row indices for each of the mask pixels
        colIndices : ndarray
            Subfield column indices for each of the mask pixels
        exposureNr : ndarray
            The image number in which the mask was derived:
            exposureNr <= imageNr
        maskSize: ndarray
            The number of pixels a mask contains 
        maskNSR: ndarray 
            The Noise-to-Signal ratio of the flux. Noise coming from 
            target + contaminants + sky + instrument. Signal coming 
            from the target.
        maskSPR: ndarray 
            Stellar pollution ratio. The ratio of the flux inside the
            mask coming from contaminants and the flux coming from 
            target + contaminants + sky. A number between 0 and 1.

        Notes
        -----
        Masks are not update continuously, but only once in a while. This 
        function searches for the most recent mask. This mask may have thus
        been derived from a previous image rather than from the given image Nr.
        """
        
        # Check if photometric data exists
        
        starIDgroupName = "starID{0}".format(starID)
        if starIDgroupName not in self.hdf5file["Photometry"]["Masks"].keys():
            print(f"Error: getPhotometricMask(): {starIDgroupName}" +
                  " not present in Photometry/Masks/ in the HDF5 file")
            return None, None, None, None, None, None

        # Fetch mask info and mask updates
        
        mask = self.hdf5file["Photometry"]["Masks"]
        exposureNrOfMaskUpdate = np.array(mask["exposureNrOfMaskUpdate"])
        numMaskUpdates = len(exposureNrOfMaskUpdate)

        # If a specific image number for the mask update is requested:
        # NOTE: masks are not updated for every exposure hence find most recent mask

        if isinstance(imageNr, int):                                # imageNr is not None

            idx = np.searchsorted(exposureNrOfMaskUpdate, imageNr, side='right') - 1
            if idx < 0:
                print("Error: getPhotometricMask(): requesting an imageNr that is too early for this HDF5 file")
                return None, None, None, None, None, None

            exposureNrOfMaskUpdate = exposureNrOfMaskUpdate[idx]

            # Extact mask size and NSR

            maskSize = np.array(mask[starIDgroupName]['maskSize'])[idx]
            maskNSR  = np.array(mask[starIDgroupName]['maskNSR'])[idx]
            maskSPR = np.array(mask[starIDgroupName]['maskSPR'])[idx]

            # Extract the indices of the proper mask

            exposureGroupName = "Exposure{0:07d}".format(exposureNrOfMaskUpdate)
            rowIndices = mask[starIDgroupName][exposureGroupName]["maskRowIndices"]
            colIndices = mask[starIDgroupName][exposureGroupName]["maskColumnIndices"]

        # Else fetch all the indices for all mask updates

        else:

            # Extact mask size and NSR

            maskSize = np.array(mask[starIDgroupName]['maskSize'])
            maskNSR  = np.array(mask[starIDgroupName]['maskNSR'])
            maskSPR = np.array(mask[starIDgroupName]['maskSPR'])

            # Extract the indices of all masks

            rowIndices = []
            colIndices = []
            for i in range(numMaskUpdates):
                exposureGroupName = "Exposure{0:07d}".format(exposureNrOfMaskUpdate[i])
                rowIndices.append(mask[starIDgroupName][exposureGroupName]["maskRowIndices"])
                colIndices.append(mask[starIDgroupName][exposureGroupName]["maskColumnIndices"])

        # Finito!

        return rowIndices, colIndices, exposureNrOfMaskUpdate, maskSize, maskNSR, maskSPR





    def getTimeQuarter(self, quarterNo):

        """Function to create and return quarter-long time column.

        This function creates a time array for a quarter long simulation. It uses
        the quarter number and the hdf5 output file to correct for the time shift
        of the parent CCD.

        Parameters
        ----------
        outputFile : str
           Full path to the HDF5 outputfile.
        quarterNo : int
           Integer specifying which quarter the time array should be created for.

        Return
        ------
        time : ndarray
            Time array matching the parent quarter defined by the user.
        """

        # Fetch times

        numExposures = self.hdf5file['InputParameters/ObservingParameters'].attrs['NumExposures']
        timeStep     = self.hdf5file['InputParameters/ObservingParameters'].attrs['CycleTime']
        timeShift    = self.hdf5file['InputParameters/CCD'].attrs['TimeShift']

        # Create time array from start and end times

        timeStart = (quarterNo-1) * 90. * 86400. + float(timeShift)
        timeEnd   = timeStart + timeStep * numExposures
        time      = np.arange(timeStart, timeEnd, timeStep)

        return time




    
    #--------------------------------------------------------------#
    #                         PLOT FUNCTIONS                       #
    #--------------------------------------------------------------#


    def showMap(self, pixelMap, clabel=False, figsize=(6,5)):

        """Make a plot any pixel map.
        
        Parameters
        ----------
        pixelMap : ndarray
            Any pixel map from the HDF5 output file
        clabel : str
            String to change the colour bar label
        figsize : mpl object
            Figure size of plot
        """

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Axis and plot
        
        im = ax.imshow(pixelMap, interpolation='nearest', origin='lower', cmap="cubehelix")

        # Colorbar
        
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes('right', size='5%', pad=0.05)
        cbar    = fig.colorbar(im, cax=cax, orientation='vertical')
        
        # Labels
        
        ax.set_xlabel(r"Pixel column, $i$")
        ax.set_ylabel(r"Pixel row, $j$")
        if clabel:
            cbar.ax.set_ylabel(clabel)
        else:
            cbar.ax.set_ylabel('Counts [ADU]')
            
        # Settings
        
        plt.tight_layout()

        # Returns

        return fig, ax





    def showImage(self, imageNr=False, imgScale="percentile", clip=5.0,
                  showStarPositions=False, showPointLikeGhostPositions=False,
                  minMag=None, maxMag=None, showStarIDs=False, count='ADU',
                  tarMarkerSize=150, showMaskOfStarID=None,
                  useTitle=False, showGrid=False,
                  colorBar=True, colorMap="Blues_r",
                  origin="lower", fontSize=15, figsize=(8,8)):

        """Make a plot of the a requested image or the entire image cube in HDF5.

        Parameters
        ----------
        imageNr: False, int
            False : Will plot all images in HDF5 using a slider
            int   : Integer sequential number of the image in the HDF5 file
        imgScale : str
            Different options to select the image scaling:
            percentile : Scale image using a percentile clipping [value, 100-value] 
            auto       : Scale image using a sigma clipping and linear scaling
            minmax     : Scale image linearly        [min, max]
            log        : Scale image logarithmically [min, max]
        clip: int, float
            Scale and to improve the image contrast (see methods above)
        showStarPositions: bool
            False : Default
            True  : Plot the average star positions (averaged over the exposure)
            "PIC" : Differentiate between a target and contaminants (useful for imagettes)
        showPointLikeGhostPositions: bool
            False : Default
            True  : Plot the average pointlike ghost position (averaged over the exposure)
        minMag: int, float
            The minimum V magnitude of the stars/ghosts for which the position should be plotted.
            Only relevant if either showStarPositions or showPointLikeGhostPositions is True.
        maxMag: int, float
            The maximum V magnitude of the stars/ghosts for which the position should be plotted.
            Only relevant if either showStarPositions or showPointLikeGhostPositions is True.
        showStarIDs: bool
            Put small labels with the star IDs next to the star positions
            Will only be executed if showStarPositions=True is set.
        showMaskOfStarID: bool
            Draw rectangles around the pixels of the mask that is used to extract the flux
            value of the star with the given ID. Only works if photometry was activated
            in the yaml inputfile.
        useTitle: bool, str
            False : Default
            True  : Show a default image title of the star ID
            str   : Provide custom title as a string
        colorMap: str
            Option to select your preferred colormap from the matplotlib library.
            Default is the colormap "hot".
        showGrid: bool -> False
            option to select a dim gray grid for a higher visibility of teh pixel grid.
            Will only be executed if showGrid=True is set.

        Return
        ------
        fig, ax : object
            Axes matplotlib.pyplot handle objects to modify plot

        Example
        -------
            >>> simfile = SimFile("Simul01.hdf5")
            >>> simfile.showImage(23)
        """

        # As default, add slider if all images are requested
        # Else get the requested image from the HDF5 file

        if imageNr is False:
            images = self.getImage()

            # Check for a single image
            if len(images.shape) == 2:
                image = images
                Nimg  = 1
                Nrows, Ncols = image.shape
            else:
                image = images[0]
                Nimg  = images.shape[0]
                Nrows, Ncols = image.shape

        else:
            image = self.getImage(imageNr)
            if image is None:
                print(f"ERROR: Image nr {imageNr} does not exist!")
                return
            Nrows, Ncols = image.shape

        # Correct for the orientation
        
        extent = [0, Ncols, 0, Nrows]

        # Normalise if requested
        
        image = np.array(image)
        if count == 'ADU':
            clabel = r'Counts [ADU]'
        elif count == 'kADU':
            clabel = r'Counts [kADU]'
            image  = image / 1000.
        elif count == 'e/s':
            clabel = r'Counts [e$^-$ s$^{-1}$]'
            image  = image / self.getReadoutTime()[0]
        elif count == 'ke/s':
            image = image / self.getReadoutTime()[0] / 1000.
            clabel = r'Counts [ke$^-$ s$^{-1}$]'        
        
        # START PLOTTING

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Add ticks
        
        ax.tick_params(axis='both', which='major', labelsize=fontSize)
        ax.tick_params(axis='both', which='minor', labelsize=fontSize)
        
        # Show image either using clip-procentage or linear scaling if using a colorbar
        # The large dynamic range of the pixel values often results in images where only
        # the brightest stars are visible. To improve the contrast, clip the color mapping.

        image, norm, vmin, vmax = imageClip(image, imgScale, clip)
        
        if imgScale == "percentile":
            imagePlot = ax.imshow(image, cmap=colorMap, interpolation="nearest",
                                    origin=origin, extent=extent, zorder=0)
            imagePlot.set_clim(vmin, vmax)
        else:
            imagePlot = ax.imshow(image, norm=norm, cmap=colorMap, interpolation="nearest",
                                    origin=origin, extent=extent, zorder=0)

        # Colorbar
        
        if colorBar:
            cbar = fig.colorbar(imagePlot, extend=None, shrink=0.84, pad=0.015)
            cbar.set_label(clabel, fontsize=fontSize, labelpad=3)
            cbar.ax.tick_params(labelsize=fontSize)

            # Adjust the colorbar to correct ADU values
            if imgScale == "log":
                cbar.ax.tick_params(which='minor', right=False, labelright=False)
                ticks_loc1  = np.logspace(np.log10(vmin), np.log10(vmax), 6)
                ticks_loc2  = np.logspace(np.log10(image.min()), np.log10(image.max()), 6)
                ticks_label = [f"{i:.2f}" for i in ticks_loc2]
                cbar.locator     = ticker.FixedLocator(ticks_loc1)
                cbar.formatter   = ticker.FixedFormatter(ticks_label)
                cbar.update_ticks()

            # Adjust the colorbar to correct ADU values for auto-scaling
            if imgScale == "auto":
                clabel = 'Normalised counts'
                ticks_label    = [f"{i:.1f}" for i in np.linspace(0, 1, 6)]
                ticks_loc      = np.linspace(vmin, vmax, 6)
                cbar.locator   = ticker.FixedLocator(ticks_loc)
                cbar.formatter = ticker.FixedFormatter(ticks_label)
                cbar.update_ticks()
                
        # If required, overplot a gray semi-transparent grid
        # NOTE: this is only meaningful for smaller imagettes

        if showGrid is True:
            ax.grid(c='gray', ls='-', alpha=0.5, zorder=1)
            
        # Overplot rectangles over those pixels that are part of the mask
        # NOTE: imshow reverses rows and columns

        if showMaskOfStarID is not None:
            rowIndices, colIndices, _, _, _, _ = self.getApertureMask(showMaskOfStarID, imageNr)
            for k in range(len(rowIndices)):
                rect = patches.Rectangle((colIndices[k], rowIndices[k]), 1, 1, linewidth=2.0,
                                         edgecolor='deeppink', facecolor='none', hatch="/",
                                         zorder=2)
                ax.add_patch(rect)
        
        # If required, overplot the true averaged star positions

        if showStarPositions:
            ID, row, col, Xmm, Ymm, flux = self.getStarCoordinates(imageNr,
                                                                   minMag=minMag,
                                                                   maxMag=maxMag)
            # Set linewidth of marker

            lw = 0.055 * fontSize
            
            # Allow differentiating between a target and its contaminants
            
            if showStarPositions == 'PIC':
                # TODO we should use the flux zero point of 20.78?
                mag = -2.5*np.log10(flux) + 25
                ax.scatter(col[0], row[0], s=tarMarkerSize, marker='o', c='lime',
                           edgecolor='k', linewidth=lw, zorder=4)

                # Scale contaminant circle with area
                
                if len(col) > 1:
                    # Scale contaminant circle with area
                    conDeltaMag   = mag[1:] - mag[0]
                    conMarkerSize = tarMarkerSize * (mag[0]/mag[1:])**2
                    ax.scatter(col[1:], row[1:], s=conMarkerSize, marker='o', c='orange',
                               edgecolor='k', linewidth=lw, zorder=4)

                # Add magnitude label above star position
                
                for m,i,j in zip(mag[1:], col[1:], row[1:]):
                    #ax.annotate(f'{m:.1f}', xy=(i-0.25, j+0.25), color='w', weight='bold')
                    ax.text(i-0.27, j+0.25, f'{m:.1f}', color='w', fontsize=14,
                        path_effects=[patheffects.withStroke(linewidth=1,
                                                             foreground='k',
                                                             capstyle="round")])

            # Or hightligth all stars the same
            
            else:
                ax.scatter(col, row, s=int(tarMarkerSize/3), marker='o',
                           facecolors='royalblue', edgecolors='k',
                           linewidth=lw, zorder=4)

            # If requested, add star IDs to plot
            
            if showStarIDs:
                for k in range(len(ID)):
                    label = "{0}".format(ID[k])
                    ax.annotate(label, (col[k], row[k]), fontsize='small',
                                fontweight='extra bold', color="black")

        # If required, overplot the true averaged point-like ghost positions

        if showPointLikeGhostPositions:
            ID, row, col, Xmm, Ymm, flux = self.getPointLikeGhostCoordinates(imageNr,
                                                                             minMag=minMag,
                                                                             maxMag=maxMag)
            ax.scatter(col, row, marker='o', s=6, c='b')
            if showStarIDs:
                for k in range(len(ID)):
                    label = "{0}".format(ID[k])
                    ax.annotate(label, (col[k], row[k]), fontsize='small',
                                fontweight='extra bold', color="black")

        # Ensure that the axes limits are properly set
        
        ax.set_xlim(0, Ncols)
        ax.set_ylim(0, Nrows)
        
        # If required, put the title

        # User defined title-string
        if isinstance(useTitle, str):
            plt.title(useTitle, fontsize=fontSize)
        # Or use default .hdf5 image nr
        elif useTitle is True:
            fileBasename = os.path.splitext(self.filename)[0]
            title = fileBasename + " - image{0:07d}".format(imageNr)
            plt.title(title, fontsize=fontSize)

        # By default, matplotlib only shows the (x,y) coordinates of each pixel,
        # but not the pixel value itself. Change this by redefining the ax.format_coord

        def format_coord(x, y):
            col = int(x)
            row = int(y)
            if col>=0 and col<Ncols and row>=0 and row<Nrows:
                z = image[row,col]
                return "x={:.1f}, y={:.1f}, z={:.1f}".format(x, y, z)
            else:
                return "x={:.1f}, y={:.1f}".format(x, y)
        ax.format_coord = format_coord

        # Show all ticks for smaller subfields or otherwise 10
        
        Nrows, Ncols = Ncols, Nrows
        if Ncols <= 25:
            plt.xticks(np.arange(0, Nrows+1))
            plt.yticks(np.arange(0, Ncols+1))
        elif Ncols > 15 and Ncols <= 100:
            plt.xticks(np.arange(0, Nrows+1, 10))
            plt.yticks(np.arange(0, Ncols+1, 10))
        elif Ncols > 100 and Ncols <= 300:
            plt.xticks(np.arange(0, Nrows+1, 50))
            plt.yticks(np.arange(0, Ncols+1, 50))
        elif Ncols > 300 and Ncols <= 1000:
            plt.xticks(np.arange(0, Nrows+1, 100))
            plt.yticks(np.arange(0, Ncols+1, 100))
        else:
            plt.xticks(np.arange(0, Nrows+1, 500))
            plt.yticks(np.arange(0, Ncols+1, 500))

        # Set labels if requested

        ax.set_xlabel(r"Pixel column, $i$", fontsize=fontSize)
        ax.set_ylabel(r"Pixel row, $j$",    fontsize=fontSize)

        if colorBar:
            if clabel:
                cbar.ax.set_ylabel(clabel, fontsize=fontSize)
            else:
                cbar.ax.set_ylabel('Counts [ADU]', fontsize=fontSize)    

        # Plot with or without a slider

        if imageNr is False and Nimg > 1:
            
            # Function to update slider
            def update_image(n=0):
                image = images[n]
                imagePlot.set_data(image)
                fig.canvas.draw()

            # Create slider
            slider = widgets.IntSlider(description='Image:',
                                       value=0, min=0, max=Nimg-1, step=1,
                                       layout=widgets.Layout(width='50%'))
            widgets.interact(update_image, n=slider)

        else:
            plt.draw()
            plt.show()
            
        # That's it!

        return fig, ax




    def showExtendedImage(self, imageNr=False, imgScale="percentile", clip=15,
                          showStarPositions=False, showPointLikeGhostPositions=False,
                          minMag=None, maxMag=None, showStarIDs=False, count='ADU',
                          tarMarkerSize=150, showMaskOfStarID=None,
                          useTitle=False, showGrid=False,
                          colorBar=True, colorMap="Blues_r",
                          flipAxes=False, ds=None,
                          origin="lower", fontSize=15, figsize=(8,8)):

        """Make a plot of the a requested image or the entire image cube in HDF5.

        Parameters
        ----------
        imageNr: False, int
            False : Will plot all images in HDF5 using a slider
            int   : Integer sequential number of the image in the HDF5 file
        imgScale : str
            Different options to select the image scaling:
            percentile : Scale image using a percentile clipping [value, 100-value] 
            auto       : Scale image using a sigma clipping and linear scaling
            minmax     : Scale image linearly        [min, max]
            log        : Scale image logarithmically [min, max]
        clip: int, float
            Scale and to improve the image contrast (see methods above)
        showStarPositions: bool
            False : Default
            True  : Plot the average star positions (averaged over the exposure)
            "PIC" : Differentiate between a target and contaminants (useful for imagettes)
        showPointLikeGhostPositions: bool
            False : Default
            True  : Plot the average pointlike ghost position (averaged over the exposure)
        minMag: int, float
            The minimum V magnitude of the stars/ghosts for which the position should be plotted.
            Only relevant if either showStarPositions or showPointLikeGhostPositions is True.
        maxMag: int, float
            The maximum V magnitude of the stars/ghosts for which the position should be plotted.
            Only relevant if either showStarPositions or showPointLikeGhostPositions is True.
        showStarIDs: bool
            Put small labels with the star IDs next to the star positions
            Will only be executed if showStarPositions=True is set.
        showMaskOfStarID: bool
            Draw rectangles around the pixels of the mask that is used to extract the flux
            value of the star with the given ID. Only works if photometry was activated
            in the yaml inputfile.
        useTitle: bool, str
            False : Default
            True  : Show a default image title of the star ID
            str   : Provide custom title as a string
        colorMap: str
            Option to select your preferred colormap from the matplotlib library.
            Default is the colormap "hot".
        showGrid: bool -> False
            option to select a dim gray grid for a higher visibility of teh pixel grid.
            Will only be executed if showGrid=True is set.

        Return
        ------
        fig, ax : object
            Axes matplotlib.pyplot handle objects to modify plot

        Example
        -------
            >>> simfile = SimFile("Simul01.hdf5")
            >>> simfile.showImage(23)
        """

        # As default, add slider if all images are requested
        # Else get the requested image from the HDF5 file

        if imageNr is False:
            images = self.getImage()

            # Check for a single image
            if len(images.shape) == 2:
                image = images
                Nimg  = 1
                Nrows, Ncols = image.shape
            else:
                image = images[0]
                Nimg  = images.shape[0]
                Nrows, Ncols = image.shape

        else:
            image = self.getImage(imageNr)
            if image is None:
                print(f"ERROR: Image nr {imageNr} does not exist!")
                return
            # Flip for saturated stars
            if flipAxes:
                image = image.T
            Nrows, Ncols = image.shape

        # Correct for the orientation
        
        extent = [0, Ncols, 0, Nrows]

        # Normalise if requested
        
        image = np.array(image)
        if count == 'ADU':
            clabel = r'Counts [ADU]'
        elif count == 'kADU':
            clabel = r'Counts [kADU]'
            image  = image / 1000.
        elif count == 'e/s':
            clabel = r'Counts [e$^-$ s$^{-1}$]'
            image  = image / self.getReadoutTime()[0]
        elif count == 'ke/s':
            image = image / self.getReadoutTime()[0] / 1000.
            clabel = r'Counts [ke$^-$ s$^{-1}$]'        
        
        # START PLOTTING

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Add ticks
        
        ax.tick_params(axis='both', which='major', labelsize=fontSize)
        ax.tick_params(axis='both', which='minor', labelsize=fontSize)
        
        # Show image either using clip-procentage or linear scaling if using a colorbar
        # The large dynamic range of the pixel values often results in images where only
        # the brightest stars are visible. To improve the contrast, clip the color mapping.

        image, norm, vmin, vmax = imageClip(image, imgScale, clip)
        
        if imgScale == "percentile":
            imagePlot = ax.imshow(image, cmap=colorMap, interpolation="nearest",
                                    origin=origin, extent=extent, zorder=0)
            imagePlot.set_clim(vmin, vmax)
        else:
            imagePlot = ax.imshow(image, norm=norm, cmap=colorMap, interpolation="nearest",
                                    origin=origin, extent=extent, zorder=0)

        # Colorbar
        
        if colorBar:
            cbar = fig.colorbar(imagePlot, extend=None, shrink=0.5, pad=0.015)
            cbar.set_label(clabel, fontsize=fontSize, labelpad=3)
            cbar.ax.tick_params(labelsize=fontSize)

            # Adjust the colorbar to correct ADU values
            if imgScale == "log":
                cbar.ax.tick_params(which='minor', right=False, labelright=False)
                ticks_loc1  = np.logspace(np.log10(vmin), np.log10(vmax), 6)
                ticks_loc2  = np.logspace(np.log10(image.min()), np.log10(image.max()), 6)
                ticks_label = [f"{i:.2f}" for i in ticks_loc2]
                cbar.locator     = ticker.FixedLocator(ticks_loc1)
                cbar.formatter   = ticker.FixedFormatter(ticks_label)
                cbar.update_ticks()

            # Adjust the colorbar to correct ADU values for auto-scaling
            if imgScale == "auto":
                clabel = 'Normalised counts'
                ticks_label    = [f"{i:.1f}" for i in np.linspace(0, 1, 6)]
                ticks_loc      = np.linspace(vmin, vmax, 6)
                cbar.locator   = ticker.FixedLocator(ticks_loc)
                cbar.formatter = ticker.FixedFormatter(ticks_label)
                cbar.update_ticks()
                
        # If required, overplot a gray semi-transparent grid
        # NOTE: this is only meaningful for smaller imagettes

        if showGrid is True:
            ax.grid(c='gray', ls='-', alpha=0.5, zorder=1)
            
        # Overplot rectangles over those pixels that are part of the mask
        # NOTE: imshow reverses rows and columns

        if showMaskOfStarID is not None:
            rowIndices, colIndices, _, _, _, _ = self.getApertureMask(showMaskOfStarID, imageNr)
            # NOTE Flip for saturated stars
            if flipAxes:
                rowIndices, colIndices = colIndices, rowIndices
            for k in range(len(rowIndices)):
                rect = patches.Rectangle((colIndices[k], rowIndices[k]), 1, 1, linewidth=2.0,
                                         edgecolor='deeppink', facecolor='none', hatch="/",
                                         zorder=2)
                ax.add_patch(rect)
        
        # If required, overplot the true averaged star positions

        if showStarPositions:
            ID, row, col, Xmm, Ymm, flux = self.getStarCoordinates(imageNr,
                                                                   minMag=minMag,
                                                                   maxMag=maxMag)
            # Set linewidth of marker
            lw = 0.055 * fontSize
            # NOTE Flip for saturated stars
            if flipAxes:
                row, col = col, row
            # Allow differentiating between a target and its contaminants
            
            if showStarPositions == 'PIC':
                # TODO we should use the flux zero point of 20.78?
                #if mag is not False:
                #mag = -2.5*np.log10(flux) + 25
                ds = ds.loc[ID-1]
                mag = ds.mag.to_numpy()
                ax.scatter(col[0], row[0], s=tarMarkerSize, marker='o', c='lime',
                           edgecolor='k', linewidth=lw, zorder=4)

                # Scale contaminant circle with area
                if len(col) > 1:
                    # Scale contaminant circle with area
                    conDeltaMag   = mag[1:] - mag[0]
                    conMarkerSize = tarMarkerSize * (mag[0]/mag[1:])**2
                    ax.scatter(col[1:], row[1:], s=conMarkerSize, marker='o', c='orange',
                               edgecolor='k', linewidth=lw, zorder=4)

                # Add magnitude label above star position
                ax.text(col[0]-0.30, row[0]+0.35, f'{mag[0]:.1f}', color='w', fontsize=16,
                        path_effects=[patheffects.withStroke(linewidth=1,
                                                             foreground='k',
                                                             capstyle="round")],
                        zorder=10)
                for m,i,j in zip(mag[1:], col[1:], row[1:]):
                    if (m - mag[0]) < 10:
                        ax.text(i-0.27, j+0.32, f'{m:.1f}', color='w', fontsize=14,
                                path_effects=[patheffects.withStroke(linewidth=1,
                                                                     foreground='k',
                                                                     capstyle="round")],
                                zorder=10)

            # Or hightligth all stars the same
            
            else:
                ax.scatter(col, row, s=int(tarMarkerSize/3), marker='o',
                           facecolors='royalblue', edgecolors='k',
                           linewidth=lw, zorder=4)

            # If requested, add star IDs to plot
            
            if showStarIDs:
                for k in range(len(ID)):
                    label = "{0}".format(ID[k])
                    ax.annotate(label, (col[k], row[k]), fontsize='small',
                                fontweight='extra bold', color="black")

        # Ensure that the axes limits are properly set        
        ax.set_xlim(0, Ncols)
        ax.set_ylim(0, Nrows)
        
        # If required, put the title

        # User defined title-string
        if isinstance(useTitle, str):
            plt.title(useTitle, fontsize=fontSize)
        # Or use default .hdf5 image nr
        elif useTitle is True:
            fileBasename = os.path.splitext(self.filename)[0]
            title = fileBasename + " - image{0:07d}".format(imageNr)
            plt.title(title, fontsize=fontSize)

        # By default, matplotlib only shows the (x,y) coordinates of each pixel,
        # but not the pixel value itself. Change this by redefining the ax.format_coord

        def format_coord(x, y):
            col = int(x)
            row = int(y)
            if col>=0 and col<Ncols and row>=0 and row<Nrows:
                z = image[row,col]
                return "x={:.1f}, y={:.1f}, z={:.1f}".format(x, y, z)
            else:
                return "x={:.1f}, y={:.1f}".format(x, y)
        ax.format_coord = format_coord

        # Show all ticks for smaller subfields or otherwise 10
        
        Nrows, Ncols = Ncols, Nrows
        ax.set_xticks(np.arange(0, Nrows+1))
        ax.set_yticks(np.arange(0, Ncols+1))
        xlabel = ax.xaxis.get_ticklabels()
        for label in xlabel:
            label.set_visible(False)
        for label in xlabel[::6]:
            label.set_visible(True)
        ylabel = ax.yaxis.get_ticklabels()
        for label in ylabel:
            label.set_visible(False)
        for label in ylabel[::6]:
            label.set_visible(True)
            
        # Set labels if requested

        ax.set_ylabel(r"Pixel column, $i$", fontsize=fontSize)
        ax.set_xlabel(r"Pixel row, $j$",    fontsize=fontSize)
        
        if colorBar:
            if clabel:
                cbar.ax.set_ylabel(clabel, fontsize=fontSize)
            else:
                cbar.ax.set_ylabel('Counts [ADU]', fontsize=fontSize)    

        # Plot with or without a slider

        if imageNr is False and Nimg > 1:
            
            # Function to update slider
            def update_image(n=0):
                image = images[n]
                imagePlot.set_data(image)
                fig.canvas.draw()

            # Create slider
            slider = widgets.IntSlider(description='Image:',
                                       value=0, min=0, max=Nimg-1, step=1,
                                       layout=widgets.Layout(width='50%'))
            widgets.interact(update_image, n=slider)

        else:
            plt.draw()
            plt.show()
            
        # That's it!
        plt.tight_layout()
        return fig, ax


    
    
        

    def showPixelLevelLightCurve(self, colorMap="cubehelix", figsize=(10,5)):

        """Make a pixel level light curve plot of a smaller subfield.

        Parameters
        ----------
        colorMap: str
            Option to select your preferred colormap from the matplotlib library.

        Return
        ------
        fig, ax : object
            Axes matplotlib.pyplot handle objects to modify plot
        """
        
        # Get all images
        
        ims = self.getImage()
        N, n, m = ims.shape
        
        # Use first exposure for background
        
        im0 = ims[0].flatten()[::-1]
        vran = im0.max()-im0.min()
        
        # Select colorbar for each pixel
        
        cmap   = plt.get_cmap(colorMap, vran)
        colors = cmap(im0.astype(int))

        # Get each pixel light curve
        
        time = self.getTime()
        lcs = []
        for i in zip(range(n)):
            for j in zip(range(m)):
                pixLC = np.array([ims[k][i,j] for k in range(N)])
                lcs.append(pixLC)

        # Reshape array to correct order
        
        x = np.arange(0, n*m).reshape((n,m))
        dex  = np.array([x[i][::-1] for i in range(m)]).flatten()        

        # Create plot
        
        fig = plt.figure(figsize=figsize)

        axes = [plt.subplot(n,m,i+1) for i in range(n*m)]
        for i,ax in zip(dex, axes):

            # Plot image map in background

            ax.set_facecolor(colors[i])

            # Plot light curve ontop

            ax.plot(time, lcs[i], '.', c='orange', ms=0.5, alpha=0.2)
            ax.set_xlim(time.min(), time.max())

            # Remove ticks and labels

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Settings
        
        fig.text(0.5, 0.04, r'Pixel column, $i$', ha='center')
        fig.text(0.08, 0.5, r'Pixel row, $j$',    va='center', rotation='vertical')
        plt.subplots_adjust(wspace=0, hspace=0)

        return fig
    



    
    def showPSF(self, datasetName, rebinToPixels=False,
                normalizeHighestPixelValue=False,
                showPixelGrid=False, colorBar=True, 
                colorMap="gist_stern", useTitle=False, figsize=(7,6)):

        """Plot the requested PSF.

        INPUT: datasetName:                the name of the PSF. Currently either: "highResPSF" or "diffusedPSF"
               rebinToPixels:              True if the subpixel PSF is to be rebinned to pixel level.
                                           Rebinning is done by summing the subpixel values within each pixel.
               normalizeHighestPixelValue: rescale the PSF so that the highest pixel has value 1.0
               useTitle:                   True is a title should be plotted, False otherwise

        OUTPUT: None
        """

        # Get the image from the HDF5 file
        # Flip (left-right) the image, then rotate it 90 degrees. This way the smearing lines
        # are vertical, and the image is oriented in such a way that overplotting the
        # star x,y coordinates from getStarPixelCoordinates() becomes straightforward.


        psf = np.rot90(np.fliplr(self.getPSF(datasetName)))

        # If requested: rebin to pixel level. Recall that the original pixel contains 8 pixels (square)
        # each of 128 subpixels.

        if rebinToPixels:
            psf = psf.reshape(8,128,8,128).sum(axis=3).sum(axis=1)

        # If requested: rescale the PSF so that the highest pixel value is 1.0. This makes the colorbar easier to read.

        if normalizeHighestPixelValue:
           psf /= psf.max()

        # Plot the image.

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)

        Nrows, Ncols = psf.shape
        image = ax.imshow(psf, cmap=colorMap, interpolation="nearest", origin='lower', extent=[0, Nrows, 0, Ncols])

        # If requested, set a default title

        if useTitle:
            fileBasename = os.path.splitext(self.filename)[0]
            title = f"{fileBasename} - {datasetName}"
            plt.title(title)

        # If requested, set colorbar

        if colorBar:
            cbar = plt.colorbar(image, orientation='vertical', extend='max', cmap=colorMap, aspect=15, fraction=0.06)
            cbar.formatter.set_powerlimits((0, 0))
            
        # Labels

        if rebinToPixels:
            ax.set_xlabel(r"$x$ [pixel]")
            ax.set_ylabel(r"$y$ [pixel]")
        else:
            ax.set_xlabel(r"$x$ [subpixel]")
            ax.set_ylabel(r"$y$ [subpixel]")

        # Limits

        ax.set_xlim(0, Nrows)
        ax.set_ylim(0, Ncols)

        # Ticks
        ax.set_xticks(np.linspace(0, Nrows, 9))
        ax.set_yticks(np.linspace(0, Ncols, 9))

        # Set grid

        if showPixelGrid:
            ax.grid(True, which='major', axis='both', ls='-', lw=0.5, color='w')
        
        # That's it!

        return fig, ax





    def plotLightCurve(self, starID, timeUnit=False, fluxUnit=False, figsize=(9,5)):

        """Function plot the simulated light curve.

        Parameters
        ----------
        time_unit : bool, str
            Pass a string with the unit that should be shown for the time axis.
        flux_unit: bool, str
            Pass a string with the unit that should be shown for the flux axis.
        figsize : list
            Matplotlib 'figsize' object to select image size.

        Return
        ------
        fig, ax : objects
            Axes matplotlib.pyplot handle object to be modified by the user.
        """

        # Fetch the light curve

        df = self.getLightCurve(starID, fluxType='estimated', df=True)

        # Change time units

        if timeUnit == 'hour':
            df.time /= 3600.
        elif timeUnit == 'day':
            df.time /= 86400.
        else:
            timeUnit = 's'

        if fluxUnit == 'ppt':
            df.flux = normalize(df.flux, factor=1e3)
        elif fluxUnit == 'ppm':
            df.flux = normalize(df.flux, factor=1e6)
        else:
            fluxUnit = r'e$^-$ s$^{-1}$'
            
        # Create matplotlib object

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot the input variable source

        if "flux_err" in df.columns:
            ax.errorbar(df["time"], df["flux"], yerr=df["flux_err"],
                        fmt=".", color='k', ecolor='darkgray', elinewidth=1,
                        capsize=0, alpha=0.2, label="Raw flux", zorder=1)
        else:
            ax.plot(df.time, df.flux, 'k.', ms=5, alpha=0.2, label="Raw flux", zorder=1)

        # Settings

        ax.set_xlim(df.time.iloc[0], df.time.iloc[-1])
        ax.set_xlabel(f"Time [{timeUnit}]")
        ax.set_ylabel(f"Flux [{fluxUnit}]")
        ax.legend(loc='best')
        plt.tight_layout()
        
        # That's it!

        return fig, ax

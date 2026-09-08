
import h5py
import numpy as np



def createCTIinputFile(outputFileName, beta, temperature, meanTrapDensityBOL, meanTrapDensityEOL, trapCaptureCrossSection, releaseTime, radiationMap):
    """
    PURPOSE: create an HDF5 inputfile that can be used by PlatoSim to include a spatially varying CTI.

    INPUT:
        - outputFileName:           name of the HDF5 file to be written      
        - beta:                     float, beta exponent in Short et al. (2013)
        - temperature:              float, [K]
        - meanTrapDensityBOL:       1D numpy array, containing for each trap species the mean trap density at Beginning-Of-Life. [traps/pix]
                                    The mean is taken over all CCD pixels. 
        - meanTrapDensityEOL:       1D numpy array, containing for each trap species the mean trap density at End-Of-Life. [traps/pix]
        - trapCaptureCrossSection:  1D numpy array, capture cross section for each species [m^2] 
        - releaseTime:              1D numpy array, electron release time for each species [s]
        - radiationMAP:             2D numpy array, same size as the CCD, containing for each pixel the average number of protons per second
                                    it receives throughout the mission. [p+ / s]
                                    The exact units are not relevant as it will be renormalized so that the mean of the entire CCD-sized map is 1.0.

        The lengths of meanTrapDensityBOL, meanTrapDensityEOL, trapCaptureCrossSection, and releaseTime are assumed to be the same and equal to the number of 
        trap species. 

    OUTPUT:
        An HDF5 file will be created. 

    EXAMPLE: to create a CTI that is not spatially varying:

        >>> import numpy as np
        >>> from CTImap import createCTIinputFile
        >>> beta = 0.37
        >>> temperature = 203.0
        >>> meanTrapDensityBOL = np.array([0.0, 0.0, 0.0, 0.0])
        >>> meanTrapDensityEOL = np.array([9.8, 3.31, 1.56, 13.24])
        >>> trapCaptureCrossSection = np.array([2.46e-20, 1.74e-22, 7.05e-23, 2.45e-23])
        >>> releaseTime = np.array([2.37e-4, 2.43e-2, 2.03e-3, 1.40e-1])
        >>> radiationMap = np.ones((4510, 4510))                             # In this example: not spatial variation
        >>> createCTIinputFile("ctiInput.hdf5", beta, temperature, meanTrapDensityBOL, meanTrapDensityEOL, trapCaptureCrossSection, releaseTime, radiationMap)
    """

    # Open the output file. If filename already existed, overwrite the previous one.

    outputfile = h5py.File(outputFileName, 'w')

    # Save the beta exponent and the temperature

    outputfile.create_dataset("beta", data=np.array([beta]))
    outputfile.create_dataset("temperature", data=np.array([temperature]))

    # Save the trap species information

    outputfile.create_dataset("meanTrapDensityBOL", data=meanTrapDensityBOL)
    outputfile.create_dataset("meanTrapDensityEOL", data=meanTrapDensityEOL)
    outputfile.create_dataset("trapCaptureCrossSection", data=trapCaptureCrossSection)
    outputfile.create_dataset("releaseTime", data=releaseTime)

    # Save the radiation map

    outputfile.create_dataset("radiationMap", data=radiationMap)
            
    # That's it!

    outputfile.close()
    return


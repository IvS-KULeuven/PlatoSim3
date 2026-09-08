
import numpy as np
from scipy.special import erf
from math import pi, sin, cos, atan2, sqrt, floor
import cmath
import platosim.referenceFrames as rf



class AnalyticPSF:

    """
    A class to generate an imagette containing the Plato PSFs of stars.
    This class is a Python adaptation of the analytic non-Gaussian PSF that is used in the PlatoSim C++ code.

    EXAMPLE USAGE:

        from analyticpsf import AnalyticPSF
        from matplotlib import pyplot as plt

        # Specify the camera characteristics

        >>> path = "/path/to/apsf_N6000K.txt"
        >>> sigmaPSF = 0.5                    # [pix]
        >>> sigmaDiffusion = 0.2              # [pix]
        >>> focalLength = 247.52              # [mm]
        >>> ccdOrientation = pi/2             # [rad]
        >>> ccdZeroPointX = -1.3              # [mm]
        >>> ccdZeroPointY = 82.48             # [mm]
        >>> pixelSize = 18                    # [micron]
        >>> Nsubpixels = 128                  # (Sqrt of) number of subpixels per pixel

        >>> analyticPSF = AnalyticPSF(path, sigmaPSF, sigmaDiffusion, focalLength, ccdOrientation, ccdZeroPointX, ccdZeroPointY, pixelSize, Nsubpixels)

        # Specify where the subfield is on the CCD, and how large it is.

        >>> zeroPointRow = 3666
        >>> zeroPointCol = 882
        >>> Nrows = 5
        >>> Ncols = 5
        
        # For each star, specify it's (subfield) row, column, and flux. The information can be found in the PlatoSim HDF5 outputfile.

        >>> rowPix = [3.4345825986, 4.4183232293, 1.9170257418]               # [pix]
        >>> colPix = [2.9271507625, 1.21225857782, 4.0628854438]              # [pix]
        >>> flux   = [6993338.0, 1185012.0, 2086817.0]                        # [photons/exposure]

        >>> mypsf  = analyticPSF.getPSF(rowPix[0], colPix[0], flux[0], zeroPointRow, zeroPointCol, Nrows, Ncols)
        >>> for n in range(1, len(flux)):
        ...     mypsf += analyticPSF.getPSF(rowPix[n], colPix[n], flux[n], zeroPointRow, zeroPointCol, Nrows, Ncols)

        

        # Plot the image

        >>> fig = plt.figure(1)
        >>> axis = fig.add_subplot(111)
        >>> imagePlot = axis.imshow(mypsf)
        >>> clipPercentile = 1 
        >>> imagePlot.set_clim(np.percentile(mypsf, clipPercentile), np.percentile(mypsf, 100-clipPercentile)) 

    """




    def __init__(self, parameterFilePath, sigmaPSF, sigmaDiffusion, focalLength, ccdOrientation, 
                       ccdZeroPointX, ccdZeroPointY, pixelSize, Nsubpixels=1):

        """
        Initialisation of the class, using the camera and CCD characteristics. 
        All input parameters have the same meaning as in the PlatoSim input yaml file.

        INPUT:
            parameterFilePath:     Full path to the analytic non-Gaussian PSF fit parameters    
            sigmaPSF:              Width of the analytic PSF (equivalent to sigma of a Gaussian) [pix]
            sigmaDiffusion:        Width of the Gaussian diffusion kernel                        [pix]
                                   Set to zero if no diffusion should be taken into account. 
            focalLength:           focal length of the camera                                    [mm]
            ccdOrientation:        CCD orientation angle                                         [rad]
            ccdZeroPointX:         X Offset of CCD origin from center of focal plane             [mm]  
            ccdZeroPointY:         Y Offset of CCD origin from center of focal plane             [mm]  
            pixelSize:             pixel size (in micron, not mm!)                               [micron] 
            NsubPixels:            Number of subpixels per pixel (e.g. 1, 32, 128, ...)

        OUTPUT:
            None
        """

        # Load the fit parameters describing the non-gaussian PSF

        self.params = np.loadtxt(parameterFilePath)

        # Each PSF is only computed in a limited window around its barycenter coordinates. The size of this window
        # depends on how large the PSF is, which in turns depends on the extend of the PSF and the charge diffusion.

        self.sigmaPSF = sigmaPSF
        self.sigmaDiffusion = sigmaDiffusion

        if sigmaDiffusion > 0.0: 
            sigmaTotal = sqrt(sigmaPSF * sigmaPSF + sigmaDiffusion * sigmaDiffusion)
        else:
            sigmaTotal = sigmaPSF
        
        self.Npixels =  2 * int(8*sigmaTotal + 1) + 1
        self.Nsubpixels = Nsubpixels
        self.size = self.Npixels * self.Nsubpixels

        # Set the Camera and CCD properties

        self.focalLength = focalLength
        self.ccdOrientation = ccdOrientation
        self.ccdZeroPointX = ccdZeroPointX
        self.ccdZeroPointY = ccdZeroPointY
        self.pixelSize = pixelSize

        self.normFactor = 0.0                        # normalization factor









    def addPart(self, partNr, ox, oy, h, sigma, r, rho, phi):

        """
        Add a part to the definite integral of the analytic PSF and adjust the normalization factor accordingly.
       
        INPUT: 
            ox:    relative x offset
            oy:    relative y offset
            h:     relative height
            sigma: width of Gaussian term
            r:     strength and type of periodic term
            rho:   distance between Gaussian term and periodic term
            phi:   orientation of periodic term

        OUTPUT:
            None. Class variables are altered
        """

        ox += (self.size - (self.size & 1)) / 2.
        oy += (self.size - (self.size & 1)) / 2.

        sr = 1. / sqrt(2.) / sigma
        if r != 0:
            fr1 = sqrt(pi * abs(h) * sigma * sigma / 4)
        else:
            fr1 = sqrt(pi * abs(h) * sigma * sigma / 2)

        if h < 0:
            fr2 = -fr1 
        else:
            fr2 = fr1

        if (self.sigmaDiffusion != 0.):
            sr /= sqrt(2. * sr * sr * self.sigmaDiffusion * self.sigmaDiffusion + 1.)

        self.erfxr[partNr][0] = erf(-sr * ox)
        self.erfyr[partNr][0] = erf(-sr * oy)
        for i in range(self.size):
            self.erfxr[partNr][i+1] = erf(sr * (i+1. - ox))
            self.erfxr[partNr][i] = (self.erfxr[partNr][i+1] - self.erfxr[partNr][i]) * fr1
            self.erfyr[partNr][i+1] = erf(sr * (i+1. - oy))
            self.erfyr[partNr][i] = (self.erfyr[partNr][i+1] - self.erfyr[partNr][i]) * fr2
        
        self.normFactor += 4. * fr1 * fr2

        if r != 0:
            delta = 2. * pi * sigma * sigma / r / r
            sc = 1. / sqrt(2.) / sigma * cmath.sqrt(1+delta*1j)
            xc = 1j * pi / abs(r) * sqrt(rho) * cos(phi) / sc
            yc = 1j * pi / abs(r) * sqrt(rho) * sin(phi) / sc
            if r < 0:
                fc = cmath.sqrt(-fr1 * fr2 * cmath.exp(-pi * rho / (1. + delta * delta) * (delta + 1j)) / (1 + 1j * delta))
            else:
                fc = cmath.sqrt(fr1 * fr2 * cmath.exp(-pi * rho / (1. + delta * delta) * (delta + 1j)) / (1 + 1j * delta))

            if self.sigmaDiffusion != 0.:
                dc = cmath.sqrt(2. * sc * sc * self.sigmaDiffusion * self.sigmaDiffusion + 1.)
                sc /= dc
                xc /= dc
                yc /= dc

            self.erfxc[partNr][0] = erf(xc - sc * ox)
            self.erfyc[partNr][0] = erf(yc - sc * oy)

            for i in range(self.size):
                self.erfxc[partNr][i + 1] = erf(xc + sc * (i + 1. - ox))
                self.erfxc[partNr][i] = (self.erfxc[partNr][i + 1] - self.erfxc[partNr][i]) * fc
                self.erfyc[partNr][i + 1] = erf(yc + sc * (i + 1. - oy)) 
                self.erfyc[partNr][i] = (self.erfyc[partNr][i + 1] - self.erfyc[partNr][i]) * fc

            self.normFactor -= 4. * (fc.real * fc.real - fc.imag * fc.imag)

        return










    def integratePSF(self, x, y, rho, phi):
        
        """
        Interpolate and rotate PSF parameters and sum up all parts to calculate the integral over each
        pixel of the analytic PSF.
        
        INPUT:
            x:          distorted x-position (column) of the star in the subfield    [pix]
            y:          distorted y-position (row) of the star in the subfield       [pix]
            rho:        radial distance of the star to the optical axis              [deg]
            phi:        azimuth angle of the star                                    [rad]

        OUTPUT: 
            None. Class variables are altered.
        """

        ox = x - floor(x)
        oy = y - floor(y)
        s = self.sigmaPSF * self.Nsubpixels
        if len(self.params[0]) > 6: 

            # Reset the matrices to build the PSF, and fill them
            
            self.erfxr = np.zeros((2*len(self.params), self.size+1), dtype=np.double)      # evaluated error functions for x
            self.erfyr = np.zeros((2*len(self.params), self.size+1), dtype=np.double)      # evaluated error functions for y
            self.erfxc = np.zeros((2*len(self.params), self.size+1), dtype=np.complex)     # evaluated complex error functions for x
            self.erfyc = np.zeros((2*len(self.params), self.size+1), dtype=np.complex)     # evaluated complex error functions for y
            self.normFactor = 0.0

            rho = rho / 1.4
            c1 = int(min(len(self.params[0]) / 7 - 1, int(rho)) * 7)
            c2 = int(min(len(self.params[0]) / 7 - 1, int(rho) + 1) * 7)
            w = rho - int(rho)
            w = 3. * w * w - 2. * w * w * w

            partNr = 0

            for i in range(len(self.params)): 
                pr = s * ((1. - w) * self.params[i,c1] + w * self.params[i,c2])
                pp =      (1. - w) * self.params[i,c1 + 1] + w * self.params[i,c2 + 1]
                h =       (1. - w) * self.params[i,c1 + 2] + w * self.params[i,c2 + 2]
                b =  s * ((1. - w) * self.params[i,c1 + 3] + w * self.params[i,c2 + 3])
                r =  s * ((1. - w) * self.params[i,c1 + 4] + w * self.params[i,c2 + 4])
                m =       (1. - w) * self.params[i,c1 + 5] + w * self.params[i,c2 + 5]
                a =       (1. - w) * self.params[i,c1 + 6] + w * self.params[i,c2 + 6]

                self.addPart(partNr,   ox + pr * cos(phi + pp), oy + pr * sin(phi + pp), h, b, r, m, phi + a)
                self.addPart(partNr+1, ox + pr * cos(phi - pp), oy + pr * sin(phi - pp), h, b, r, m, phi - a)
                partNr += 2
        else:
            # Reset the matrices to build the PSF, and fill them
            
            self.erfxr = np.zeros((1, self.size+1), dtype=np.double)      # evaluated error functions for x
            self.erfyr = np.zeros((1, self.size+1), dtype=np.double)      # evaluated error functions for y
            self.erfxc = np.zeros((1, self.size+1), dtype=np.complex)     # evaluated complex error functions for x
            self.erfyc = np.zeros((1, self.size+1), dtype=np.complex)     # evaluated complex error functions for y
            self.normFactor = 0.0

            self.addPart(0, ox, oy, 1., s)











    def getPSF(self, row0Star, column0Star, flux, rowZeroPointSubfield, colZeroPointSubfield, NrowSubfield, NcolSubfield):

        """
        Given the pixel coordinates and the brightness of a star, compute an imagette of the subfield with its analytic
        non-Gaussian PSF. 

        INPUT:
            row0Star:             Real-valued distorted row number of the barycenter of the star in the subfield (not CCD)      [pix]
            column0Star:          Real-valued distorted column number of the barycenter of the star in the subfield (not CCD)   [pix]
            flux:                 Total flux of the star during one exposure                                                    [photons]
            rowZeroPointSubfield: CCD row number of pixel (0,0) of the subfield                                                 [pix] 
            colZeroPointSubfield: CCD column number of pixel (0,0) of the subfield                                              [pix]
            NrowSubfield:         Number of rows of the subfield to be generated
            NcolSubfield:         Number of columns of the subfield to be generated

        OUTPUT:
            PSFmap: 2D numpy array of shape (NrowSubfield, NcolSubfield) containing the PSF map

        NOTE:
            rows correspond to the y-coordinate in the Focal Plane reference frame
            columns correspond to the x-coordinate in the Focal Plane reference frame
        """

        # Determine the lower left starting pixel coordinates of the window in the PSF map that will
        # be filled with the PSF

        sx = int(floor(column0Star*self.Nsubpixels - (self.size - 1.) / 2.))
        sy = int(floor(row0Star*self.Nsubpixels - (self.size - 1.) / 2.))

        # Determine the radial distance angle rho of the star to the optical axis, and its azimuth angle phi

        colCCD = column0Star + colZeroPointSubfield
        rowCCD = row0Star + rowZeroPointSubfield
        xFP, yFP = rf.pixelToFocalPlaneCoordinates(colCCD, rowCCD, self.pixelSize, self.ccdZeroPointX, self.ccdZeroPointY, self.ccdOrientation)
        rho = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, self.focalLength))   # [deg]
        phi = atan2(yFP, xFP) - self.ccdOrientation                                              # [rad]

        # Given the position on the CCD, set up the correct PSF

        self.integratePSF(column0Star, row0Star, rho, phi)

        # Build the PSF map using the integrated error functions

        PSFmap = np.zeros((NrowSubfield*self.Nsubpixels, NcolSubfield*self.Nsubpixels))

        for y in range(max(0, sy), min(NrowSubfield*self.Nsubpixels, sy + self.size)):
            for x in range(max(0, sx), min(NcolSubfield*self.Nsubpixels, sx + self.size)):
                
                for k in range(0, min(len(self.erfxr), len(self.erfyr))):
                    PSFmap[y,x] += self.erfxr[k][x-sx] * self.erfyr[k][y-sy]

                for k in range(0, min(len(self.erfxr), len(self.erfyr))):
                    PSFmap[y,x] -= self.erfxc[k][x-sx].real * self.erfyc[k][y-sy].real - self.erfxc[k][x-sx].imag * self.erfyc[k][y-sy].imag

                PSFmap[y,x] *= flux / self.normFactor

        return PSFmap



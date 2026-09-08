import h5py
import os
import platosim.referenceFrames as rf
import numpy                    as np
import pandas as pd

from test                import Test
from math                import degrees, pow, sqrt
from platosim.validation import fitGaussian2D, gaussian2D
from scipy.ndimage       import rotate
import matplotlib.pyplot as plt



"""This test checks checks three things:
1. It checks the simulation with a mapped gaussian PSF model. This is done by generating a theoretical sub pixel
   image and subtracting this from the sub pixel image obtained from the simulator.

2. It check that the charge diffusion spreads the image out. This is done by fitting the sub pixel image of a simulation with a large diffusion to a gaussian function. The test passes if the fitted standard deviation is much higher then the one we would expect from a simulation without charge diffusion.

3. It checks that the jitter smearing spreads the image out. This is done by comparing the pixel image with and without jitter smearing.
"""





class MappedGaussianPSF(Test):

    def setNr(self):

        self.nr = "008.1"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeConvolution"]           = "yes"

        self.sim["ControlHDF5Content/WriteSubPixelImages"] = "yes"
        self.sim["PSF/Model"] = "MappedFromFile"

        self.sim["SubField/NumRows"]    = 100
        self.sim["SubField/NumColumns"] = 100

        self.sim["ControlHDF5Content/WriteHighResolutionPSF"] = "yes"

        # Create a SkyMap with a star that would fall onto that point in the sub field.
        nRows     = self.sim["SubField/NumRows"]
        nColumns  = self.sim["SubField/NumColumns"]

        self.pRow = nRows / 2
        self.pCol = nColumns / 2

        ra, dec = rf.pixelToSkyCoordinates(self.sim, "2", self.pRow, self.pCol)

        ra  = degrees(ra)
        dec = degrees(dec)


        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"
        myFile = open(starCatalogFilename, "w")
        myFile.write("# RA DEC Vmag starID\n")
        myFile.write("{0}  {1}  {2}  {3}\n".format(ra, dec, 16.5, 1))
        myFile.close()

        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename



    def runSimulation(self):

        simFile = self.sim.run(removeOutputFile = True)
        self.image = simFile.getSubPixelImage(0)
        self.psf   = simFile.getPSF("highResPSF")
        self.angle = np.rad2deg(simFile.hdf5file["PSF"].attrs["rotationAngle"])




        self.sim["PSF/MappedFromFile/IncludeChargeDiffusion"] = 'yes'
        self.sim["PSF/MappedFromFile/ChargeDiffusionStrength"] = 2
        simFile = self.sim.run(removeOutputFile = True)
        self.image2 = simFile.getSubPixelImage(0)

        self.sim["PSF/MappedFromFile/IncludeChargeDiffusion"] = 'no'
        self.sim["PSF/MappedFromFile/IncludeJitterSmoothing"] = "yes"
        simFile = self.sim.run(removeOutputFile = True)
        self.image3 = simFile.getSubPixelImage(0)








    def compareMapped(self):
        # Check that the selected PSF is the correct one
        pixelSize = self.sim["CCD/PixelSize"]
        focalLength = self.sim["Camera/FocalLength/ConstantValue"] * 10**3
        xFP, yFP = rf.pixelToFocalPlaneCoordinates(self.pRow, self.pCol, pixelSize, -1.3, 82.48, 3 * np.pi / 2)

        psfFile = self.sim["PSF/MappedFromFile/Filename"]
        psfFile = os.environ["PLATO_PROJECT_HOME"] + "/" +  psfFile

        collectionPSF = self.getPSFFromFile(psfFile, xFP, yFP)

        difference = collectionPSF - rotate(self.psf, angle=self.angle, reshape=False)

        return np.max(difference) < 1 and abs(np.min(difference)) < 1



    def getPSFFromFile(self,fileName, x, y):

        file = h5py.File(fileName, "r")
        df = pd.DataFrame({"length^2" : [], "psfImage" : []})
        
        for number in file.keys():
            if (number == "Coordinates map"):
                continue
            deltaX = float(file[number].attrs["centerCoordinates1"]) - x
            deltaY = float(file[number].attrs["centerCoordinates2"]) - y

            pixelImage = np.zeros(file[number].shape, file[number].dtype)
            file[number].read_direct(pixelImage)
            
            data = pd.DataFrame({"length^2" : [deltaX**2 + deltaY**2], "psfImage" : [pixelImage]})
            df = pd.concat([df,data])

        return df["psfImage"][df["length^2"]==df["length^2"].min()].iat[0]



    def compareDiffusion(self):
        # We check that the image with diffusion strength 2, gives a Gaussian function with a
        # much higher standard deviation then the image without diffusion.

        subPixels = self.sim["SubField/SubPixels"]
        amplitude = np.max(self.image2) - np.min(self.image2)

        sigma0    = self.sim["PSF/MappedFromFile/ChargeDiffusionStrength"] * subPixels
        sigma     = 0.2 * subPixels

        param  = fitGaussian2D(self.image2, amplitude, self.pRow, self.pCol, sigma0, sigma0)
        param2 = fitGaussian2D(self.image, np.max(self.image) - np.min(self.image), self.pRow, self.pCol, sigma, sigma)
 
        return (param[-1] > 7*param2[-1] and param[-2] > param2[-2])



    
    def compareJitterSmooting(self):
        # We checks that the image with jitter smoothing has a higher standard deviation then
        # the image without jitter smoothing.

        subPixels  = self.sim["SubField/SubPixels"]

        amplitude1 = np.max(self.image) -  np.min(self.image)
        amplitude3 = np.max(self.image3) - np.min(self.image3)

        sigma0     = 0.2 * subPixels
        #sigmaJ     = sqrt( pow(sigmaPSFt, 2) + pow( 0.5 / subPixels, 2)) * subPixels

        param  = fitGaussian2D(self.image,  amplitude1, self.pRow, self.pCol, sigma0, sigma0)
        param2 = fitGaussian2D(self.image3, amplitude3, self.pRow, self.pCol, sigma0, sigma0)

        param  = np.array(param[-1:-3:-1])
        param2 = np.array(param2[-1:-3:-1])

        difference = (param2 - param)
        return np.all(difference <= 0)

       

        



    def compare(self):
        test1 = self.compareMapped()
        test2 = self.compareDiffusion()
        test3 = self.compareJitterSmooting()

        return test1 and test2







if __name__ == "__main__":
    t = MappedGaussianPSF()
    print(t.run())

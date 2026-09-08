from test import Test
import numpy as np


class FullWellSaturation(Test):

    def setNr(self):
        self.nr = "019"

    def setAllEffects(self):

        super().setAllEffects()
        self.middleOfRows = 2255
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 4510
        self.sim["SubField/NumColumns"] = 1

        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]
        #wself.sim["PSF/Model"] = "MappedFromFile"
        #self.sim["PSF/MappedFromFile/Filename"] = self.inputDir + "/psf.hdf5"

        starCatalogFilename = self.inputDir + "/starCatalog" + self.nr + ".txt"
        self.sim.createStarCatalogFileFromPixelCoordinates(np.array([self.middleOfRows + 0.5]), np.array([0.5]), np.array([7.5]), np.array([1]), starCatalogFilename)
        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename 


    def runSimulation(self):

        # Column without blooming
        self.sim["CCD/IncludeFullWellSaturation"] = "no"
        simFile = self.sim.run(removeOutputFile = True)
        self.columnWithoutBlooming = simFile.getImage(0)

        # Column with blooming
        self.sim["CCD/IncludeFullWellSaturation"] = "yes"
        simFile = self.sim.run(removeOutputFile = True)
        self.columnWithBlooming = simFile.getImage(0)


    def compare(self):

        saturationLimit = self.sim["CCD/FullWellSaturation"]

        self.columnWithBlooming    = np.ravel(self.columnWithBlooming[self.middleOfRows - 25: self.middleOfRows + 25])
        self.columnWithBlooming    = self.columnWithBlooming - np.min(self.columnWithBlooming)
        
        self.columnWithoutBlooming = np.ravel(self.columnWithoutBlooming[self.middleOfRows - 25: self.middleOfRows + 25])
        self.columnWithoutBlooming = self.columnWithoutBlooming - np.min(self.columnWithoutBlooming)


        #  The total flux with or withouth blooming should be the same.
        condition1 = abs(np.sum(self.columnWithBlooming) - np.sum(self.columnWithoutBlooming)) <= 1.
        

        # The maximum value the immage gets with blooming can not exceed the saturation limit
        condition2 = np.max(self.columnWithBlooming) < saturationLimit

        
        xValue = np.arange(-25, 25)
        normalizationBlooming = (np.sum(xValue) * np.sum(self.columnWithBlooming))
        normalizationWithoutBlooming = (np.sum(xValue) * np.sum(self.columnWithoutBlooming))
        # Both images are centered around the position of the star
        condition3 = int(abs(xValue.dot(self.columnWithBlooming)/normalizationBlooming)) == 0 and int(abs(xValue.dot(self.columnWithoutBlooming)/normalizationWithoutBlooming)) == 0

        
        # The image of the star with blooming is more spread out then the image of the star without blooming
        condition4 = (xValue**2).dot(self.columnWithBlooming) > (xValue**2).dot(self.columnWithoutBlooming)

        return condition1 and condition2 and condition3 and condition4
        

        
    
    








if __name__ == "__main__":

    t = FullWellSaturation()
    print(t.run())

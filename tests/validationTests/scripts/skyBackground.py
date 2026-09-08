import platosim.referenceFrames as rf
import numpy                    as np
import os

from math import degrees
from test import Test




"""This test is designed to check the sky background. The test runs the simulation when no stars are exposed to it. From the output file of the simulation, the theoretically predicted sky background is then subtracted. The resulting image should be filled with values around zero.   """










class SkyBackGround(Test):

    def setNr(self):
        self.nr = "007"


    def setAllEffects(self):

        # Configure the input file and take only one exposure
        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1

        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"
        # Create an empty sky map
        if os.path.isfile(starCatalogFilename):
            os.remove(starCatalogFilename)

        self.sim.createStarCatalogFile(np.array([]), np.array([]), np.array([]), np.array([]), starCatalogFilename)


        # Make sure no star does not fall in the sub field.
        self.sim["CCD/Position"] = "3"




    def compare(self):
        # This checks that the output image is approximately constant and around the expected sky background.

        image = self.simFile.getImage(0)


        exposureTime = self.sim["ObservingParameters/CycleTime"] - self.sim.getReadoutTime()[0]
        theoBackground = self.sim["Sky/SkyBackground/BackgroundValue"] * exposureTime * self.sim["Telescope/TransmissionEfficiency/BOL"]

        difference = image - int(theoBackground)
        RMS = np.sqrt(np.sum(difference*difference) / len(difference))
        return RMS < 0.01


if __name__ == "__main__":

    t = SkyBackGround()
    print(t.run())

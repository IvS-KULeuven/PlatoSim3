from test import Test
import numpy as np
import matplotlib.pyplot as plt

class DigitalSaturation(Test):
    """This test checks that digital saturation is applied. It runs the simulation with and without digital saturation switched on/off. 
    When the simulation runs with saturation the test checks that the maximum value of the output image is equal to the digital saturation limit. 
    Without the simulation, the simulation should fail because the output value is not in the interval [0, 2^16[."""
    
    
    def setNr(self):

        self.nr = "020.4"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] =1
        self.sim["SubField/NumRows"] = 4510
        self.sim["SubField/NumColumns"] = 4510

        self.sim["CCD/IncludeQuantisation"] = "yes"
        self.sim["CCD/IncludeDigitalSaturation"] = "yes"
        self.saturationLimit = self.sim["CCD/DigitalSaturation"]

        # Reset the Camera so that the stars in the input file fall onto the CCD.
        self.sim["CCD/Position"] = "Custom"
        self.sim["Telescope/GroupID"] = "Custom"
        self.sim["Telescope/AzimuthAngle"] = 0.
        self.sim["Telescope/TiltAngle"]    = 0.

        self.sim.createStarCatalogFileFromPixelCoordinates(np.array([2000]), np.array([2000]), np.array([4]), np.array([1]), self.outputDir + "/starCatalog.txt")
        self.sim["ObservingParameters/StarCatalogFile"] = self.outputDir + "/starCatalog.txt"

        

    def runSimulation(self):

        # The simulation is run with digital saturation switched on
        self.simFileWithSaturation = self.sim.run(removeOutputFile = True)

        # We try to run the simulation without digital saturation. This simulation should fail.
        self.sim["CCD/IncludeDigitalSaturation"] = "no"
        try:
            simFileWithoutSaturation = self.sim.run(removeOutputFile = True)
            self.imageWithout = simFileWithoutSaturation.getImage(0)
            self.condition2 = False
        except:
            self.condition2 = True


    def compare(self):

        # check that the max value of the image with saturation is equal to the saturation limit
        imageWithSaturation = self.simFileWithSaturation.getImage(0)
        condition1 = 0 <= abs(np.max(imageWithSaturation) - self.saturationLimit) <= 1
        
        return condition1 and self.condition2






if __name__ == "__main__":
    t = DigitalSaturation()
    print(t.run())
        

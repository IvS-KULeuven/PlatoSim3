import numpy as np
from test import Test
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import math




"""
This test checks the readout noise of Platosim. The test takes one exposure once with readout noise and once without readout noise. The effect of readout noise on the 
subfield is obtained by subtracting these two images. The noise should follow a Normal distribution with mean 0 and variance the sum of the variance of the CCD and FEE. 
The test checks the mean and standard deviation of the noise. 
"""
class ReadoutNoise(Test):

    def setNr(self):

        self.nr = "018"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 1000
        self.sim["SubField/NumColumns"] = 1000

        # Make sure no stars are located on the subfield
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]
        

    def runSimulation(self):

        # Run the simulation with readout noise
        self.sim["CCD/IncludeReadoutNoise"]     = "yes"
        simFileWithNoise    = self.sim.run(removeOutputFile = True)

        # Run the simulation without readout noise
        self.sim["CCD/IncludeReadoutNoise"]     = "no"
        simFileWithoutNoise = self.sim.run(removeOutputFile = True)
        
        self.difference = simFileWithNoise.getImage(0) - simFileWithoutNoise.getImage(0)


    def compare(self):

        theoStd = (np.sqrt(self.sim["FEE/ReadoutNoise"]**2 + self.sim["CCD/ReadoutNoise"]**2))
        return abs(np.mean(self.difference)) / len(self.difference)  < 0.01 and abs(np.std(self.difference) / theoStd - 1) < 0.01




if __name__ == "__main__":
    t = ReadoutNoise()
    print(t.run())

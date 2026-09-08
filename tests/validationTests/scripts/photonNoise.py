import numpy as np
from test import Test
import matplotlib.pyplot as plt







""" 
This test is designed to test the photon noise of PlatoSim. It takes a subfield of one pixel for 300 exposures with no stars on the subfield, so we end up with
a time series of the flux. This time series should follow a Poisson distribution around the value of the sky background. The test passes if the mean value and 
variance of the time series are around the sky background.
"""





class PhotonNoise(Test):

    def setNr(self):
        self.nr = "017"

    def setAllEffects(self):

        super().setAllEffects()
        self.numExposures = 300
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["SubField/NumRows"]    = 100
        self.sim["SubField/NumColumns"] = 100

        # Make sure no stars are on the subfield.
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]

        # Include photon noise.
        self.sim["CCD/IncludePhotonNoise"] = "yes"



    def compare(self):

        # Expected Sky Background
        exposureTime = self.simFile.getInputParameter("ObservingParameters", "CycleTime") - self.sim.getReadoutTime()[0]
        expectedSkyBackground = self.simFile.getInputParameter("Sky/SkyBackground", "BackgroundValue") * exposureTime * self.simFile.getInputParameter("Telescope/TransmissionEfficiency", "BOL")

        # Flux obtained from PlatoSim
        flux = np.array([self.simFile.getImage(exposure) for exposure in range(self.numExposures)])
        
        condition1 = abs((np.mean(flux)) / expectedSkyBackground - 1) < 0.01
        condition2 = abs(np.std(flux) / np.sqrt(expectedSkyBackground) - 1) < 0.01


        return condition1 and condition2

    

if __name__ == "__main__":
    t = PhotonNoise()
    print(t.run())

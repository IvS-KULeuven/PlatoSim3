from test import Test
import numpy as np
import matplotlib.pyplot as plt

class ElectronicOffset(Test):
    """This test is designed to check the offset of PlatoSim. The test checks that for an increasing temperature for the FEE there is a linear increase in electronic offset. 
    This test checks that the fitted slope is equal to the theoretically predicted one from the electronic offset stability.""" 


    
    def setNr(self):
        self.nr = "020.2"


    def setAllEffects(self):

        super().setAllEffects()

        self.numExposures = 100
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["SubField/NumRows"]    = 1
        self.sim["SubField/NumColumns"] = 1

        # Create and link to the temperature file for the temperature variations of the FEE.
        self.deltaTemperature = 10
        cycleTime        = self.sim["ObservingParameters/CycleTime"]
        
        time             = np.array([0, self.numExposures * cycleTime])
        temperature      = np.array([self.sim["FEE/NominalOperatingTemperature"], self.sim["FEE/NominalOperatingTemperature"] + self.deltaTemperature])

        self.sim["FEE/Temperature"] = "FromFile"
        temperatureFileName = self.outputDir + "/temperature.txt"
        self.sim["FEE/TemperatureFileName"] = temperatureFileName
        np.savetxt(temperatureFileName, np.c_[time, temperature])

        # Include Quantisation
        self.sim["CCD/IncludeQuantisation"] = "yes"
        self.sim["FEE/Gain/Stability"] = 0


    def compare(self):

        offset = np.array([self.simFile.getImage(exposure)[0][0] for exposure in range(self.numExposures)])

        # Calculate the expected sky background
        exposureTime = self.simFile.getInputParameter("ObservingParameters", "CycleTime") - self.sim.getReadoutTime()[0]
        backGround = self.simFile.getInputParameter("Sky/SkyBackground", "BackgroundValue")
        transMissionEfficiencyBOL = self.simFile.getInputParameter("Telescope/TransmissionEfficiency", "BOL")
        expectedSkyBackground = backGround * exposureTime * transMissionEfficiencyBOL * self.sim["FEE/Gain/RefValueLeft"] * self.sim["CCD/Gain/RefValueLeft"]

        # Calculate the theorectically expected slope and the fitter slope from PlatoSim
        slope = self.getBestSlope(np.arange(self.numExposures), np.round(offset - expectedSkyBackground))
        theoSlope = self.deltaTemperature / self.numExposures * self.sim["FEE/ElectronicOffset/Stability"]
        
        return abs((slope - theoSlope) / theoSlope) < 0.01




    def getBestSlope(self, x, y):

        # Get the best fit of a linear fit of the data
        return (x - np.mean(x)).dot(y - np.mean(y)) / (x - np.mean(x)).dot(x - np.mean(x))






if __name__ == "__main__":
    t = ElectronicOffset()
    print(t.run())

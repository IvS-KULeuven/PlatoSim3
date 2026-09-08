from test import Test

import numpy as np







"""
This test is designed to test the temperature dependence of the dark noise on the CCD. The test check that for a increasing temperature, the average noise and the variance increase linear with the temperature. The test runs the simulation with 1000 exposures. The noise over time is transformed to a distribution that should have zero mean and a variance of 1. The test passes if the obtained mean and variance are close to this value. 
"""









class TempVariationOfCCD(Test):

    def setNr(self):
        self.nr = "012.3"

    
    def setAllEffects(self):

        super().setAllEffects()

        temp0  = self.sim["CCD/NominalOperatingTemperature"]
        self.deltaT = 1
        self.sim["ObservingParameters/NumExposures"] = 500
        self.sim["CCD/DarkSignal/Stability"] = 5.0
        self.sim["CCD/DarkSignal/DSNU"]      = 0
        self.sim["CCD/Temperature"] = "FromFile"
        temperatureFilename = self.outputDir + "/temperature.txt"
        self.sim["CCD/TemperatureFileName"] = temperatureFilename
        time        = np.array([0, 500 * self.sim["ObservingParameters/CycleTime"]])
        temperature = np.array([ temp0, temp0 + self.deltaT])
        np.savetxt(temperatureFilename, np.c_[time, temperature])


    def runSimulation(self):

        self.sim["CCD/IncludeDarkSignal"] = "yes"
        self.simFile1 = self.sim.run(removeOutputFile = True)

        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeDarkSignal"] = "no"
        self.simFile2 = self.sim.run(removeOutputFile = True)


    def compare(self):

        expected  = self.sim["CCD/DarkSignal/DarkCurrent"] * self.sim["ObservingParameters/CycleTime"]
        stability = self.sim["CCD/DarkSignal/Stability"] * self.sim["ObservingParameters/CycleTime"]
        dark = [self.simFile1.getImage(exp)[0][0] -self.simFile2.getImage(0)[0][0] for exp in range(500)]
        theo = [expected + self.deltaT * stability * exp / 500 for exp in range(500) ]
        difference = [ (image - imageT) / np.sqrt(imageT)  for image, imageT in zip(dark, theo)]
        
        condition1 = abs(np.mean(difference)) < 0.1
        condition2 = 0.8 < np.std(difference) < 1.2

        return condition1 and condition2


if __name__ == "__main__":
    t = TempVariationOfCCD()
    print(t.run())

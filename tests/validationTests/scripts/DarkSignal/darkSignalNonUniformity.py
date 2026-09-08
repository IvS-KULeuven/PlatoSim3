import numpy as np

from test import Test


"""
This test is designed to check the dark signal non-uniformity. The test checks that the dark signal has as mean the
cycle time and the dark current as specified in the input file, and the variance is given by
  (mean * dsnu)**2 + mean
"""












class DarkSignalNonUniformity(Test):

    def setNr(self):
        self.nr = "012.2"

    def setAllEffects(self):

        super().setAllEffects()

        self.numExposures = 500
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["SubField/NumRows"]         = 1
        self.sim["SubField/NumColumns"]      = 1

        self.sim["CCD/DarkSignal/Stability"] = 0
        self.sim["CCD/DarkSignal/DSNU"]      = 15

        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]

    def runSimulation(self):
        self.sim["CCD/IncludeDarkSignal"] = "yes"
        self.simFile1 = self.sim.run(removeOutputFile = True)

        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeDarkSignal"] = "no"
        self.simFile2 = self.sim.run(removeOutputFile = True)



    def compare(self):

        file1 = self.simFile1
        file2 = self.simFile2
        exposures   = self.numExposures
        dsnu        = self.sim["CCD/DarkSignal/DSNU"] / 100.
        darkCurrent = self.sim["CCD/DarkSignal/DarkCurrent"]
        cycleTime   = self.sim["ObservingParameters/CycleTime"]


        dark = np.array([file1.getImage(exp)[0][0] - file2.getImage(0)[0][0] for exp in range(exposures)])

        mean       = np.mean(dark)
        meanT      = darkCurrent * self.sim["ObservingParameters/CycleTime"]
        condition1 = abs(mean - meanT) / meanT < 0.1

        std        = np.std(dark)
        sigma      = np.sqrt((meanT * dsnu)**2 + meanT)
        condition2 = abs(std - sigma) / sigma  < 0.1

        return condition1 and condition2





if __name__ == "__main__":
    t = DarkSignalNonUniformity()
    t.runSimulation()
    print(t.compare())


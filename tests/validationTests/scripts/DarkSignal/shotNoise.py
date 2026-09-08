import numpy             as np

from test import Test







"""
This test checks the behaviour of the shot noise. The simulation is run four times with different Dark Current values. For each run, the obtained dark signal 
and the test checks the mean dark signal and standard deviation. The test passes is the RMS difference between the obtainded and prediced values of the mean 
and standard deviation of the signal is smaller then 0.01.
"""









class ShotNoise(Test):

    def setNr(self):
        self.nr = "012.1"

    def setAllEffects(self):

        super().setAllEffects()

        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]         = 2000
        self.sim["SubField/NumColumns"]      = 2000

        self.sim["CCD/DarkSignal/Stability"] = 0
        self.sim["CCD/DarkSignal/DSNU"]      = 0

        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]

    def runSimulation(self):

        self.simFiles = []
        self.slope    = 1.2
        self.sim["CCD/IncludeDarkSignal"]      = "yes"

        # The simulation is run with Dark Signal switched on for difference values of the Dark Current. This is then compared to a run with the
        # Dark signal switched off. 
        for i in range(1, 5):
            self.sim["CCD/DarkSignal/DarkCurrent"] = self.slope * i
            self.simFiles.append(self.sim.run(removeOutputFile = True).getImage(0))

        self.sim["CCD/IncludeDarkSignal"]    = "no"
        self.withoutDark = self.sim.run(removeOutputFile = True).getImage(0)

    def compare(self):

        meanSlope   = self.sim["ObservingParameters/CycleTime"] * self.slope
        images      = [image - self.withoutDark for image in self.simFiles]
        
        means       = [np.mean(image) for image in images]
        theoMean    = [ meanSlope * n for n in range(1, 5)]
        meanRMS     = np.sqrt(np.sum([ (mean - tMean)**2 / len(means) for mean, tMean in zip(means, theoMean)]))
        condition1  = meanRMS < 0.01

        variations     = [np.std(image) for image in images]
        theoVariations = [np.sqrt(meanSlope * n) for n in range(1, 5)]
        variationRMS   = np.sqrt(np.sum([(var - tVar)**2 / len(variations) for var, tVar in zip(variations, theoVariations)]))
        condition2     = variationRMS < 0.01

        return condition1 and condition2

        


    


if __name__ == "__main__":
    t = ShotNoise()
    print(t.run())


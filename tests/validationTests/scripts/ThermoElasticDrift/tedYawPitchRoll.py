import random
import numpy                      as np
import platosim.referenceFrames   as rf

from test import Test
from math import degrees



"""This test is designed to check weather the thermo-elastic drift behaves as RMS * N(0, 1), where RMS is the standard deviation parameter for the drift. We do this by checking that the standard deviation and the mean of the drift scale linearly with input RMS. The RMS parameters are given from the input file."""










class TedYawPitchRoll(Test):

    def setNr(self):
        self.nr = "005.1"

    def setAllEffects(self):

        #Configure the input file.
        super().setAllEffects()
        self.numExposures                                 = 250
        self.sim["ObservingParameters/NumExposures"]      = self.numExposures
        self.sim["Telescope/UseDrift"]                    = "yes"
        self.sim["Telescope/DriftSource"]                 = "FromRedNoise"
        self.sim["ControlHDF5Content/WriteStarPositions"] = "yes"

        # Pick a random point in the sub field and write it to a star catalog.
        nRows    = self.sim["SubField/NumRows"]
        nColumns = self.sim["SubField/NumColumns"]


        pRows    = random.randint(10, nRows-10)
        pColumns = random.randint(10, nColumns-10)


        # Create a SkyMap with a star that would fall onto that point in the sub field.
        raIn, decIn = rf.pixelToSkyCoordinates(self.sim, "2", pColumns, pRows)

        raIn  = degrees(raIn)
        decIn = degrees(decIn)


        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"
        myFile = open(starCatalogFilename, "w")
        myFile.write("# RA DEC Vmag starID\n")
        myFile.write("{0}  {1}  {2}  {3}\n".format(raIn, decIn, 16.5, 1))
        myFile.close()

        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename



    def runSimulation(self):

        # We run the simulation three times, each time we allow drift in one dimension of rotation
        # while the other angles are kept constant.

        self.axes = ["DriftYawRms", "DriftPitchRms", "DriftRollRms"]

        self.resultMean = []
        self.resultSdf  = []

        # The simulation is then run for multiple RMS values, and we save the (mean drift) / RMS
        # and (standard deviation of drift) / RMS for every run. We would expect this value to be
        # approximately constant, so we check the the variance of these two values is small.

        for i in range(3):
            mean, sdf = zip(*[self.runForRMS(rms, 0) for rms in [10, 20 , 30 , 50, 70]])
            mean = np.array(mean)
            sdf  = np.array(mean)

            self.resultMean.append(abs(mean.var()) < 0.01)
            self.resultSdf.append(sdf.var() < 0.01)







    def runForRMS(self, rms, ax):

        # The simulation is run for a given drift RMS around one given axis = ax.
        # The drift for every exposure is saved in the variable delta and the function returns
        # the mean / RMS and (standard deviation) / RMS of the drift for all the exposures.

        numEx = self.sim["ObservingParameters/NumExposures"]

        self.sim["Telescope/" + self.axes[ax]]   = rms
        simFile = self.sim.run(removeOutputFile = True)
        self.sim["Telescope/" + self.axes[ax]] = 0

        pos = [simFile.getStarCoordinates(exp)[1:3] for exp in range(numEx)]
        row, col = zip(*pos)
        row    = np.concatenate(row, axis = 0)
        col    = np.concatenate(col, axis = 0)

        rad, dec = rf.pixelToSkyCoordinates(self.sim, "2", col, row)

        posRad = [degrees(ra) * 3600 for ra in rad]
        posDec = [degrees(de) * 3600 for de in dec]

        displ  = [abs(dec) + abs(rad) for dec, rad in zip(posDec, posRad)]
        delta  = np.array(displ[0:-1:1]) - np.array(displ[1::])

        return delta.mean() / rms , np.sqrt(delta.var()) / rms




    def compare(self):

        # The test passes if the value of the variance of the mean / RMS and (stand. dev) / RMS
        # are small enough. ==> These values are approximately constant.
        return(all(self.resultMean) and all(self.resultSdf))


if __name__ == "__main__":
    t = TedYawPitchRoll()
    print(t.run())

from test import Test

from matplotlib import pyplot as plt
import platosim.referenceFrames as rf
import numpy as np









class TedOnCCDs(Test):
    """
    This test is designed to test that the jitter that is observed is independent of the CCD. The test simulates one star on the center of a 
    CCD for the different CCDs, and checks that the observed jitter at a certain time is independent of the CCD.
    """

    def setNr(self):
        self.nr = "005.3"

    def setAllEffects(self):

        #Config the input file
        super().setAllEffects()
        numExposures = 40
        self.sim["ObservingParameters/NumExposures"] = numExposures
        self.sim["Telescope/UseDrift"]               = "yes"
        self.sim["ObservingParameters/CycleTime"]    = 25
        self.sim["Telescope/DriftTimeScale"]          = 250




        
    def runSimulation(self):

        self.drift = {}
        self.CCDs = ["1", "2", "3", "4"]
        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"

        # Get the amount of pixels in the subfield
        nRows    = np.array([ self.sim["SubField/NumRows"] / 2])
        nColumns = np.array([ self.sim["SubField/NumColumns"] / 2])
        mag      = np.array([ 16.2 ])
        
        for CCD in self.CCDs:

            # Run the simulation once for avery camera group 
            self.sim["CCD/Position"] = CCD
            self.sim.createStarCatalogFileFromPixelCoordinates(nRows, nColumns, mag, np.array([1]), starCatalogFilename)

            # Get the jitter from the simulation
            simFile = self.sim.run(removeOutputFile = True)
            time, yaw, pitch, roll = simFile.getTelescopeYawPitchRoll(getTime=True)
            self.drift[CCD] = [time, yaw, pitch, roll]




    
    def compare(self):

        # Save a plot of the jitter on the different CCDs
        self.makePlot()

        # Save the time and jitter in these variables
        drift   = []
        
        # Get the common time interval
        clTime  = self.sim["ObservingParameters/CycleTime"]
        minTime = max([ np.min(drift[0]) for drift in self.drift.values()])
        minTime = max(minTime, clTime)
        maxTime = min([ np.max(drift[0]) for drift in self.drift.values()])

        # Store the Yaw, Pitch, Roll  is the repective variables
        for value in self.drift.values():
            
            mask  = np.logical_and(value[0] > minTime, value[0] <= maxTime)
            yaw   = np.array(value[1][mask])
            pitch = np.array(value[2][mask])
            roll  = np.array(value[3][mask])
          
            drift.append([yaw, pitch, roll])

        # Check that the Yaw, Pitch, Roll is the same for all the CCDs
        condition = np.all(np.diff(drift, axis=0) == 0)
        return condition


    def makePlot(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.set_size_inches(18.5, 10.5)
        ax1.set_title("Yaw"  , fontsize=12)
        ax2.set_title("Pitch", fontsize=12)
        ax3.set_title("Roll" , fontsize=12)        
        for key, drift in self.drift.items():

            ax1.plot(drift[0], drift[1], "x", label='drift (CCD ' + key + ')')
            ax2.plot(drift[0], drift[2], "x", label='drift (CCD ' + key + ')')
            ax3.plot(drift[0], drift[3], "x", label='drift (CCD ' + key + ')')

        compare = self.drift["1"]
        ax1.step(compare[0], compare[1])
        ax2.step(compare[0], compare[2])
        ax3.step(compare[0], compare[3])
        
        ax1.legend()
        ax2.legend(loc='lower right')
        ax3.legend()

        fig.suptitle("Drift between the CCDs", fontsize=18)
        plt.savefig(self.outputDir + "/DriftOnCCDs.pdf")
         



if __name__ == "__main__":
    t = TedOnCCDs()
    print(t.run())
               










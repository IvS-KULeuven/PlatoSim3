from test import Test

from matplotlib import pyplot as plt
import platosim.referenceFrames as rf
import numpy as np









class JitterOnCCDs(Test):
    """
    This test is designed to test that the jitter that is observed is independent of the CCD. The test simulates one star on the center of a
    CCD for the different CCDs, and checks that the observed jitter at a certain time is independent of the CCD.
    """

    def setNr(self):
        self.nr = "006.3"

    def setAllEffects(self):

        #Config the input file
        super().setAllEffects()
        numExposures = 40
        self.sim["ObservingParameters/NumExposures"] = numExposures
        self.sim["Platform/UseJitter"]               = "yes"
        self.sim["Platform/JitterSource"]            = "FromRedNoise"
        self.sim["ObservingParameters/CycleTime"]    = 25

        self.sim["Platform/JitterTimeScale"] = 250







        
    def runSimulation(self):

        self.jitter = {}
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
            time, yaw, pitch, roll = simFile.getPlatformYawPitchRoll(getTime=True)
            self.jitter[CCD] = [time, yaw, pitch, roll]




    
    def compare(self):

        # Save a plot of the jitter on the different CCDs
        self.makePlot()

        # Save the time and jitter in these variables
        jitter   = []
        
        # Get the common time interval
        clTime  = self.sim["ObservingParameters/CycleTime"]
        minTime = max([ np.min(jitter[0]) for jitter in self.jitter.values()])
        minTime = max(minTime, clTime)
        maxTime = min([ np.max(jitter[0]) for jitter in self.jitter.values()])



        # Store the Yaw, Pitch, Roll  is the repective variables
        for value in self.jitter.values():
            
            mask  = np.logical_and(value[0] > minTime, value[0] <= maxTime)
            yaw   = np.array(value[1][mask])
            pitch = np.array(value[2][mask])
            roll  = np.array(value[3][mask])
          
            jitter.append([yaw, pitch, roll])

        # Check that the Yaw, Pitch, Roll is the same for all the CCDs
        condition = np.all(np.diff(jitter, axis=0) == 0)
        return condition



    
    def makePlot(self):

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.set_size_inches(18.5, 10.5)
        ax1.set_title("Yaw"  , fontsize=12)
        ax2.set_title("Pitch", fontsize=12)
        ax3.set_title("Roll" , fontsize=12)        
        for key, jitter in self.jitter.items():

            ax1.plot(jitter[0], jitter[1], "x", label='jitter (CCD ' + key + ')')
            ax2.plot(jitter[0], jitter[2], "x", label='jitter (CCD ' + key + ')')
            ax3.plot(jitter[0], jitter[3], "x", label='jitter (CCD ' + key + ')')

        compare = self.jitter["1"]
        ax1.step(compare[0], compare[1])
        ax2.step(compare[0], compare[2])
        ax3.step(compare[0], compare[3])

        ax1.legend()
        ax2.legend(loc='lower right')
        ax3.legend()

        fig.suptitle("Jitter between the CCDs", fontsize=18)
        plt.savefig(self.outputDir + "/jitterOnCCDs.pdf")
         
    



if __name__ == "__main__":
    t = JitterOnCCDs()
    print(t.run())
            

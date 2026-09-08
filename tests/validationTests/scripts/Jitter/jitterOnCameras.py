from test import Test

import platosim.referenceFrames as rf
import numpy as np









class JitterOnCameras(Test):
    """This test is designed to test that the jitter that is observed is independent of the cameragroup. The test simulates one star on the center of a 
    CCD for the different camera groups, and checks that the observed jitter is the same for all of these.
    """

    def setNr(self):
        self.nr = "006.4"

    def setAllEffects(self):

        #Config the input file
        super().setAllEffects()
        numExposures = 250
        self.sim["ObservingParameters/NumExposures"] = 250
        self.sim["Platform/UseJitter"]               = "yes"
        self.sim["Platform/JitterSource"]            = "FromRedNoise"






        
    def runSimulation(self):

        self.jitter = {}
        self.cameras = ["1", "2", "3", "4"]
        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"

        # Get the amount of pixels in the subfield
        nRows    = np.array([ self.sim["SubField/NumRows"] / 2])
        nColumns = np.array([ self.sim["SubField/NumColumns"] / 2])
        mag      = np.array([ 16.2 ])
        
        for camera in self.cameras:

            # Run the simulation once for avery camera group 
            self.sim["Telescope/GroupID"] = camera
            self.sim.createStarCatalogFileFromPixelCoordinates(nRows, nColumns, mag, np.array([1]), starCatalogFilename)

            # Get the jitter from the simulation
            simFile = self.sim.run(removeOutputFile = True)
            time, yaw, pitch, roll = simFile.getPlatformYawPitchRoll(getTime=True)
            self.jitter[camera] = [time, yaw, pitch, roll]



            
    def compare(self):

        # Save the values of the jiter in a numpy array for every cameragroup
        time  = np.array([value[0] for key, value in self.jitter.items()])
        yaw   = np.array([value[1] for key, value in self.jitter.items()])
        pitch = np.array([value[2] for key, value in self.jitter.items()])
        roll  = np.array([value[3] for key, value in self.jitter.items()])

        # Checks that the time, yaw, pich and roll is the same for very camera group
        condition1 = np.all(np.diff(time, axis=0) == 0)
        condition2 = np.all(np.diff(yaw, axis=0) == 0)
        condition3 = np.all(np.diff(pitch, axis=0) == 0)
        condition4 = np.all(np.diff(roll, axis=0) == 0)

        return condition1 and condition2 and condition3 and condition4

    



if __name__ == "__main__":
    t = JitterOnCameras()
    print(t.run())
            

            

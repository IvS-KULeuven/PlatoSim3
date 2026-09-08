from test import Test

import numpy             as np
import matplotlib.pyplot as plt







"""
This test is designed to test two aspects of the Pixel-Responsivity Non-Uniformity. It checks that:
1. The mean value of the PRNU is 1.
2. The standard deviation of the PRNU should follow the values given by the input file.
"""





class PRNU(Test):

    def setNr(self):
        self.nr = "009"




    def setAllEffects(self):

        super().setAllEffects()

        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 2000
        self.sim["SubField/NumColumns"] = 2000

        self.sim["CCD/IncludeFlatfield"] = "yes"
        self.sim["CCD/Flatfield/Source"] = "FromRedNoise"
        
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]

        self.sim["ControlHDF5Content/WriteSubPixelImages"] = "true"



    def runSimulation(self):

        self.sim["CCD/Flatfield/FromRedNoise/FlatfieldNoiseRMS"] = 0.1
        self.simFile1 = self.sim.run(removeOutputFile = True)
        image1   = self.simFile1.getImage(0)


        self.sim["CCD/Flatfield/FromRedNoise/FlatfieldNoiseRMS"] = 1
        self.simFile2 = self.sim.run(removeOutputFile = True)
        image2   = self.simFile2.getImage(0)




        
    def compare(self):

        flat1 = self.simFile1.getPRNU()
        flat2 = self.simFile2.getPRNU()

        std1  = np.std(flat1)
        std2  = np.std(flat2)

        condition1 = abs(np.mean(flat1) - 1) < 0.01  and abs(np.mean(flat2) - 1) < 0.01
        condition2 = 0.01 > np.sqrt( ((std1 - 0.1)**2 + (std2 - 1 )**2)/2 )
        return condition1 and condition2



        




    
if __name__ == "__main__":
    t = PRNU()
    print(t.run())


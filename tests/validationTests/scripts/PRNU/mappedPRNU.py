
from test import Test
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


"""
This test is designed to test the prnu that is mapped from a file. It creates a prnu.hdf5 file
that devides the prnu in three "equal" parts with respective values 1, 0.5 and 0. We then then check
that the background level in the three different region matches these values. 
"""


class  MappedPRNU(Test):

    def setNr(self):
        self.nr = "44"


    def setAllEffects(self):

        super().setAllEffects()

        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 100
        self.sim["SubField/NumColumns"] = 4510

        self.sim["CCD/IncludeFlatfield"] = "yes"
        self.sim["CCD/Flatfield/Source"] = "FromFile"
        prnu_path = self.generatePRNUmap()
        self.sim["CCD/Flatfield/FromFile/FilePath"] = prnu_path
        


        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]


    def runSimulation(self):
        
        simfile = self.sim.run(removeOutputFile = True)
        self.image = simfile.getImage(0)
        



    def generatePRNUmap(self):

        prnu_path = self.outputDir + "/prnu.hdf5"

        prnu = np.ones((4510, 4510))
        prnu.transpose()[1500:3000] = 0.5
        prnu.transpose()[3000:] = 0

        with h5py.File(self.outputDir + "/prnu.hdf5", "w") as f:
            f.create_dataset("PRNU", data=prnu)
        return prnu_path

    def compare(self):

        image1 = self.image[:,:1500]
        image2 = self.image[:,1500:3000]
        image3 = self.image[:,3000:]

        # First check that the images are relativly flat (no source in FOV)
        con1 = abs(np.min(image1)- np.max(image1)) < 0.001
        con2 = abs(np.min(image2)- np.max(image2)) < 0.001
        con3 = abs(np.min(image3)- np.max(image3)) < 0.001
        condition1 = con1 and con2 and con3

        #Secondly we check that their values scale with the prnu map. (map3 is 0 and 2* map2 = map1)

        con1 = abs(np.max(image3)) < 0.001
        con2 = abs(np.max(image1) - 2*np.max(image2)) < 0.001
        condition2 = con1 and con2

        return condition1 and condition2


        


if __name__ == "__main__":
    t = MappedPRNU()
    print(t.run())

from test import Test

import numpy as np







"""
This test is designed to check the particulate contamination. The test is run with particulate contamination switch on and with particule contamination
swich off. The throughput contamination is then compared to the one from the input file. 
"""






class ParticulateContamination(Test):

    def setNr(self):
        self.nr = "011.5"


        
    def setAllEffects(self):

        super().setAllEffects()
        
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 2000
        self.sim["SubField/NumColumns"] = 2000

        self.sim["Platform/Orientation/Angles/DecPointing"] = - self.sim["Platform/Orientation/Angles/DecPointing"]


        
        
    def runSimulation(self):

        # Run the simulation with Particulate Contamination
        self.sim["CCD/IncludeParticulateContamination"] = "yes"
        self.simFile1 = self.sim.run(removeOutputFile = True)

        # Run the simulation without Particulate Contamination.
        self.sim["CCD/IncludeParticulateContamination"] = "no"
        self.simFile2 = self.sim.run(removeOutputFile = True)

    def compare(self):
        quotient       = self.simFile1.getImage(0) / self.simFile2.getImage(0)
        contamination = self.sim["CCD/Contamination/ParticulateContaminationEfficiency"]

        minRatio       = np.min(quotient)
        maxRatio       = np.max(quotient)
        stdRatio       = np.std(quotient)

        condition1 = stdRatio < 0.01
        condition2 = minRatio - stdRatio < contamination < maxRatio + stdRatio

        return condition1 and condition2



    
if __name__ == "__main__":
    t = ParticulateContamination()
    print(t.run())


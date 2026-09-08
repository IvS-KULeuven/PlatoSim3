import numpy as np

from test import Test






"""
This test checks the simple model for the Charge Transfer Inefficiency (CTI). The test simulates one full column of the CCD and CTE enabled without any stars. 

We would expect that the pixel values in the output image behaves as ~ <CTI>^(row + 1). (where <CTI> is the mean CTI as given in the input file. 

The test passes if the RMS difference between the observed and expected values in the subfield is smaller than 0.05 
"""

class SimpleCTI(Test):
    
    def setNr(self):
        self.nr = "016.1"

    def setAllEffects(self):

        super().setAllEffects()
        self.numRows = 4510
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = self.numRows
        self.sim["SubField/NumColumns"] = 1

        # Make sure no stars fall onto the subfield. 
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]
        # Include Simple CTI effects
        self.sim["CCD/IncludeCTIeffects"] = "yes"
        self.sim["CCD/CTI/Model"] = "Simple"


    def compare(self):

        # Get the image of the exposure
        column = self.simFile.getImage(0)
        cte    = self.sim["CCD/CTI/Simple/MeanCTE"]
        rows   = np.arange(self.numRows)

        # Expected CTE
        sel    = np.arange(0, self.numRows, 100)
        expected = np.array([cte**(row + 1) for row in rows[sel]])

        # Test that the theoretical CTE is close to the one we find from PlatoSIM. 
        condition1 = np.std(expected - column[sel]/column[0]) < 0.05
        condition2 = abs(np.mean(expected - column[sel]/column[0])) < 0.01

        return condition1 and condition2





if __name__ == "__main__":
    t = SimpleCTI()
    print(t.run())

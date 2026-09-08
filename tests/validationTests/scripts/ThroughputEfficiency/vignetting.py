from test import Test
import platosim.referenceFrames   as rf
import numpy                      as np
import matplotlib.pyplot          as plt
import pandas                     as pd


"""
This test is designed to check vignetting. The test is runs the simulation without a source into the subfield.
The loss of effeciency (LoI) is calculated with respect to the distance of the optical axis. There are three conditions
that should pass for the test to be succesfull.
1. The FOV radius should be checked with respect to the one given in the input file.
2. Outside the FOV radius, the LoI should be 1.
3. Inside the FOV the LoI should be increase.
"""





class Vignetting(Test):

    def setNr(self):
        self.nr = "011.2"




    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 4200
        self.sim["SubField/NumColumns"] = 4200
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]
        self.sim["CCD/IncludeRelativeTransmissivity"] = "yes"





    def runSimulation(self):

        simFile = self.sim.run(removeOutputFile = True)
        self.image = simFile.getImage(0)
        self.image = self.image / np.max(self.image)






    def compare(self):

        pixelSize      = self.sim["CCD/PixelSize"]
        focalL         = self.sim["Camera/FocalLength/ConstantValue"] * 1000
        rows           = range(0, self.sim["SubField/NumRows"], 250)[1:-1]
        cols           = range(0, self.sim["SubField/NumColumns"], 250)[1:-1]
        radius         = self.sim["CCD/RelativeTransmissivity/RadiusFOV"]

        FPCoordinates  = [rf.pixelToFocalPlaneCoordinates(col, row, pixelSize, -1.3 , 82.48, 3 * np.pi / 2) for row in rows for col in cols]
        distance       = np.array([rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, focalL) for xFP, yFP in FPCoordinates])
        distance       = np.rad2deg(distance)

        efficiencyLoss = [1 - self.image[row, col] / self.image[4199][0] for row in rows for col in cols]


        d  = {'distance' : distance , 'LoE' : efficiencyLoss}
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=['distance'])
        df = df.reset_index(drop=True)

        df['LoEshift'] = df['LoE'].shift(periods=1, fill_value=0)
        df['delta']    = df['LoE'] - df['LoEshift']

        maxPos = df['distance'][df['delta'] == df['delta'].max()]
        beforeMax, afterMax = maxPos.index[0] - 1 , maxPos.index[0] + 1
        [posBeforeMax] = df['distance'][df.index == beforeMax].values
        [posAfterMax]  = df['distance'][df.index == afterMax].values

        condition1 = posBeforeMax < radius < posAfterMax
        condition2 = all(df['LoE'][df.index >= afterMax] == 1)
        condition3 = all(df['delta'] >= 0.)

        return condition1 and condition2 and condition3






if __name__ == "__main__":
    t = Vignetting()
    print(t.run())



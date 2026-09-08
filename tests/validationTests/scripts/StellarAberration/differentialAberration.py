#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import platosim.referenceFrames as rf

from platosim.simulation import Simulation
from test                import Test
from skimage.measure     import EllipseModel
from matplotlib.patches  import Ellipse


"""
This test checks the differential aberration. The configuration starts with a
custom CCD whose origin (the optimal axis) alligns with the pointing of the detector.
We simulate different stars on the CCD with increasing distance from the optimal
axis. We simulate these stars for an entire year.

1. For the first test we keep track of the absolute difference between a simulation
with absolute aberation and without absolute abberation. We show how this evolves
over the period of one year.

2. The second test shows the path the star makes on the CCD over one year. This path
can be approximated with an ellipse and we test that the dimmensions of the ellips
increase with inceasing distance from the optimal axis.
"""


class DifferentialAberration(Test):

    def setNr(self):
        self.nr = "003.2"

    def setAllEffects(self):
        super().setAllEffects()

        deltaStep = int(20*86400/25)
        endTime   = int(360*86400/25)

        self.exposures = np.arange(0, endTime, deltaStep)
        self.positions = np.arange(10,3500,700)
        DKA = "yes"




    def simulate(self, DKA, pos, beginExposureNr, outputName):

        # Set up a simulation object
        sim = Simulation(outputName, outputDir=self.outputDir)

        # Define and set star catalog file
        starFileName = f'{self.outputDir}/starCatalog{self.nr}_{DKA}_pos{pos}_exp{beginExposureNr}.txt'
        row = col = np.array([pos])
        mag = np.array([11])
        sid = np.array([1])
        sim.createStarCatalogFileFromPixelCoordinates(row+4, col+4, mag, sid, starFileName)
        # Simulate only a small subfield
        sim["SubField/NumRows"]         = 8
        sim["SubField/NumColumns"]      = 8
        sim["SubField/ZeroPointRow"]    = pos
        sim["SubField/ZeroPointColumn"] = pos

        # Simulate a single exposure
        sim["ObservingParameters/NumExposures"]    = 1
        sim["ObservingParameters/BeginExposureNr"] = beginExposureNr

        # Activate kinematic differential aberration
        sim["Camera/IncludeAberrationCorrection"] = DKA
        sim["Camera/AberrationCorrection/Type"]   = "differential"

        # Turn off all effect that can cause pixel displacements
        sim["Platform/UseJitter"] = "no"
        sim["Telescope/UseDrift"] = "no"

        # Run simulation
        f = sim.run(removeOutputFile=True)

        # Fetch parameters
        row, col = f.getStarPositions(sid[0])
        return 4-row[0], 4-col[0]



    def run(self):

        npos = len(self.positions)
        nexp = len(self.exposures)
        c    = np.zeros((npos,nexp))

        dx = np.zeros((npos, nexp))
        dy = np.zeros((npos, nexp))

        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i in range(npos):
            for j in range(nexp):
                dx[i,j], dy[i,j] = self.simulate(True, self.positions[i], self.exposures[j], f"output{self.nr}")                
                c[i,j] = np.sqrt((dx[i,j]**2 + dy[i,j]**2))


        # Plot results
        fig, ax = plt.subplots()
        for i in range(npos):
            ax.plot(self.exposures*25/86400, c[i,:], 'o-', label=f'(row,col) = ({self.positions[i]}, {self.positions[i]}) pix', color=colors[i])

        ax.set(ylabel='Delta position [pixel]')
        fig.legend()
        plt.savefig(self.ioPath + "/test" + self.nr +  "/distanceToNoAberration.png")

        widths  = []
        heights = []


        fig, ax = plt.subplots()
        for i in range(npos):

            points = np.array([ (x, y) for x, y in zip(dx[i,:], dy[i,:])])
            ell = EllipseModel.from_estimate(points)

            xc, yc = ell.center
            a, b = ell.axis_lengths
            theta = ell.theta

            plt.scatter(dx[i,:], dy[i,:], label=f'(row, col) = ({self.positions[i]}, {self.positions[i]}) pix', color=colors[i])
            plt.scatter([0.3, 0.3, -0.3, -0.3], [0.3, -0.3, 0.3, -0.3])
            ell_path = Ellipse((xc, yc), 2*a, 2*b, angle=theta*180/np.pi, edgecolor=colors[i], facecolor='none')

            widths.append(min(a, b))
            heights.append(max(a,b))

            ax.add_patch(ell_path)

        plt.xlabel("dx [pxl]")
        plt.ylabel("dy [pxl]")
        plt.legend()
        plt.savefig(self.ioPath + "/test" + self.nr +  "/ellipsOnCCD.png")

        return doesIncrease(widths) and doesIncrease(heights)



def doesIncrease(input_list):
    inpt = np.array(input_list)
    return np.all( (inpt[1:] - inpt[:-1]) >= 0)



if __name__ == "__main__":
    t = DifferentialAberration()
    print(t.run())



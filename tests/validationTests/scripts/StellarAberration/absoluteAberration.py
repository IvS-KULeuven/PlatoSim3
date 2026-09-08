import numpy           as np
import platosim.referenceFrames as rf
from test       import Test
from math       import radians
from platosim.validation import equatorial2galactic, galactic2equatorial, aberration
import math
import matplotlib.pyplot as plt
from test import eprint



"""
This test checks the absolute aberration. The test consists out of 3 parts:

1. The first test plots the position of an aberrated star over one
year and a theoretical predicted path (where we assume a circular orbit with a constant velocity for the spacecraft) over one year. This figure is
saved into the ioFiles directory, it is possible to visually confirm that the two paths are similar.

2. The second test checks that the aberration is zero for a star whose position wrt the spacecraft is orthogonal to the spacecrafts velocity.

3. The last test checks that the aberration is maximal for a star whose position is parallel to the velocity.
"""


class AbsoluteAberration(Test):

    def setNr(self):
        self.nr = "003.1"

    def setAllEffects(self):
        super().setAllEffects()

        # Define star catalog file
        starFileName = self.outputDir + "/starCatalog" + self.nr + ".txt"
        pos  = np.array([100])
        mag  = np.array([12.5])
        self.sim.createStarCatalogFileFromPixelCoordinates(pos, pos, mag, np.array([1]), starFileName)
        self.sim["ObservingParameters/StarCatalogFile"] = starFileName


        dim = 200
        self.sim["SubField/NumRows"]         = dim
        self.sim["SubField/NumColumns"]      = dim
        self.sim["SubField/ZeroPointRow"]    = 0
        self.sim["SubField/ZeroPointColumn"] = 0


        self.sim["ObservingParameters/NumExposures"]   = 1
        self.sim["Camera/IncludeAberrationCorrection"] = "yes"
        self.sim["Camera/AberrationCorrection/Type"]   = "absolute"

        self.startTime = 9442527.53198146
        self.velocity  = [-0.4283823880272332, 0.9035748636371567, 0.006402767462494304]
        self.sim["Camera/AberrationCorrection/StartTime"] = self.startTime



    def runForYear(self):
        # We run the simulation multiple times over one year with 5 days
        #between two consecutive exposures.
        raPlatform            = radians(self.sim["Platform/Orientation/Angles/RAPointing"])
        decPlatform           = radians(self.sim["Platform/Orientation/Angles/DecPointing"])
        solarPanelOrientation = radians(self.sim["Platform/Orientation/Angles/SolarPanelOrientation"])
        tiltAngle             = radians(self.sim["Telescope/TiltAngle"])
        azimuthAngle          = radians(self.sim["Telescope/AzimuthAngle"])
        focalPlaneAngle       = radians(self.sim["Camera/FocalPlaneOrientation/ConstantValue"])
        focalLength           = self.sim["Camera/FocalLength/ConstantValue"] * 1000
        numExposures          = 365 * 24 * 60 * 60 // 25
        deltaNumExposures     = 5 * (24 * 60 * 60 // 25)

        outputRa  = []
        outputDec = []
        startTime0 = self.startTime

        for exposureNr in range(0, numExposures + 1, deltaNumExposures):
            self.sim["ObservingParameters/BeginExposureNr"] = exposureNr
            self.sim["Camera/AberrationCorrection/StartTime"] = startTime0 + exposureNr*25
            self.simFile = self.sim.run(removeOutputFile=True)
            ([x], [y])   = self.simFile.getStarCoordinates(exposureNr)[3:5]

            ra, dec      = rf.focalPlaneToSkyCoordinates(x, y, raPlatform, decPlatform, solarPanelOrientation,   \
                                                         tiltAngle, azimuthAngle, focalPlaneAngle, focalLength)
            outputRa.append(ra)
            outputDec.append(dec)

        self.outputRa  = np.rad2deg(np.array(outputRa))
        self.outputDec = np.rad2deg(np.array(outputDec))


    def runForSecondTest(self):

        # Rest begin exposure numbers
        self.sim["ObservingParameters/BeginExposureNr"] = 0
        self.sim["Camera/AberrationCorrection/StartTime"] = self.startTime

        lamb = np.arctan2(-self.velocity[0], self.velocity[1])
        beta = 0

        ra, dec = np.rad2deg(rf.ecliptic2equatorial(lamb, beta))

        self.sim["Platform/Orientation/Source"] = "Angles"
        self.sim["Platform/Orientation/Angles/RAPointing"] = ra
        self.sim["Platform/Orientation/Angles/DecPointing"] = dec

        dec = np.deg2rad(dec)
        ra  = np.deg2rad(ra)

        self.sim["SubField/NumRows"]         = 510
        self.sim["SubField/NumColumns"]      = 510

        self.sim.setSubfieldAroundSkyCoordinates(ra, dec, 510, 510)

        # Set the star catalog
        starFileName = self.outputDir + "/starCatalog" + self.nr + ".txt"
        myFile = open(starFileName, "w")
        myFile.write("# RA DEC Vmag starID\n")
        myFile.write("{0}  {1}  {2}  {3}\n".format(np.rad2deg(ra), np.rad2deg(dec), 16.5, 1))
        myFile.close()

        # Run the simulation with aberration
        simFile = self.sim.run(removeOutputFile=True)
        dummy, dummy, dummy, xFP1, yFP1, dummy = simFile.getStarCoordinates(0)
        xFP1, yFP1 = xFP1[0], yFP1[0]

        # Run the simulation without aberration
        self.sim["Camera/IncludeAberrationCorrection"] = "no"
        simFile = self.sim.run(removeOutputFile=True)
        dummy, dummy, dummy, xFP2, yFP2, dummy = simFile.getStarCoordinates(0)
        xFP2, yFP2 = xFP2[0], yFP2[0]

        self.xDelta1, self.yDelta1 = abs(xFP1 - xFP2) < 0.1, abs(yFP1 - yFP2) < 0.1



    def runForThirdTest(self):

        # Calculate the orthogonal velocity
        orthogonalVelocity = [(self.velocity[1] - self.velocity[2]), self.velocity[2] - self.velocity[0], self.velocity[0] - self.velocity[1]]
        orthogonalNorm     = np.sqrt(orthogonalVelocity[0]**2 + orthogonalVelocity[1]**2 + orthogonalVelocity[2]**2)
        orthogonalVelocity = [ x / orthogonalNorm for x in orthogonalVelocity]


        lamb = np.arctan2(orthogonalVelocity[1], orthogonalVelocity[0])
        beta = np.arcsin(orthogonalVelocity[2])
        ra, dec = np.rad2deg(rf.ecliptic2equatorial(lamb, beta))

        self.sim["Platform/Orientation/Source"] = "Angles"
        self.sim["Platform/Orientation/Angles/RAPointing"] = ra
        self.sim["Platform/Orientation/Angles/DecPointing"] = dec

        dec = np.deg2rad(dec)
        ra  = np.deg2rad(ra)

        self.sim["SubField/NumRows"]    = 510
        self.sim["SubField/NumColumns"] = 510

        self.sim.setSubfieldAroundSkyCoordinates(ra, dec, 510, 510)

        # Set the star catalog
        starFileName = self.outputDir + "/starCatalog" + self.nr + ".txt"
        myFile = open(starFileName, "w")
        myFile.write("# RA DEC Vmag starID\n")
        myFile.write("{0}  {1}  {2}  {3}\n".format(np.rad2deg(ra), np.rad2deg(dec), 16.5, 1))
        myFile.close()

        # Run the simulation with aberration
        self.sim["Camera/IncludeAberrationCorrection"] = "yes"
        simFile = self.sim.run(removeOutputFile=True)
        dummy, dummy, dummy, xFP1, yFP1, dummy = simFile.getStarCoordinates(0)
        xFP1, yFP1 = xFP1[0], yFP1[0]

        # Run the simulation without aberration
        self.sim["Camera/IncludeAberrationCorrection"] = "no"
        simFile = self.sim.run(removeOutputFile=True)
        dummy, dummy, dummy, xFP2, yFP2, dummy = simFile.getStarCoordinates(0)
        xFP2, yFP2 = xFP2[0], yFP2[0]

        self.xDelta2, self.yDelta2 = abs(xFP1 - xFP2) > 0.001, abs(yFP1 - yFP2) > 0.001


    def runSimulation(self):
        # Visually check that the abberation is in line with a theoretical estimate (with constant velocity 30km/s)
        self.runForYear()
        self.compareToTheoreticalEstimate()

        # Run for second test
        self.runForSecondTest()

        # Run for the third test
        self.runForThirdTest()




    def compareToTheoreticalEstimate(self):

        # The theoretical predicted aberration is calculated over the span of one year. We check whether the observed displacements over
        # one year fall withing the predicted range.

        ([inputRa], [inputDec]) = (self.simFile.getStarCatalog()[1:3])
        lonStar, latStar = equatorial2galactic(inputRa, inputDec)

        # Displacement that is measured over the span of a year.
        dxMeasurement  = (self.outputRa - inputRa) * 3600 * np.cos(np.deg2rad(self.outputDec))
        dyMeasurement  = (self.outputDec - inputDec) * 3600
        radialDistance = [x0**2 + y0**2 for x0, y0 in zip(dxMeasurement, dyMeasurement)]

        # Theoretically predicted aberration over one year
        theoAber = [aberration(lonStar, latStar, lSun) for lSun in np.linspace(0, 2*np.pi, 360)]
        tAberGal = [galactic2equatorial(lamStar, betaStar) for lamStar, betaStar in theoAber]

        raTheo, decTheo = zip(*tAberGal)

        dxTheo     = (raTheo - inputRa) * 3600 * np.cos(np.deg2rad(decTheo))
        dyTheo     = (decTheo - inputDec ) * 3600

        plt.plot(dxMeasurement, dyMeasurement, 'r')
        plt.plot(dxTheo, dyTheo)
        plt.savefig(self.ioPath + "/test" + self.nr +  "/aberrationOverYear.png")



    def NoAberration(self):
        print(self.xDelta1,self.yDelta1)
        return self.xDelta1 and self.yDelta1



    def maxAberration(self):
        return self.xDelta2 or self.yDelta2



    def compare(self):

        result1 = self.NoAberration()
        result2 = self.maxAberration()
        
        return (result1 and result2)







if __name__ == "__main__":
    t = AbsoluteAberration()
    print(t.run())


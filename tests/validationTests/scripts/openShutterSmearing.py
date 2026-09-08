import matplotlib.pyplot as plt

from platosim.validation import switchOffAllEffects
import platosim.referenceFrames as rf
import numpy as np
from test import Test
import math



"""This test is designed to check open-shutter smearing of PlatoSim. It runs the simulator with and without open-shutter-smearing, and with mechanical
   vignetting. The difference image between these two images contains the flux that is accumulated during readout. The test checks the difference in different
   rows and different columns of the image. The first condition for the test to pass is if pixels in the same column accumulate the same flux. The second condition
   is that the pixels with columns located within the FOV gather flux over the duration of the whole readout, and beyond that decrease proportional to the number of rows. """


class OpenShutterSmearing(Test):

    def setNr(self):
        self.nr = "015"

    def setAllEffects(self):

        switchOffAllEffects(self.sim)
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 4510
        self.sim["SubField/NumColumns"] = 4510

        self.sim["Platform/JitterTimeScale"] = 3600

        # No stars in the subfield
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]
        self.sim["CCD/IncludeRelativeTransmissivity"] = "yes"        

        self.sim["PSF/Model"] = "AnalyticNonGaussian"

    def runSimulation(self):

        # Runs the text without open shutter smearing
        self.sim["CCD/IncludeOpenShutterSmearing"] = "no"
        self.simFileWithoutOSS = self.sim.run(removeOutputFile = True)

        # Runs the text with open shutter smearing
        self.sim["CCD/IncludeOpenShutterSmearing"] = "yes"
        self.simFileWithOSS = self.sim.run(removeOutputFile = True)

    def compare(self):

        # Get the image from the simulations
        image1 = self.simFileWithOSS.getImage(0)
        image2 = self.simFileWithoutOSS.getImage(0)
        diffImage = image1 - image2

        columns = np.array([])
        fluxes  = np.array([])

        for row in range(0, self.sim["SubField/NumRows"], 1000):
            for col in range(0, self.sim["SubField/NumColumns"], 100):
                plt.plot([col + 0.5], diffImage[row][col] / diffImage[0][0], "bx")
                columns = np.append(columns, col)
                fluxes  = np.append(fluxes, diffImage[row][col] / diffImage[0][0])



        pixelSize  = self.sim["CCD/PixelSize"] / 1000.
        fovDegrees = self.sim["CCD/RelativeTransmissivity/RadiusFOV"]
        coefficients = self.sim["CCD/RelativeTransmissivity/Coefficients"]




        xFP, yFP = rf.focalPlaneCoordinatesFromGnomonicRadialDistance(np.deg2rad(fovDegrees), self.sim["Camera/FocalLength/ConstantValue"] * 1000,  0)
        radiusPixels, zero = rf.focalPlaneToPixelCoordinates(xFP, yFP, self.sim["CCD/PixelSize"], 0, 0, 0)
        intersection = np.sqrt(radiusPixels**2 - self.sim["CCD/NumColumns"]**2)

        # Get the accumulated flux after the FOV
        pixelDistances      = np.arange(intersection, radiusPixels)
        openShutterSmearing = np.sqrt(radiusPixels**2 - pixelDistances**2) / np.sqrt(radiusPixels**2 - intersection**2)


        modelRelTransmissivityInFOV  = np.array([])

        # Get the theoretical accumulated flux after FOV
        for pixelDistance in pixelDistances:

            xFP, yFP = rf.pixelToFocalPlaneCoordinates(pixelDistance, 0, self.sim["CCD/PixelSize"], 0, 0, 0)     # Focal-plane coordinates[mm]"
            distance = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, self.sim["Camera/FocalLength/ConstantValue"] * 1000))     # Angular distance from the OA [degrees]
            model = coefficients[0] * math.pow(distance, 2) + coefficients[1] * math.pow(distance, 4) + coefficients[2] * math.pow(distance, 6)
            modelRelTransmissivityInFOV = np.append(modelRelTransmissivityInFOV, 1 - model / 100.)

        plt.plot(pixelDistances, openShutterSmearing * modelRelTransmissivityInFOV, "g")



        # Get the theoretical accumulated flux after FOV
        pixelDistances         = np.arange(math.floor(intersection))
        modelRelTransmissivity = np.array([])

        for pixelDistance in pixelDistances:

            xFP, yFP = rf.pixelToFocalPlaneCoordinates(pixelDistance, 0, self.sim["CCD/PixelSize"], 0, 0, 0)     # Focal-plane coordinates[mm]"
            distance = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, self.sim["Camera/FocalLength/ConstantValue"] * 1000))     # Angular distance from the OA [degrees]
            model = coefficients[0] * math.pow(distance, 2) + coefficients[1] * math.pow(distance, 4) + coefficients[2] * math.pow(distance, 6)
            modelRelTransmissivity = np.append(modelRelTransmissivity, 1 - model / 100.)


        theoreticalCurve = np.concatenate((modelRelTransmissivity, openShutterSmearing * modelRelTransmissivityInFOV))
        theoreticalCurve = np.array([theoreticalCurve[num] for num in range(0, self.sim["SubField/NumColumns"], 100)])


        #  Checks if pixels in the same column accumulate the same flux
        condition1 = all([np.std(fluxes[columns == num]) < 0.05 for num in range(0, self.sim["SubField/NumColumns"], 100) ])

        fluxes     = np.array([ np.mean(fluxes[columns == num]) for num in range(0, self.sim["SubField/NumColumns"], 100) ])

        # Check that the flux in the pixels matches the theoretical predicted one
        condition2 = abs(np.mean(fluxes - theoreticalCurve)) < 0.05 and np.std(fluxes  - theoreticalCurve) < 0.05
        return condition1 and condition2







if __name__ == "__main__":
    t = OpenShutterSmearing()
    print(t.run())

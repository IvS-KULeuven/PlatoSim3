
import os
import numpy as np
from test import Test,eprint
from platosim.simulation import Simulation




class NonlinearGainTest(Test):

    def __init__(self) -> None:

        # Create the folder where all simulation output files will be written

        self.nr = 1234
        self.outputDir = os.environ["PLATO_PROJECT_HOME"] + f"/tests/validationTests/ioFiles/test{self.nr}"
        if not os.path.isdir(self.outputDir):
            os.mkdir(self.outputDir)

        # Select the input configuration yaml file

        self.inputfile = os.environ["PLATO_PROJECT_HOME"] + "/inputfiles/inputfile.yaml"



    def runNonLinearGainTest(self) -> bool:

        # To get sources with ADUs between 0 and 65K we take stars with different magnitudes 
        # No brighter than 8.6 otherwise the ADUs do no longer fit into an uint16, as blooming is switched off.
        # All simulations use the same sky coordinates.

        stellarMagnitudes = np.linspace(8.8, 17.0, 20)                 # Possible magnitudes of this one star
        starID = np.array([1])
            
        # We'll use the CCD#1 of a normal camera. The exact position of the imagette on the CCD is not important. 
        # It's size is taken small as we only have one star in our catalog.

        ccdCode = '1'
        xCCDpixel = 100                   # Column coordinate
        yCCDpixel = 100                   # Row coordinate
        subfieldSizeX = 10
        subfieldSizeY = 10

        # First run the simulator without the non-linear gain present

        pixels_without_NLG = np.array([])

        for n, Vmag in enumerate(stellarMagnitudes):
            # Create a simulation object; NLG = non-linear gain

            sim_without_NLG = Simulation(f"test{self.nr}_{n}_without_NLG",
                                         configurationFile = self.inputfile, outputDir = self.outputDir)

            # Switch off all effects, in particular the photon and the readout noise
            # We do need the quantisation part to go from e- to ADU.

            sim_without_NLG.turnOffAllEffects()
            sim_without_NLG["CCD/IncludeQuantisation"] = "yes"

            # To go to the lowest ADU pixel values with faint stars, we need to switch
            # off the sky background.

            sim_without_NLG["Sky/SkyBackground/UseConstantSkyBackground"] = "yes"
            sim_without_NLG["Sky/SkyBackground/BackgroundValue"] = 0.0

            # Create a small imagette around the star

            success = sim_without_NLG.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel, yCCDpixel, subfieldSizeX, subfieldSizeY)
            if not success: 
                eprint("Error: could not set subfield around star")
                return
  
            # Create a star catalog with the proper name

            starCatalogName = self.outputDir + f"/starCatalog{n}.txt"
            sim_without_NLG.createStarCatalogFileFromPixelCoordinates(np.array([yCCDpixel]), np.array([xCCDpixel]), 
                                                                      np.array([Vmag]), starID, starCatalogName)
            sim_without_NLG["ObservingParameters/StarCatalogFile"] = starCatalogName


            # We only need one exposure to determine the stellar flux
        
            sim_without_NLG["ObservingParameters/NumExposures"] = 1

            # Run the PlatoSim simulator 

            simFile = sim_without_NLG.run(removeOutputFile=True)
            image = simFile.getImage(0)
            biasMapLeft = simFile.getBiasMapLeft(0)
            biasMapRight = simFile.getBiasMapRight(0)
            meanBiasLevel = int(0.5*(biasMapLeft.mean() + biasMapRight.mean()))
            pixels_without_NLG = np.append(pixels_without_NLG, image.flatten() - meanBiasLevel)


        # Next, run the simulator *with* a non-linear gain

        pixels_with_NLG = np.array([])

        for n, Vmag, in enumerate(stellarMagnitudes):
            # Create a simulation object; NLG = non-linear gain

            sim_with_NLG = Simulation(f"test{self.nr}_{n}_with_NLG",
                                      configurationFile = self.inputfile, outputDir = self.outputDir)

            # Switch off all effects, in particular the photon and the readout noise
            # We do need the quantisation part to go from e- to ADU, and switch on again the non-linear gain

            sim_with_NLG.turnOffAllEffects()
            sim_with_NLG["CCD/IncludeQuantisation"]     = "yes"
            sim_with_NLG["CCD/IncludeGainNonlinearity"] = "yes"


            # To go to the lowest ADU pixel values with faint stars, we need to switch
            # off the sky background.

            sim_with_NLG["Sky/SkyBackground/UseConstantSkyBackground"] = "yes"
            sim_with_NLG["Sky/SkyBackground/BackgroundValue"] = 0.0

            # Create a small imagette around the star

            success = sim_with_NLG.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel, yCCDpixel, subfieldSizeX, subfieldSizeY, 
                                                                     normal=True)
            if not success: 
                eprint("Error: could not set subfield around star")
                return

            # Create a star catalog with the proper name

            starCatalogName = self.outputDir + f"/starCatalog{n}.txt"
            mag = np.array([Vmag])
            sim_with_NLG.createStarCatalogFileFromPixelCoordinates(np.array([yCCDpixel]), np.array([xCCDpixel]), 
                                                                      np.array([Vmag]), starID, starCatalogName)
            sim_with_NLG["ObservingParameters/StarCatalogFile"] = starCatalogName

            # We only need one exposure to determine the stellar flux
        
            sim_with_NLG["ObservingParameters/NumExposures"] = 1

            # Run the PlatoSim simulator 

            simFile = sim_with_NLG.run(removeOutputFile=True)
            image = simFile.getImage(0)
            biasMapLeft = simFile.getBiasMapLeft(0)
            biasMapRight = simFile.getBiasMapRight(0)
            meanBiasLevel = int(0.5*(biasMapLeft.mean() + biasMapRight.mean()))
            pixels_with_NLG = np.append(pixels_with_NLG, image.flatten() - meanBiasLevel)


        # Save the pixel values to a txt file, so that we could make a plot if necessary

        x = pixels_without_NLG
        y = pixels_with_NLG - pixels_without_NLG 
        np.savetxt(self.outputDir + "/pixelvalues_in_pixelmap.txt", np.transpose([x,y]))

        # Fit the pixel values with a parabole
        # y = coeff[2] + coeff[1] * x + coeff[0] * x^2     (different order than the coefficients in the input yaml file!)

        coeff = np.polyfit(x,y,2)

        # Compare with the expected coefficients in the input yaml file
        # Flip to have the same order of coefficients as in the output of polyfit()

        expectedCoeff = np.flip(np.array(sim_with_NLG["CCD/Gain/Nonlinearity"]))

        intercepts_are_close = np.abs(coeff[2]) < 5.0e-3
        slopes_are_close = np.abs(expectedCoeff[1] - coeff[1]) / np.abs(expectedCoeff[1]) < 0.05
        leadingcoeffs_are_close = np.abs(expectedCoeff[0] - coeff[0]) / np.abs(expectedCoeff[0]) < 0.05

        if intercepts_are_close and slopes_are_close and leadingcoeffs_are_close:
            return True
        else:
            return False


    def run(self) -> bool:
        success = self.runNonLinearGainTest()
        return success



if __name__ == "__main__":
    myTest = NonlinearGainTest()
    print(myTest.runNonLinearGainTest())


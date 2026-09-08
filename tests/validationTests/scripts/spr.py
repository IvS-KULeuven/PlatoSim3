
import os
import numpy as np
from test import Test,eprint
from platosim.simulation import Simulation



# Class to test the Stellar pollution ratio

class SPRTest(Test):

    def __init__(self) -> None:

        # Create the folder where all simulation output files will be written

        self.nr = 9325
        self.outputDir = os.environ["PLATO_PROJECT_HOME"] + f"/tests/validationTests/ioFiles/test{self.nr}"
        if not os.path.isdir(self.outputDir):
            os.mkdir(self.outputDir)

        # Select the input configuration yaml file

        self.inputDir = os.environ["PLATO_PROJECT_HOME"] + "/inputfiles/"
        self.inputfile = os.environ["PLATO_PROJECT_HOME"] + "/inputfiles/inputfile.yaml"




    def runSPRTestWithOneStar(self) -> bool:
        """ 
        Test: if there is only one star in the subfield, the SPR should be 0.0.
        """

        starIDs = np.array([1])
            
        # We'll use the CCD#1 of a normal camera. The exact position of the imagette on the CCD is not important. 
        # It's size is taken small as we only have one or two stars in our catalog.

        ccdCode = '1'
        xCCDpixel = 100                   # Column coordinate
        yCCDpixel = 100                   # Row coordinate
        subfieldSizeX = 10
        subfieldSizeY = 10

        # Run the simulator

        sim = Simulation(f"test{self.nr}_SPR_one_star", configurationFile = self.inputfile, outputDir = self.outputDir)

        # Be sure to switch on photometry, and save the star ID in the photomtry file, so that the light curve is extracted

        sim["Photometry/IncludePhotometry"] = "yes"
        sim["Photometry/TargetFileName"] = self.outputDir + "myphotometry.txt"
        np.savetxt(self.outputDir + "myphotometry.txt", starIDs, fmt="%d")

        # Create a small imagette around the star

        success = sim.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel, yCCDpixel, subfieldSizeX, subfieldSizeY)
        if not success: 
            eprint("Error: could not set subfield around star")
            return

        # Create a star catalog with the proper star ID, and a magnitude of 10.5

        starCatalogName = self.outputDir + "/starCatalog.txt"
        sim.createStarCatalogFileFromPixelCoordinates(np.array([yCCDpixel]), np.array([xCCDpixel]), np.array([10.5]),   \
                                                      starIDs, starCatalogName)
        sim["ObservingParameters/StarCatalogFile"] = starCatalogName

        # We only need one exposure to determine the SPR
        
        sim["ObservingParameters/NumExposures"] = 1

        # Run the PlatoSim simulator 

        simFile = sim.run(removeOutputFile=True)
        dummy, dummy, dummy, dummy, dummy, SPR = simFile.getApertureMask(starIDs[0])

        if SPR[0] == 0.0:
            return True
        else:
            return False







    def runSPRTestWithTwoEqualApproachingStars(self) -> bool:
        """ 
        Test with two stars: the nearer the stars, the larger the SPR should become
        """

        # The entire catalog only consists of 2 stars:

        starIDs = np.array([1,2])
            
        # For only 1 star we need the photometry. That one is the target, the other one is the contaminant

        np.savetxt(self.outputDir + "myphotometry.txt", np.array([1]), fmt="%d")

        # We'll use the CCD#1 of a normal camera. The exact position of the imagette on the CCD is not important. 
        # It's size is taken small as we only have one or two stars in our catalog.

        ccdCode = '1'
        xCCDpixel_center = 100                   # Column coordinate
        yCCDpixel_center = 100                   # Row coordinate
        subfieldSizeX = 10
        subfieldSizeY = 10

        # Run the simulator for three different distances between the two stars

        SPRs = []
        for n in range(3):

            sim = Simulation(f"test{self.nr}_SPR_two_stars_dist{n}", configurationFile = self.inputfile, outputDir = self.outputDir)

            # Be sure to switch on photometry, and save the star ID in the photomtry file, so that the light curve is extracted

            sim["Photometry/IncludePhotometry"] = "yes"
            sim["Photometry/TargetFileName"] = self.outputDir + "myphotometry.txt"

            # Create a small imagette around the star

            success = sim.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel_center, yCCDpixel_center, subfieldSizeX, subfieldSizeY)
            if not success: 
                eprint("Error: could not set subfield around star")
                return

            # Create a star catalog with the proper star ID, and the magnitudes both 10.5
            # Same y-coordinate, but increasing x-coordinate.

            starCatalogName = self.outputDir + "/starCatalog.txt"
            Vmags = np.array([10.5, 10.5])
            xpixCoord = np.array([xCCDpixel_center, xCCDpixel_center+n+1])
            ypixCoord = np.array([yCCDpixel_center, yCCDpixel_center])
            sim.createStarCatalogFileFromPixelCoordinates(ypixCoord, xpixCoord, Vmags,   \
                                                          starIDs, starCatalogName)
            sim["ObservingParameters/StarCatalogFile"] = starCatalogName

            # We only need one exposure to determine the SPR
            
            sim["ObservingParameters/NumExposures"] = 1

            # Run the PlatoSim simulator 

            simFile = sim.run(removeOutputFile=True)
            dummy, dummy, dummy, dummy, dummy, SPR = simFile.getApertureMask(starIDs[0])

            SPRs.append(SPR[0])

        # Verify if they are all between 0 and 1 and that the SPR is decreasing with 
        # increasing distance.
        
        if 1 > SPRs[0] > SPRs[1] > SPRs[2] > 0:
            return True
        else:
            return False







    def runSPRTestWithTwoUnequalStars(self) -> bool:
        """ 
        Test with two stars: the brighter the contaminant, the larger the SPR should become.
        """

        # The entire catalog only consists of 2 stars:

        starIDs = np.array([1,2])
            
        # For only 1 star we need the photometry. That one is the target, the other one is the contaminant

        np.savetxt(self.outputDir + "myphotometry.txt", np.array([1]), fmt="%d")

        # We'll use the CCD#1 of a normal camera. The exact position of the imagette on the CCD is not important. 
        # It's size is taken small as we only have one or two stars in our catalog.

        ccdCode = '1'
        xCCDpixel_center = 100                   # Column coordinate
        yCCDpixel_center = 100                   # Row coordinate
        subfieldSizeX = 10
        subfieldSizeY = 10

        # Run the simulator for three different magnitudes of the contaminant

        starCatalogName = self.outputDir + "/starCatalog.txt"
        xpixCoord = np.array([xCCDpixel_center, xCCDpixel_center+2])
        ypixCoord = np.array([yCCDpixel_center, yCCDpixel_center])

        SPRs = []
        for n in range(3):

            sim = Simulation(f"test{self.nr}_SPR_two_stars_mag{n}", configurationFile = self.inputfile, outputDir = self.outputDir)

            # Be sure to switch on photometry, and save the star ID in the photomtry file, so that the light curve is extracted

            sim["Photometry/IncludePhotometry"] = "yes"
            sim["Photometry/TargetFileName"] = self.outputDir + "myphotometry.txt"

            # Create a small imagette around the star

            success = sim.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel_center, yCCDpixel_center, subfieldSizeX, subfieldSizeY)
            if not success: 
                eprint("Error: could not set subfield around star")
                return

            # Create a star catalog with the proper star IDs, the magnitudes of the target = 10.5,
            # and the magnitude of the contaminant 10.5, 11.5, and 12.5

            Vmags = np.array([10.5, 10.5+n])
            sim.createStarCatalogFileFromPixelCoordinates(ypixCoord, xpixCoord, Vmags, starIDs, starCatalogName)
            sim["ObservingParameters/StarCatalogFile"] = starCatalogName

            # We only need one exposure to determine the SPR
            
            sim["ObservingParameters/NumExposures"] = 1

            # Run the PlatoSim simulator 

            simFile = sim.run(removeOutputFile=True)
            dummy, dummy, dummy, dummy, dummy, SPR = simFile.getApertureMask(starIDs[0])

            SPRs.append(SPR[0])

        # Verify if they are all between 0 and 1 and that the SPR is decreasing with increasing magnitude of the contaminant
        
        if 1 > SPRs[0] > SPRs[1] > SPRs[2] > 0:
            return True
        else:
            return False



    def run(self) -> bool:
        success1 = self.runSPRTestWithOneStar()
        success2 = self.runSPRTestWithTwoEqualApproachingStars()
        success3 = self.runSPRTestWithTwoUnequalStars()
        return success1 and success2 and success3



if __name__ == "__main__":
    myTest = SPRTest()
    print(myTest.runSPRTestWithOneStar())
    print(myTest.runSPRTestWithTwoEqualApproachingStars())
    print(myTest.runSPRTestWithTwoUnequalStars())


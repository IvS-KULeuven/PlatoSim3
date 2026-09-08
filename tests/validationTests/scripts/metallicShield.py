from test import Test
import platosim.referenceFrames as rf

from math import radians, degrees





"""
This test is developed to check the behavious of the metallic shield for F-cams. We simulate a 600x600 pixel subfield with on it 9(=3x3)
evenly spaces stars. We change the edges of the metallic shield during every run so that the amount of stars that are exposed changes for
every run. The test checks that the correct amount of stars are simulated for the different runs.

This test is repeated for the different PSF types in PlatoSim (MappedFromFile, AnalyticGaussian and AnalyticNonGaussian)
"""





class MetallicShield(Test):

    def setNr(self):
        self.nr = "021"

    def setAllEffects(self):

        # Configure the input file and take only one exposure
        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1

        # Set a 500x500 subfield at the center of the CCD
        self.sim["SubField/ZeroPointRow"]    = 1955
        self.sim["SubField/ZeroPointColumn"] = 1955

        self.sim["SubField/NumRows"]    = 600
        self.sim["SubField/NumColumns"] = 600

        # Set the simulation to have fastcam + shielding
        self.sim["Telescope/GroupID"] = "Fast"
        #self.sim["CCDPositions/MetallicShield/IncludeMetallicShield"] = "yes"

        self.createStarCatalog()





    def createStarCatalog(self):
        # Create a SkyMap with 9 stars evenly spaces over the subfield
        self.position = [100, 300, 500]
        subFieldPositions = [ (i, j) for i in self.position for j in self.position]

        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"

        myFile = open(starCatalogFilename, "w")
        myFile.write("# RA DEC Vmag starID\n")

        n=1
        for (row, col) in subFieldPositions:
            ra, dec = rf.pixelToSkyCoordinates(self.sim, "2F", 1955 + row, 1955 + col)

            ra  = degrees(ra)
            dec = degrees(dec)

            myFile.write("{0}  {1}  {2}  {3}\n".format(ra, dec, 14, n))
            n+=1

        myFile.close()

        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename


    def runSimulation(self):
        
        self.createStarCatalog()
        test2 = self.testForCamera()

        self.sim["PSF/Model"] = "AnalyticNonGaussian"
        self.createStarCatalog()
        test3 = self.testForCamera()

        self.results = (test2 and test3)


    def testForCamera(self):
        positions = [1955, 2155, 2355, 2555]
        stars = []
        
        for miRow in positions:
            for miCol in positions:
                for maRow in positions:
                    for maCol in positions:
                        if (maRow > miRow and maCol > miCol):

                            self.sim["CCDPositions/MetallicShield/ShieldColumnCoordinates"] = [miCol, maCol]
                            self.sim["CCDPositions/MetallicShield/ShieldRowCoordinates"]    = [miRow, maRow]

                            simFile = self.sim.run(removeOutputFile=True)

                            stars.append(self.compareForRun([miCol, maCol, miRow, maRow], simFile.getStarCatalog()))

        return all(stars)


    def amountOfStars(self, miCol, maCol, miRow, maRow):
        starPositions = [ 50, 300, 550 ]
        NStars = 9
        l = 0
        r = 0
        b = 0
        t = 0
        miCol = miCol - 1955
        maCol = maCol - 1955
        miRow = miRow - 1955
        maRow = maRow - 1955

        for star in starPositions:
            if miCol > star:
                l += 1
            if maCol < star:
                r += 1
            if miRow > star:
                b += 1
            if maRow < star:
                t += 1

        NStars = NStars - 3*(l+r) - (t+b)*(NStars - 3*(l+r))/3

        if NStars < 0:
            NStars = 0

        return NStars

    def compareForRun(self, borders, stars):

        miCol, maCol, miRow, maRow = borders

        if all([x is None for x in stars]):
            return self.amountOfStars(miCol, maCol, miRow, maRow) == 0
        else:
            return self.amountOfStars(miCol, maCol, miRow, maRow) == len(stars[0])


    def compare(self):
        return self.results





if __name__ == "__main__":
    t = MetallicShield()
    print(t.run())


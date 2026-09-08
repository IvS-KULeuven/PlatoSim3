import numpy as np
from platosim.scripts.CTImap import createCTIinputFile
from test                    import Test






""" 
This test checks the Short2013FromFile CTI model. This test consists out of 2 parts: 
1. It checks that for a input file with uniform radiation map (filled with all 1), the output is equivallent as would be the case for the Short2013 model. 
2. For a input file, filled with 0 on the left side and 1 on the right size we expect little to no CTI on the left side and CTI on the right side. 
"""

class Short2013CTIFromFile(Test):

    def setNr(self):
        self.nr = "016.3"

    def setAllEffects(self):

        super().setAllEffects()
        # We simulate one exposure at the end of the life of the simulation so that CTI is higher 
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["ObservingParameters/BeginExposureNr"] = 8000000
        self.sim["Sky/SkyBackground/BackgroundValue"] = 100

        self.numRows    = 4500
        self.numColumns = 100
        self.sim["CCD/Position"] = 1

        self.zeroPointRow = 2000
        self.sim["SubField/NumRows"]    = self.numRows
        self.sim["SubField/NumColumns"] = self.numColumns
        self.sim["SubField/ZeroPointColumn"] = self.zeroPointRow
        self.sim["CCD/IncludeCTIeffects"] = "yes"

        # We create a star catalog with evenly spaces stars on the subfield 
        self.rows = np.array([])
        self.columns = np.array([])
        magnitudes = np.array([])

        for i in np.arange(10, self.numColumns, 10):
            self.rows    = np.append(self.rows, np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))
            self.columns = np.append(self.columns, (i) * np.ones(np.size(np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))))
            magnitudes   = np.append(magnitudes, 14*np.ones(np.size(np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))))

        self.sim.createStarCatalogFileFromPixelCoordinates(self.rows, self.columns+self.zeroPointRow, magnitudes, np.arange(1, np.size(self.rows)+1, 1), self.ioPath + "/test" + self.nr + "/starCatalog" + self.nr + ".txt")









    def runSimulation(self):

        # Run with CTI from file with uniform radiation map
        outputFile  = self.ioPath + "/test" + self.nr + "/ctiInput.hdf5"
        self.createUniformInput(outputFile)
        self.sim["CCD/CTI/Model"] = "Short2013FromFile"
        self.sim["CCD/CTI/Short2013FromFile/CTIFileName"] = outputFile
        output = self.sim.run(removeOutputFile=True)
        figure1 = output.getImage(8000000)

        self.sim["CCD/CTI/Model"] = "Short2013"
        output = self.sim.run(removeOutputFile=True)
        figure2 = output.getImage(8000000)

        # Check that these two simulations are equivallent
        self.difference = figure1 - figure2

        # We create a non uniform radiationmap and check that on the side with no/little radiation CTI is much less
        outputFile = self.ioPath + "/test" + self.nr + "/ctiInputNO.hdf5"
        self.sim["CCD/CTI/Short2013FromFile/CTIFileName"] = outputFile

        self.createNonUniformInput(outputFile)
        self.sim["CCD/CTI/Model"] = "Short2013FromFile"
        output = self.sim.run(removeOutputFile=True)
        figure = output.getImage(8000000)

        self.starsWithCTI = [ np.array([[ figure[i][j] for j in np.arange(int(col-10), int(col+10)) ] for i in np.arange(int(row-10), int(row+10))]) for row, col in zip(self.rows, self.columns) ]
        







    def createUniformInput(self, outputFile):

        beta = 0.37
        temperature = 203.0
        meanTrapDensityBOL = np.array([0.0, 0.0, 0.0, 0.0])
        meanTrapDensityEOL = np.array([9.8, 3.31, 1.56, 13.24])
        trapCaptureCrossSection = np.array([2.46e-20, 1.74e-22, 7.05e-23, 2.45e-23])
        releaseTime = np.array([2.37e-4, 2.43e-2, 2.03e-3, 1.40e-1])
        radiationMap = np.ones((4510, 4510))
        createCTIinputFile(outputFile, beta, temperature, meanTrapDensityBOL, meanTrapDensityEOL, trapCaptureCrossSection, releaseTime, radiationMap)







    def createNonUniformInput(self, outputFile):

        beta = 0.37
        temperature = 203.0
        meanTrapDensityBOL = np.array([0.0, 0.0, 0.0, 0.0])
        meanTrapDensityEOL = 1000 * np.array([9.8, 3.31, 1.56, 13.24])
        trapCaptureCrossSection = np.array([2.46e-20, 1.74e-22, 7.05e-23, 2.45e-23])
        releaseTime = np.array([2.37e-4, 2.43e-2, 2.03e-3, 1.40e-1])
        radiationMap = np.ones((4510, 4510))
        for row in radiationMap:
            row[:2050:] = 0
        createCTIinputFile(outputFile, beta, temperature, meanTrapDensityBOL, meanTrapDensityEOL, trapCaptureCrossSection, releaseTime, radiationMap)






    def compare(self):

        condition1 = not np.any(self.difference)
        condition2 = self.compareNonUniformImage()

        return condition1 and condition2







    def compareNonUniformImage(self):

        variance1, variance2 = zip(*[self.getCovariance(star) for star in self.starsWithCTI])
        variance1 = np.array(variance1)
        variance1 = np.resize(variance1, (9,9))
        condition = np.array([np.mean(var) for var in variance1])
        condition = condition / np.mean(condition)
        condition = np.all([ var < 1 if col<=50 else var > 1 for var, col in zip(condition, self.columns[::9])])

        return condition







    def getCovariance(self, star):

        # we return the variance of the star in the vertical direction (column) and the horizontal direction (row) 
        star = star / np.sum(star)
        
        mean1 = np.sum([(i)*np.sum([star[i][j] for j in np.arange(0, 20)]) for i in np.arange(0,20)])
        mean2 = np.sum([(j)*np.sum([star[i][j] for i in np.arange(0, 20)]) for j in np.arange(0,20)])
        
        cov11 = np.sum([(i-mean1)**2 * np.sum([star[i][j] for j in np.arange(20)]) for i in np.arange(20)])
        cov22 = np.sum([np.sum([(j-mean2)**2 * star[i][j] for j in np.arange(20)]) for i in np.arange(20)])

        return cov11, cov22






if __name__ == "__main__":
    t = Short2013CTIFromFile()
    print(t.run())

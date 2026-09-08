import numpy             as np
from test import Test 






""" 
This test checks the Short2013 CTI model. The tests simulates 81 stars, evenly spaced, on a subfield of 4500 rows and 4500 columns.
The test checks that the stars have a tail in the direction of the columns. The test checks that: 
1. the tail increases when we move away from the readout register 
2. for stars that are on the same row the tail is similar 
3. there is no tail in the direction of the rows
"""

class Short2013CTI(Test):
    
    def setNr(self):
        self.nr = "016.2"

    def setAllEffects(self):

        super().setAllEffects()

        # We simulate one exposure at the end of the life of the simulation so that CTI is higher 
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["ObservingParameters/BeginExposureNr"] = 8000000
        self.sim["Sky/SkyBackground/BackgroundValue"] = 100

        self.numRows    = 4500
        self.numColumns = 100
        self.sim["CCD/Position"] = 1

        self.sim["SubField/NumRows"]    = self.numRows
        self.sim["SubField/NumColumns"] = self.numColumns
        self.sim["SubField/ZeroPointColumn"] = 2000

        self.sim["CCD/CTI/Model"] = "Short2013"

        # We create a star catalog with evenly spaces stars on the subfield 
        self.rows = np.array([])
        self.columns = np.array([])
        magnitudes = np.array([])

        for i in np.arange(10, self.numColumns, 10):
            self.rows    = np.append(self.rows, np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))
            self.columns = np.append(self.columns, (i) * np.ones(np.size(np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))))
            magnitudes   = np.append(magnitudes, 14*np.ones(np.size(np.arange(int(self.numRows/10), self.numRows, int(self.numRows/10)))))

        self.sim.createStarCatalogFileFromPixelCoordinates(self.rows, self.columns+2000, magnitudes, np.arange(1, np.size(self.rows)+1, 1), self.ioPath + "/test" + self.nr + "/starCatalog" + self.nr + ".txt")

                                                           


        
    def runSimulation(self):

        self.sim["CCD/IncludeCTIeffects"] = "yes"
        output = self.sim.run(removeOutputFile=True)
        figure1 = output.getImage(8000000)
        figure1 = figure1 - np.min(figure1)

        # We take a 20x20 part of the subfield around each simulated star 
        self.starsWithCTI = [ np.array([[ figure1[i][j] for j in np.arange(int(col-10), int(col+10)) ] for i in np.arange(int(row-10), int(row+10))]) for row, col in zip(self.rows, self.columns) ]
 





        
        
    def compare(self):

        # obtain the variance of the star in the direction of the columns and in the direction of the rows 
        variance1, variance2 = zip(*[self.getCovariance(star) for star in self.starsWithCTI])

        variance1 = np.array(variance1)
        variance1 = np.resize(variance1, (9,9))

        variance2 = np.array(variance2)
        variance2 = np.resize(variance2, (9,9))

        # We check that variance (tail) in the direction of the column increases over the columns when we move away from the readout register
        # and the the variance (tail) in the direction of the column remains similar over one row. 
        increasesOnColumns = [np.all([row[i] < row[i+1] for i in np.arange(np.size(row)-1)]) for row in variance1]
        remainsOnRow       = [ np.std(row) < 0.1 for row in np.transpose(variance1)]

        # Similarly we check that variance in the direction of the rows remains the same over rows and columns
        remainsOnColumns = [np.std(row[1:-1]) < 1 for row in variance2]
        remainsOnRow2     = [np.std(row[1:-1]) < 1 for row in np.transpose(variance2)[1:-1]]

        return np.all(increasesOnColumns) and np.all(remainsOnRow) and np.all(remainsOnColumns) and np.all(remainsOnRow2)

      
      


    def getCovariance(self, star):

        # we return the variance of the star in the vertical direction (column) and the horizontal direction (row) 
        star = star / np.sum(star)
        
        mean1 = np.sum([(i)*np.sum([star[i][j] for j in np.arange(0, 20)]) for i in np.arange(0,20)])
        mean2 = np.sum([(j)*np.sum([star[i][j] for i in np.arange(0, 20)]) for j in np.arange(0,20)])
        
        cov11 = np.sum([(i-mean1)**2 * np.sum([star[i][j] for j in np.arange(20)]) for i in np.arange(20)])
        cov22 = np.sum([np.sum([(j-mean2)**2 * star[i][j] for j in np.arange(20)]) for i in np.arange(20)])

        return cov11, cov22

            
            
        
        
        
        
if __name__ == "__main__":
    t = Short2013CTI()
    print(t.run())

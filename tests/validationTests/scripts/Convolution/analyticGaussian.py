import platosim.referenceFrames as rf
import numpy                    as np

from test                import Test
from platosim.validation import fitGaussian1D



"""
This test checks that the spread of the anaytic Gaussian PSF increases linearly with distance from the optical axis. 
"""










class AnalyticGaussianPSF(Test):

    def setNr(self):

        self.nr = "008.3"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["Telescope/AzimuthAngle"]           = 0
        self.sim["Telescope/TiltAngle"]              = 0
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeConvolution"]           = "yes"

        #self.sim["ControlHDF5Content/WriteSubPixelImages"] = "yes"
        self.sim["PSF/Model"] = "AnalyticGaussian"

        self.sigma0   = 5
        self.sigmaX18 = 30
        self.sigmaY18 = 10
        self.sim["PSF/AnalyticGaussian/Sigma00"]  = self.sigma0
        self.sim["PSF/AnalyticGaussian/SigmaX18"] = self.sigmaX18
        self.sim["PSF/AnalyticGaussian/SigmaY18"] = self.sigmaY18

        self.sim["SubField/NumRows"]    = 4510
        self.sim["SubField/NumColumns"] = 4510

        # Create a star catalog file with multiple stars aligned over the diagonal of the subfield. 
        starCatalogFilename = self.outputDir + "/starCatalogFile"+ self.nr + ".txt"
        
        self.xPosition = np.arange(200, 3010, 400)
        self.yPosition = np.arange(4310, 1500, -400)
                
        xPosition = self.xPosition
        yPosition = self.yPosition
        
        magnitude = 12.5 * np.ones(len(xPosition))
        index     = np.arange(1, len(xPosition) + 1)

        self.sim.createStarCatalogFileFromPixelCoordinates(yPosition, xPosition, magnitude, index, starCatalogFilename)
        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename






    def runSimulation(self):
        # Run the simulation
        simFile    = self.sim.run(removeOutputFile = True)
        self.image = simFile.getImage(0)



    def compare(self):
        # The stars in the image are isolated in seperate subimages of 400 x 400 subimage.
        # The one dimensional spread of these stars into the x and y direction  are then saved into
        # seperate lists. 
        
        
        xPos        = self.xPosition[1:]
        yPos        = self.yPosition[1:]
        image       = self.image

        focalLength = self.sim["Camera/FocalLength/ConstantValue"]*1000
        pixelSize   = self.sim["CCD/PixelSize"]

        
        stars       = [np.array([[image[y+i-200][x+j-200] for i in range(400)] for j in range(400)]) for x, y in zip(xPos, yPos) ]
        xSpread     = np.array([[star[i][i] for i in range(400)] for star in stars])
        ySpread     = np.array([[star[i][399-i] for i in range(400)] for star in stars])

        # The sigma for these spread is then estimated using the fitGaussian1D function. 
        xParameters = [abs(fitGaussian1D(x, np.max(x)-np.min(x), 200, 25)[-1]) for x in xSpread]
        yParameters = [abs(fitGaussian1D(y, np.max(y)-np.min(y), 200, 5)[-1])  for y in ySpread]

        xFP, yFP = rf.pixelToFocalPlaneCoordinates(xPos, yPos, pixelSize, -1.3, 82.48, 3/2*np.pi)
        gnomic   = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, focalLength))

        # The fitted sigma is then compared versus a linear extrapolition between sigma0 and sigma18. The test passes when RMS
        # difference between these two is smaller then 0.01.
        xFunction   = self.sigma0 + (gnomic / 18) * (self.sigmaX18 - self.sigma0)
        yFunction   = self.sigma0 + (gnomic / 18) * (self.sigmaY18 - self.sigma0)

        xSSquare    = np.sum(np.array([(x - y)**2 for x, y in zip(xFunction, xParameters)]))
        ySSquare    = np.sum(np.array([(x - y)**2 for x, y in zip(yFunction, yParameters)]))
        
        xRMS        = np.sqrt(xSSquare / len(xFunction))
        yRMS        = np.sqrt(ySSquare / len(yFunction))

        return xRMS < 0.01 and yRMS < 0.01



        
        






if __name__ == "__main__":
    t = AnalyticGaussianPSF()
    print(t.run())

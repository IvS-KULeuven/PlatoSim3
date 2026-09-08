from test import Test
from platosim.validation import fitGaussian2D

import numpy as np

import matplotlib.pyplot as plt



"""
This test is designed to check the Brighter-Fatter Effect. This checks two behaviours of the BFE in particular. 

1. The first test checks that the BFE is applied: 

The simulation is run for a star in the middle of 9x9 pixel subfield for one exposure, both with and without the BFE effect. This is repeated 
for increasing magnitudes of the star. The test passes if the difference in width of the star between the two simulations, decreases with increasing 
magnitude and that the width of the star is independent of the magnitude for the simulations without the BFE effect.

2. The second tests cheks that the BFE effects the photon transfer curve. Because of the BFE, the photon transfer curve for an empty subfield, with 
increasing skyback background no longer happens linear. The affect reduces the variance of the pixels, and we end up with a concave increasing function instead.  
"""











class BrighterFatterEffect(Test):

    def setNr(self):
        self.nr = "013"
    
    def setAllEffects(self):

        # Set the input parameters for the first test
        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeConvolution"]           = "yes"

        self.dim                        = 20
        self.sim["SubField/NumRows"]    = self.dim
        self.sim["SubField/NumColumns"] = self.dim

        self.sim["ControlHDF5Content/WriteSubPixelImages"] = "yes"
        self.numSubPixels = self.sim["SubField/SubPixels"]

        #self.sim["PSF/Model"] = "MappedFromFile"






    def setEffectsForSecondTest(self):

        # Set the input paramters for the second test
        super().setAllEffects()
        self.sim["CCD/IncludeQuantisation"] = "yes"
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeConvolution"]           = "yes"

        self.sim["SubField/NumRows"]    = 1000
        self.sim["SubField/NumColumns"] = 1000

        self.sim["CCD/IncludePhotonNoise"] = "yes"
        self.sim["CCD/IncludeBFE"]         = "yes"

        # No star in the subfield.
        self.sim["Platform/Orientation/Angles/DecPointing"] = - self.sim["Platform/Orientation/Angles/DecPointing"]



    def runSimulation(self):
        # Run the first test
        self.runFirstTest()

        # Set the paramters and run the second test
        self.setEffectsForSecondTest()
        self.runSecondTest()




    def runFirstTest(self):

        # Define list of the different magnitudes of the star and their position on the subfield
        magnitudes = np.arange(8, 20, 0.5)
        position   = np.array([self.dim / 2])

        # Specify the star catalog file name
        starCatalogFilename = self.outputDir + "/starCatalog" + self.nr + ".txt"
        #self.sigma           = self.sim["PSF/MappedGaussian/Sigma"]

        widthWithBFE     = np.array([]), np.array([])
        widthWithoutBFE  = np.array([]), np.array([])
        
        for mag in magnitudes:
            # Run the simulation for stars with increasing magnitude, centered on the subfield.

            # Create the star catalog file            
            self.sim.createStarCatalogFileFromPixelCoordinates(position, position, np.array([mag]), np.array([1]), starCatalogFilename)

            # Run the simulation with BFE and add the width (sigma of a gaussian fit) to widthWithBFE.
            self.sim["CCD/IncludeBFE"] = "yes"
            widthWithBFE = self.storeWidth(widthWithBFE)
            
            # Run the similation without BFE and add the width (sigma of gaussian fit) to widthWithoutBFE.
            self.sim["CCD/IncludeBFE"] = "no"
            widthWithoutBFE = self.storeWidth(widthWithoutBFE)

        self.widthWithBFE    = widthWithBFE
        self.widthWithoutBFE = widthWithoutBFE





    def runSecondTest(self):

        # Run the test for increasing sky backgrounds and store the corresponding mean and standard deviation of the image in
        # self.means and self.stds.
        backgrounds = np.arange(0, 50000, 9000)                              # [ph/pix/s]
        means       = np.array([])
        stds        = np.array([])

        for background in backgrounds:

            self.sim["Sky/SkyBackground/BackgroundValue"] = background
            simFile = self.sim.run(removeOutputFile = True)
            image = simFile.getImage(0)
            stds  = np.append(stds, np.std(image)**2)
            means = np.append(means, np.mean(image))

        self.means = means
        self.stds  = stds




        





    def storeWidth(self, width):
        # Runs the simulator with the specific input paramters and fits the output image to 2D gaussian.
        # The sigma of this fit gets added to the input width array.
        
        #sigma    = self.sigma
        simFile  = self.sim.run(removeOutputFile = True)
        image    = simFile.getImage(0)

        params   = fitGaussian2D(image, np.max(image), self.dim / 2, self.dim / 2, 1, 1, subtractConstant = True)
        width    = np.append(width, params[3])


        return width
        
        
            
    def compare(self):
        # Check the results of the two test
        
        stds  = self.stds
        means = self.means

        # Checks that brighter star are fatter

        difference = self.widthWithBFE - self.widthWithoutBFE
        isFatter   = all(x or y for x,y in zip(difference[:-1] > difference[1:],
                                               abs(difference[:-1] - difference[1:]) < 1e-5))

        # Checks that without BFE the variance remains approximately constant
        varWithout  = np.std(self.widthWithoutBFE)
        withoutIsConst = varWithout < 0.3
        condition1  =  isFatter and withoutIsConst

        # Checks that the phton trasfer curve is increasing (condition2a) in a concave (condition2b) manner.
        condition2a = all([stds[n] > stds[n-1] for n in range(1, len(stds))])
        condition2b = all([ (stds[n] - stds[n-1]) * (means[n+1] - means[n]) > (stds[n+1] - stds[n]) * (means[n] - means[n-1]) for n in range(1, len(means) - 1)])
        condition2  = condition2a and condition2b

        # Save the plot of the photon transfer curve
        self.makePlot(stds, means)
        return condition1 and condition2

    

    def makePlot(self, stds, means):
        # Make and save plot of the photon trasfer curve
        
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(1, 1, 1)

        plt.title("Photon Transfer Curve (PTC)", fontsize = 32)
        plt.plot( means, stds, "bx")
        plt.plot( means, means * (stds[1] - stds[0]) / (means[1] - means[0]) + stds[0] - means[0] *  (stds[1] - stds[0]) / (means[1] - means[0]), "r")

        plt.xlabel("Mean flux", fontsize = 24)
        plt.ylabel("Variance of flux", fontsize = 24)
        
        plt.savefig(self.outputDir + '/PhotonTransferCurve.pdf')
 

        

       





        


if __name__ == "__main__":
    t = BrighterFatterEffect()
    print(t.run())

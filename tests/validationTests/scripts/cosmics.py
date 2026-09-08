from test    import Test
from skimage import morphology
import numpy as np
import h5py





"""
This test is designed to check the behaviour of the cosmics in the PlatoSIM simulator. The test runs the simulation three times,
once without cosmic hits, once with a lower cosmic hit rate and trail length and once with a higher cosmic hit rate and trail length. The 
conditions for this test to pass are: 
 - The observed amount of cosmic hits and trail length matches the one expected from the input parameters.
 - The observed amount of cosmic hits and trail length increases w.r.t. increasing CosmicHitRate- and TrailLength-parameters. 
 - The recorded cosmics in the output.hdf5 file watches the expected one.  
"""







class Cosmics(Test):

    def setNr(self):
        self.nr = "014"




    def setAllEffects(self):

        super().setAllEffects()

        self.numExposures = 1
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["SubField/NumRows"] = 1
        self.sim["SubField/NumColumns"] = 1

        self.sim["Platform/Orientation/Angles/DecPointing"] = - self.sim["Platform/Orientation/Angles/DecPointing"]


    def runSimulation(self):

        self.sim["Sky/IncludeCosmicsInSubField"] = "no"
        self.simFileWithoutCos = self.sim.run(removeOutputFile = True)



        self.numExposures = 100
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["SubField/NumRows"] = 400
        self.sim["SubField/NumColumns"] = 400


        self.sim["Sky/Cosmics/CosmicHitRate"] = 10
        self.sim["Sky/Cosmics/TrailLength"][1] = 10
        self.sim["Sky/IncludeCosmicsInSubField"] = "yes"
        self.sim["Sky/IncludeCosmicsInSmearingMap"] = "yes"
        self.sim["Sky/IncludeCosmicsInBiasMap"] = "yes"
        self.simFileWithCos1 = self.sim.run(removeOutputFile = True)

        self.sim["Sky/Cosmics/CosmicHitRate"] = 30
        self.sim["Sky/Cosmics/TrailLength"][1] = 20
        self.simFileWithCos2 = self.sim.run(removeOutputFile = True)



    def compare(self):
        condition1, mean1, trail1  = self.getAmountOfCosmicsAndTrailLength(self.simFileWithCos1, self.simFileWithoutCos, 10, 10)
        condition2, mean2, trail2  = self.getAmountOfCosmicsAndTrailLength(self.simFileWithCos2, self.simFileWithoutCos, 30, 20)
        condition3         = mean1 < mean2
        condition4         = trail1 < trail2
        condition5         = self.checkOutPut()
        return condition1 and condition2 and condition3 and condition4 and condition5






    def getAmountOfCosmicsAndTrailLength(self, fileWithCosmics, fileWithoutCosmics, cosmicHitRate, trailLengthParameter):
                 numIsland1  = np.array([])
                 numIsland2  = np.array([])
                 trailLength = np.array([])

                 numWithoutCosmics = 0

                 noCosmics = np.full(shape=fileWithCosmics.getImage(0).shape,
                                     fill_value=fileWithoutCosmics.getImage(0)[0,0])
                 for exp in range(self.numExposures):

                     diff = fileWithCosmics.getImage(exp) - noCosmics
                     diff[diff!=0] = 1

                     if (np.sum(diff) != 0):
                         trailLength = np.append(trailLength, np.sum(diff))

                         labels, num1 = morphology.label(diff, connectivity=1, return_num=True)
                         numIsland1 = np.append(numIsland1, num1)

                         labels, num2 = morphology.label(diff, connectivity=2, return_num=True)
                         numIsland2 = np.append(numIsland2, num2)
                     else:
                         numWithoutCosmics += 1


                 trailLength1       = np.mean(np.concatenate((trailLength / numIsland2 / np.sqrt(2), np.zeros(numWithoutCosmics)), axis=0))
                 trailLength2       = np.mean(np.concatenate((trailLength / numIsland1 / np.sqrt(2), np.zeros(numWithoutCosmics)), axis=0))
                 theoTrail          = (trailLengthParameter) / 2
                 
                 minTrail           = min(trailLength1, trailLength2)
                 maxTrail           = max(trailLength1, trailLength2)
                 condition1         = minTrail < theoTrail < maxTrail
                 
                 cycleTime          = self.sim["ObservingParameters/CycleTime"]
                 numPixels          = self.sim["SubField/NumRows"] * self.sim["SubField/NumColumns"]
                 pixelSize          = self.sim["CCD/PixelSize"] / 10000.0
                 
                 
                 meanCosmics        = cosmicHitRate * cycleTime * numPixels * pixelSize**2
                 mean1Island        = np.mean(numIsland1)
                 mean2Island        = np.mean(numIsland2)
                 
                 minMean            = min(mean1Island, mean2Island)
                 maxMean            = max(mean1Island, mean2Island)
                 condition2         = minMean < meanCosmics < maxMean
                 
                 return condition1 and condition2, meanCosmics, theoTrail
                 
                 
                 
                 
                 
    def checkOutPut(self):
                 
        image1 = self.simFileWithCos1.getImage(0) - self.simFileWithoutCos.getImage(0)
        row, col, flux = self.simFileWithCos1.getCosmicsAffectedPixels(0)
                 
        image2 = np.zeros((1000, 1000))
        for c, r, f in zip(col, row, flux):
            image2[r][c] = f
                 
        condition = all([all([pnt1 == 0. if pnt2 == 0. else True for pnt1, pnt2 in zip(col1, col2)]) for col1, col2 in zip(image1, image2)])
        return condition
                 
                 
                 
                 
if __name__ == "__main__":
    t = Cosmics()
    print(t.run())

                 

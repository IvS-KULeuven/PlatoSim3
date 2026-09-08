import math
import numpy             as np
import matplotlib.pyplot as plt

from test                import Test
from platosim.validation import switchOffAllEffects




class StellarVariability(Test):


    def setNr(self):

        self.nr = "002"



    def setAllEffects(self):

        switchOffAllEffects(self.sim)
        # Define star catalog and input file for variable stars

        starCatalogFileName = self.outputDir + "/starCatalog" + self.nr + ".txt"
        variabilityFileName = self.outputDir + "/variability.txt"
        varSourceFileName   = self.outputDir + "/varsource.txt"

        dim = 100
        position       = np.array([4000])
        self.magnitude = np.array([12.5])
        self.sim.createStarCatalogFileFromPixelCoordinates(position, position, self.magnitude, np.array([1]), starCatalogFileName)


        period   = 20 * self.sim["ObservingParameters/CycleTime"]
        time     = np.arange(3000)
        self.sin = lambda t : np.sin(2 * math.pi * t / period)

        varFile = open(variabilityFileName, "w")
        varFile.writelines([str(t) + " " + str(self.sin(t)) + "\n" for t in time])
        varFile.close()

        varSourceFile = open(varSourceFileName, "w")
        varSourceFile.write("{0} {1}\n".format(1, variabilityFileName))
        varSourceFile.close()

        #Configure the input file

        self.sim["ObservingParameters/NumExposures"]    = 100

        self.sim["SubField/NumRows"]                    = 4510
        self.sim ["SubField/NumColumns"]                = 4510

        self.sim["Sky/VariableSourceList"]              = varSourceFileName

        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFileName
        self.sim["Sky/IncludeVariableSources"]          = "yes"

        self.sim["SubField/ZeroPointRow"]    = position[0] - dim // 2
        self.sim["SubField/ZeroPointColumn"] = position[0] - dim // 2
        self.sim["SubField/NumRows"]         = dim
        self.sim["SubField/NumColumns"]      = dim

        self.sim["FEE/Gain/RefValueLeft"]    = 1.0
        self.sim["FEE/Gain/RefValueRight"]   = 1.0
        self.sim["CCD/Gain/RefValueLeft"]    = 1.0
        self.sim["CCD/Gain/RefValueRight"]   = 1.0



    def compare(self):


        # We calculate the output magnitude from the Fluxes obtained from the simulation. This is then compared to the
        # input magnitudes. The test succeeds if the root mean square difference between the two values is smaller than 0.01

        cycleTime       = self.sim["ObservingParameters/CycleTime"]
        exposureTime    = cycleTime - self.sim.getReadoutTime()[0]
        fluxM0          = self.sim["ObservingParameters/Fluxm0"]
        throughput      = self.sim["Camera/ThroughputBandwidth"]
        lightArea       = self.sim["Telescope/LightCollectingArea"]
        transmissionBOL = self.sim["Telescope/TransmissionEfficiency/BOL"]


        nExp            = self.sim["ObservingParameters/NumExposures"]
        fluxes          = np.array([self.simFile.getStarCoordinates(exposure)[5][0] for exposure in range(nExp)])
        outputMagnitude = -2.5 * np.log10(fluxes / (exposureTime * fluxM0 * throughput * lightArea * transmissionBOL * 1E-4))

        theoryMagnitude = [self.sin(t*cycleTime) + self.magnitude[0] for t in range(len(outputMagnitude))]


        RMS             = math.sqrt(sum(((out - theo )**2 for out, theo in zip(outputMagnitude, theoryMagnitude))))
        return RMS < 0.001


if __name__ == "__main__":
    t = StellarVariability()
    print(t.run())

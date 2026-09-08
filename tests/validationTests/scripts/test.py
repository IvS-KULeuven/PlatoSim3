import sys
import os
import random

from platosim.simulation import Simulation
from platosim.validation import switchOffAllEffects


def eprint(*args, **kwargs):
    """ 
    print to stderr rather than to stdout. Useful to debug since all tests are ran with stdout suppressed. 
    """
    print(*args, file=sys.stderr, **kwargs)





class Test:

    def __init__(self):

        # For every test that runs, a folder is created that stores the input, output
        # and star catalog file so that the test can be directly run in c++.


        self.setNr()
        self.outputDir = os.environ["PLATO_PROJECT_HOME"] + "/tests/validationTests/ioFiles/test" + self.nr
        self.inputDir  = os.environ["PLATO_PROJECT_HOME"] + "/inputfiles"
        self.ioPath   = os.environ["PLATO_PROJECT_HOME"] + "/tests/validationTests/ioFiles"
        if not os.path.isdir(self.ioPath):
            os.mkdir(self.ioPath)
            
        if not os.path.isdir(self.outputDir):
            os.mkdir(self.outputDir)

        self.sim = Simulation("test" + self.nr, outputDir = self.outputDir)
        self.setAllEffects()


    def setNr(self):
        self.nr = ""


    def setAllEffects(self):
        switchOffAllEffects(self.sim)
        self.sim["Telescope/GroupID"] = "2"
        self.sim["CCD/Position"]      = "2"
        self.sim["PSF/Model"]         = "AnalyticNonGaussian"
        self.sim["Sky/SkyBackground/BackgroundValue"] = 342.

        # Configure the angles of the input file

        n      = self.sim["Telescope/GroupID"]
        anglAZ = self.sim["CameraGroups/AzimuthAngle"]
        self.sim["Telescope/AzimuthAngle"] = anglAZ[n-1]

        anglTi = self.sim["CameraGroups/TiltAngle"]
        self.sim["Telescope/TiltAngle"] = anglTi[n-1]



    def runSimulation(self):
        # Run PlatoSIM
        self.simFile = self.sim.run(removeOutputFile=True)


    def compare(self):
        return None


    def run(self):
        self.runSimulation()
        return self.compare()


import random
import numpy                    as np
import platosim.referenceFrames as rf

from test                import Test
from math                import degrees
from platosim.validation import switchOffAllEffects


"""
This test is designed the check the jitter from a given file. The test first generated three files, with only a change in  pitch, yaw or roll.
For a linear jitter in the pitch or yaw direction, the path that the star follows should be a straight line.
For a change in roll, we the path should follow a circle around the optical axis.
"""






class JitterFromFile(Test):

    def setNr(self):
        self.nr = "006.2"

    def setAllEffects(self):

        #Configure the input file.
        super().setAllEffects()
        self.numExposures                                 = 60
        readoutTime                                       = self.sim.getReadoutTime()[0]
        self.sim["ObservingParameters/NumExposures"]      = self.numExposures
        self.sim["Platform/UseJitter"]                    = "yes"
        self.sim["Platform/JitterSource"]                 = "FromFile"
        self.sim["ControlHDF5Content/WriteStarPositions"] = "yes"
        self.sim["SubField/NumRows"]                      = 1000
        self.sim["SubField/NumColumns"]                   = 1000
        self.sim["ObservingParameters/CycleTime"]         = readoutTime + 1.

        # Pick a random point in the sub field and write it to a star catalog.
        nRows    = self.sim["SubField/NumRows"]
        nColumns = self.sim["SubField/NumColumns"]


        self.pRows    = 250
        self.pColumns = 250


        # Create a SkyMap with a star that would fall onto that point in the sub field.
        raIn, decIn = rf.pixelToSkyCoordinates(self.sim, "2", self.pColumns, self.pRows)

        raIn  = degrees(raIn)
        decIn = degrees(decIn)


        starCatalogFilename = self.outputDir + "/starCatalog"+ self.nr + ".txt"
        myFile = open(starCatalogFilename, "w")
        myFile.write("# RA DEC Vmag starID\n")
        myFile.write("{0}  {1}  {2}  {3}\n".format(raIn, decIn, 16.5, 1))
        myFile.close()

        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename
        self.sim["Platform/JitterTimeScale"] = 25
        self.createJitterFiles()




    def createJitterFiles(self):
        # Create three jitter files. Each file has only a change in Pitch, Roll or Yaw.
        jitterTimeScale     = 25.
        jitter              = -300.

        numJitterSteps      = int(25 / jitterTimeScale) * 500
        time               = np.arange(numJitterSteps) * jitterTimeScale
        noChange           = np.zeros(numJitterSteps)
        change             = np.arange(numJitterSteps) * jitter

        jitterFilenameYaw   = self.outputDir + "/yawJitter.txt"
        jitterFilenamePitch = self.outputDir + "/pitchJitter.txt"
        jitterFilenameRoll  = self.outputDir + "/rollJitter.txt"

        changeYaw          = -change
        changePitch        = change * 0.9
        changeRoll         = change * 100

        np.savetxt(jitterFilenameYaw, np.c_[time, changeYaw, noChange, noChange])
        np.savetxt(jitterFilenamePitch, np.c_[time, noChange, changePitch, noChange])
        np.savetxt(jitterFilenameRoll, np.c_[time, noChange, noChange, changeRoll])

        self.jitterFile = {"Yaw" : jitterFilenameYaw, "Pitch" : jitterFilenamePitch, "Roll" : jitterFilenameRoll}





    def runSimulation(self):

        self.rowYaw, self.colYaw     = self.runYaw()
        self.rowPitch, self.colPitch = self.runPitch()
        self.rowRoll, self.colRoll   = self.runRoll()




    def runYaw(self):
        # The simulation is run from from the jitter input file where only the Yaw is changed.
        # The function returns the position of the star on the sub field.

        numEx      = self.sim["ObservingParameters/NumExposures"]
        path       = self.jitterFile["Yaw"]
        self.sim["Platform/JitterFileName"] = path
        simFile    = self.sim.run(removeOutputFile = True)

        pos        = [simFile.getStarCoordinates(exp)[1:3] for exp in range(numEx)]
        return zip(*pos)




    def runPitch(self):
        # The simulation is run from from the jitter input file where only the Pitch is changed.
        # The function returns the position of the star on the sub field.

        numEx      = self.sim["ObservingParameters/NumExposures"]
        path       = self.jitterFile["Pitch"]
        self.sim["Platform/JitterFileName"] = path
        simFile    = self.sim.run(removeOutputFile = True)

        pos        = [simFile.getStarCoordinates(exp)[1:3] for exp in range(numEx)]
        return zip(*pos)





    def runRoll(self):
        # The simulation is run from from the jitter input file where only the Roll is changed.
        # Before the simulation is run, the telescope and CCD is reset to custom, so that the path
        # of the star is a circle on the sub field. The function returns the position of the star
        # on the sub field.

        self.sim["Telescope/GroupID"]      = "Custom"
        self.sim["Telescope/AzimuthAngle"] = 0
        self.sim["Telescope/TiltAngle"]   = 0
        self.sim["CCD/Position"]           = "Custom"

        self.sim["ObservingParameters/NumExposures"]      = 50
        self.sim["Platform/UseJitter"]                    = "yes"
        self.sim["Platform/JitterSource"]                 = "FromFile"
        self.sim["ControlHDF5Content/WriteStarPositions"] = "yes"




        numEx  = self.sim["ObservingParameters/NumExposures"]
        path   = self.jitterFile["Roll"]
        self.sim["Platform/JitterFileName"] = path

        # A new star input file is written so that the star fall into the subfield.
        starCatalogFilename = self.outputDir + "/starCatalogRoll"+ self.nr + ".txt"
        self.sim.createStarCatalogFileFromPixelCoordinates(np.array([0]), np.array([10]), np.array([12.5]), np.array([1]), starCatalogFilename)
        self.sim["ObservingParameters/StarCatalogFile"] = starCatalogFilename


        simFile = self.sim.run(removeOutputFile = True)
        pos     = [simFile.getStarCoordinates(exp)[1:3] for exp in range(numEx-1)]

        # Remove values for when no stars fall on the subfield
        pos     = [(x, y) for x, y in pos if not ((x is None) and (y is None))]

        return zip(*pos)





    def compare(self):


        # We check that the the rows and columns for a change in Yaw and Pitch follows
        # a linear path. The test passes if the root mean square difference are smaller then 0.1.
        rowYaw      = np.array([x for [x] in self.rowYaw])
        colYaw      = np.array([x for [x] in self.colYaw])
        RMS         = self.getRMS(rowYaw, colYaw)
        resultYaw   = RMS < 0.1

        rowPitch    = np.array([x for [x] in self.rowPitch])
        colPitch    = np.array([x for [x] in self.colPitch])
        RMS         = self.getRMS(rowPitch, colPitch)
        resultPitch = RMS < 0.1

        # We check that the path the star follows is a circle. This is done by calculating
        # the radius. The test passes if the root mean square between the radius and the
        # mean radius is smaller then 0.1.
        rowRoll     = np.array([x for [x] in self.rowRoll])
        colRoll     = np.array([x for [x] in self.colRoll])
        radius      = np.sqrt(rowRoll * rowRoll + colRoll * colRoll)
        RM          = (radius - np.mean(radius)).dot(radius - np.mean(radius)) / len(radius)
        RMS         = np.sqrt(RM)
        resultRoll  = RMS < 0.1

        return resultYaw and resultPitch and resultRoll





    def getSlopeBias(self, values):
        # This function calculates the best linear fit to the data given in the input numpy array and
        # return this fit as a linear function.

        length = len(values)
        time   = np.arange(length)
        slope = (values - np.mean(values)).dot(time - np.mean(time)) / (time - np.mean(time)).dot(time - np.mean(time))
        bias   = (np.sum(values) - slope * np.sum(time))/length
        return lambda x : bias + x * slope




    def getRMS(self, input1, input2):
        # This function calculates the root mean square difference between two numpy arrays and their linear extrapolation
        # as given by function getSlopeBias.

        fun1   = self.getSlopeBias(input1)
        fun2   = self.getSlopeBias(input2)

        time1  = np.arange(len(input1))
        time2  = np.arange(len(input2))
        theo   = np.array([fun1(x) for x in time1])
        theo2  = np.array([fun2(x) for x in time2])

        MS     = ((input1 - theo).dot(input1 - theo) + (input2 - theo2).dot(input2 - theo2)) / (len(input1) + len(input2))
        return np.sqrt(MS)




if __name__ == "__main__":
    t = JitterFromFile()
    #t.runSimulation()
    print(t.run())

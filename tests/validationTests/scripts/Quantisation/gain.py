from test import Test
import numpy as np
import matplotlib.pyplot as plt


class Gain(Test):
    """This class is designed to test the gain of PlatoSIM simulator. The first tests checks the gain at the nominal 
    temperature. The test runs PlatoSim with and without gain (by switching IncludeQuantisation on/off). It then checks 
    if multiplying the image without the gain, by the gain (obtained from the input file) is similar to the image with
    the gain. 
    Secondly, it checks the temperature dependence of the left and right CCD and the left and right FEE. This is done by 
    increasing the temperature of the CCD or FEE  and checking that the gain decreases along the theorectically expected 
    curve. This test is done for the left and right side of the CCD or FEE.  """


    

    def setNr(self):
        self.nr = "020.1"


        
    def setAllEffects(self):

        super().setAllEffects()

        self.sim["CCD/IncludeDigitalSaturation"]   = "yes"
        self.sim["FEE/ElectronicOffset/RefValue"]  = 0
        self.sim["FEE/ElectronicOffset/Stability"] = 0
        cycleTime = self.sim["ObservingParameters/CycleTime"]

        # Make sure there are no stars on the subfield.
        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]

        # Create the termperature file for the FEE and the CCD and points to the right directory. 
        time = np.array([0, 100*cycleTime])
        tempFileNameFEE = self.outputDir + "/temperatureFEE.txt"
        self.temperatureFEE  = np.array([self.sim["FEE/NominalOperatingTemperature"], self.sim["FEE/NominalOperatingTemperature"] + 10])
        self.sim["FEE/TemperatureFileName"] = tempFileNameFEE
        np.savetxt(tempFileNameFEE, np.c_[time, self.temperatureFEE])

        tempFileNameCCD = self.outputDir + "/temperatureCCD.txt"
        self.temperatureCCD  = np.array([self.sim["CCD/NominalOperatingTemperature"], self.sim["CCD/NominalOperatingTemperature"] + 50]) 
        self.sim["CCD/TemperatureFileName"] = tempFileNameCCD
        np.savetxt(tempFileNameCCD, np.c_[time, self.temperatureCCD])
        


        


    def runAtNominalTemperature(self):

        self.sim["SubField/NumRows"]    = 1000
        self.sim["SubField/NumColumns"] = 1000

        # Run the simulation without Gain
        self.sim["CCD/IncludeQuantisation"] = "no"
        simFileWithoutGain = self.sim.run(removeOutputFile = True)
        outputWithoutGain = simFileWithoutGain.getImage(0)

        # Run the simulation with Gain
        self.sim["CCD/IncludeQuantisation"] = "yes"
        simFileWithGain    = self.sim.run(removeOutputFile = True)
        outputWithGain = simFileWithGain.getImage(0)

        # Returns the images with without gain
        return outputWithGain, outputWithoutGain


    

    def runTemperatureDependencyOfFEE(self):

        # Decrease the size of the subfield so that it consists out of 1 pixel
        self.sim["SubField/NumRows"]    = 1
        self.sim["SubField/NumColumns"] = 1

        # Include temperature variation for FEE
        self.sim["FEE/Temperature"] = "FromFile"

        # Runs the simulation
        return self.sim.run(removeOutputFile = True)
        

    

    def runTemperatureDependencyOfCCD(self):

        # Sets the size of the subfield so that it consists out of 1 pixel
        self.sim["SubField/NumRows"]    = 1
        self.sim["SubField/NumColumns"] = 1

        # Include temperature variation for the CCD 
        self.sim["FEE/Temperature"]     = "Nominal"
        self.sim["CCD/Temperature"]     = "FromFile"

        # Runs the simulation
        return self.sim.run(removeOutputFile = True)

    

    

    def runSimulation(self):

        # Runs the simulation for the gain at nominal temperature
        self.sim["ObservingParameters/NumExposures"] = 1
        self.outputWithGain, self.outputWithoutGain = self.runAtNominalTemperature()

        # Inceases the amount of exposures to 100
        self.sim["ObservingParameters/NumExposures"] = 100

        # Runs the test for the temperature dependancy of the left FEE
        self.leftSimFileFEE  = self.runTemperatureDependencyOfFEE()
        self.sim["SubField/ZeroPointColumn"] = 3000
        # Runs the test for the temperature dependancy of the right FEE
        self.rightSimFileFEE = self.runTemperatureDependencyOfFEE()

        # Runs the test for the temperature dependancy fo the left CCD
        self.sim["SubField/ZeroPointColumn"] = 0
        self.leftSimFileCCD  = self.runTemperatureDependencyOfCCD()
        # Runs the test for the temperature dependancy fo the right CCD
        self.sim["SubField/ZeroPointColumn"] = 3000
        self.rightSimFileCCD = self.runTemperatureDependencyOfCCD()

        
        
        
    def compareAtNominalTemparture(self):

        # Calculates the theoretical gain
        saturation = self.sim["CCD/DigitalSaturation"]
        gain       = self.sim["FEE/Gain/RefValueLeft"] * self.sim["CCD/Gain/RefValueLeft"]

        
        quotient   = np.floor(self.outputWithoutGain * gain) / self.outputWithGain
        # We only include those pixels that are not saturated
        quotient   = quotient[self.outputWithoutGain < saturation]

        # The condition passes if all the elements in quotient are around the value 1 
        condition1 = (np.max(quotient) - np.min(quotient)) < 0.01 and abs(np.mean(quotient) - 1) < 0.01

        return condition1


    
    def compareFEE(self, simFile, position):

        # That returns the correct bias from the input file, dependent on where we are on the FEE
        gainRefValue = {'left' : self.sim["FEE/Gain/RefValueLeft"], 'right' : self.sim["FEE/Gain/RefValueRight"] }[position]
        numExposures = self.sim["ObservingParameters/NumExposures"]

        # FEE gain from PlatoSim
        feeGain = np.array([simFile.getImage(exposure)[0][0] for exposure in range(numExposures)])
        feeGain =  feeGain / feeGain[0] * numExposures

        # Predicted slope from the input file
        theoSlope = (self.temperatureFEE[-1] - self.temperatureFEE[0])  * self.sim["FEE/Gain/Stability"] / gainRefValue
        # Best fit for the slope based on the input data from PlatoSim
        slope     = self.getBestSlope(np.arange(numExposures), feeGain)

        return abs((slope - theoSlope) / theoSlope) < 0.01
        




    def compareCCD(self, simFile, position):

        # That returns the correct bias from the input file, dependent on where we are on the CCD
        gainRefValue = {'left' : self.sim["CCD/Gain/RefValueLeft"], 'right' : self.sim["CCD/Gain/RefValueRight"] }[position]
        numExposures = self.sim["ObservingParameters/NumExposures"]

        # CCD gain from PlatoSim
        ccdGain = np.array([simFile.getImage(exposure)[0][0] for exposure in range(numExposures)])
        ccdGain = ccdGain / ccdGain[0] * numExposures

        # Predicted slope from the input file
        theoSlope = (self.temperatureCCD[-1] - self.temperatureCCD[0])  * self.sim["CCD/Gain/Stability"] / gainRefValue
        # Best fit for the slope based on the input data from PlatoSim
        slope     = self.getBestSlope(np.arange(numExposures), ccdGain)

        return abs((slope - theoSlope) / theoSlope) < 0.01

        

    def compare(self):

        # Gets and checks all the conditions for these tests
        condition1 = self.compareAtNominalTemparture()
        condition2 = self.compareFEE(self.leftSimFileFEE, 'left')
        condition3 = self.compareFEE(self.rightSimFileFEE, 'right')
        condition4 = self.compareCCD(self.leftSimFileCCD, 'left')
        condition5 = self.compareCCD(self.rightSimFileCCD, 'right')

        return condition1 and condition2 and condition3 and condition4 and condition5


    def getBestSlope(self, x, y):

        # Returns the best slope for a linear fit of the data
        return (x - np.mean(x)).dot(y - np.mean(y)) / (x - np.mean(x)).dot(x - np.mean(x))

        



        






if __name__ == "__main__":
    t = Gain()
    print(t.run())
    


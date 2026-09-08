from test import Test

import platosim.referenceFrames as rf
import numpy                    as np
import matplotlib.pyplot        as plt





"""
This test is designed to check the Quantum Efficiency. The test runs the simulation with the QuantumEfficiency switched on and with the QuantumEfficiency switched off with no stars in the subfield. 
The quotient between these two images is calculated and compared to the theoretically predicted one from the input file for different point on the subfield.
"""







class QuantumEfficiency(Test):

    def setNr(self):
        self.nr = "011.4"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 2000
        self.sim["SubField/NumColumns"] = 2000

        self.sim["Platform/Orientation/Angles/DecPointing"] = - self.sim["Platform/Orientation/Angles/DecPointing"]


    def runSimulation(self):
        # Run the simulation with Quantum Efficiency switched on.
        self.sim["CCD/IncludeQuantumEfficiency"] = "yes"
        self.simFile1 = self.sim.run(removeOutputFile = True)

        # Run the simulation with Quantum Efficiency switched off.
        self.sim["CCD/IncludeQuantumEfficiency"] = "no"
        self.simFile2 = self.sim.run(removeOutputFile = True)
        

    def compare(self):

        quotient = self.simFile1.getImage(0) / self.simFile2.getImage(0)
        distances = np.array([])
        quantumEfficiency = np.array([])
        expectedQE = self.sim["CCD/QuantumEfficiency/MeanQuantumEfficiency"] * self.sim["CCD/QuantumEfficiency/MeanAngleDependency"]

        for row in range(0, self.sim["SubField/NumRows"], 50):
    
            for column in range(0, self.sim["SubField/NumColumns"], 50):
                        
                # Calculate the angular distance of pixel (row, column) from the optical axis
        
                xFP, yFP = rf.pixelToFocalPlaneCoordinates(column, row, self.sim["CCD/PixelSize"], 0, 0, 0)     # Focal-plane coordinates[mm]
                distance = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP, self.sim["Camera/FocalLength/ConstantValue"] * 1000))     # Angular distance from the OA [degrees]
        
                distances = np.append(distances, distance)
                quantumEfficiency = np.append(quantumEfficiency, quotient[row, column])


        expected = expectedQE * np.ones(len(distances))
        RMS = np.sqrt(np.sum([(qe - theo)**2 / len(expected) for qe, theo in zip(quantumEfficiency, expected)]))

        return RMS < 0.01








    
if __name__ == "__main__":
    t = QuantumEfficiency()
    print(t.run())


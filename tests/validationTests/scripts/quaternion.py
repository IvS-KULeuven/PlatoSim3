
import numpy as np
from test import Test, eprint
import platosim.referenceFrames as rf
from platosim.simulation import Simulation



""" 
This test checks whether specifying the platform orientation with plain equatorial angles gives the same result as 
specifying it with a quaternion.
"""



def qmul(qa, qb):
    """
    \brief  Multiplication of two quaternions qa and qb.
     
    \input Both quaternions are structured as (q0, qx, qy, qz)
    
    \output A quaternion with the same structure.
    """
    result = [0.0, 0.0, 0.0, 0.0]
    result[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]
    result[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2]
    result[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1]
    result[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
    return result




class Quaternion(Test):

    def setNr(self):
        """ 
        This function is called by super.__init__() to initialize the output directories
        """
        self.nr = "022"                                             # This should probably not be hardcoded, but passed as an argument


    def setAllEffects(self):                                        # This should be called "configureSimulationInput()"
        super().setAllEffects()                                     # A bit of a misnomer, because it "un"sets most effects, i.e. switches them off. 
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/ZeroPointRow"]    = 1955
        self.sim["SubField/ZeroPointColumn"] = 1955
        self.sim["SubField/NumRows"]    = 600
        self.sim["SubField/NumColumns"] = 600
        self.sim["PSF/Model"] = "AnalyticNonGaussian"


    def runSimulation(self):
        
        # 1) a run where we specify the platform pointing with sky angles

        self.sim = Simulation("test" + self.nr + "_angles", outputDir = self.outputDir)        # self.outputDir was set in super.__init__()
        self.setAllEffects()                                                                   # configures part of the yaml input file
        self.sim["Platform/Orientation/Source"] = "Angles"
        RA, dec = 86.8, -46.4                                       # [deg]
        solarPanelOrientation = 54.7                                # [deg]
        self.sim["Platform/Orientation/Angles/RAPointing"] = RA
        self.sim["Platform/Orientation/Angles/DecPointing"] = dec
        self.sim["Platform/Orientation/Angles/SolarPanelOrientation"] = solarPanelOrientation
        simFile_angles = self.sim.run(removeOutputFile=True)                                   # Remove any possible pre-existing HDF5 output file
        self.starIDs_angles, self.row_angles, self.col_angles, dummy, dummy, dummy = simFile_angles.getStarCoordinates(0)     # Image nr 0


        # 2) a run where we specify the platform pointing with a quaternion

        self.sim = Simulation("test" + self.nr + "_quaternion", outputDir = self.outputDir)    # self.outputDir was set in super.__init__()
        self.setAllEffects()                                                                   # configures part of the yaml input file
        self.sim["Platform/Orientation/Source"] = "Quaternion"
        q_EQ2A = [np.cos(np.deg2rad(RA)/2.), 0.0, 0.0, np.sin(np.deg2rad(RA)/2.0)]
        q_A2B  = [np.cos(np.pi/4 - np.deg2rad(dec)/2.), 0.0, np.sin(np.pi/4 - np.deg2rad(dec)/2.0), 0.0]
        q_B2PLM = [np.cos(np.deg2rad(solarPanelOrientation)/2.), 0.0, 0.0, np.sin(np.deg2rad(solarPanelOrientation)/2.0)]
        q_EQ2PLM = qmul(q_EQ2A, qmul(q_A2B, q_B2PLM))
        self.sim["Platform/Orientation/Quaternion/Components"] = q_EQ2PLM
        simFile_quaternion = self.sim.run(removeOutputFile=True)                          # Remove any possible pre-existing HDF5 output file
        self.starIDs_quat, self.row_quat, self.col_quat, dummy, dummy, dummy = simFile_quaternion.getStarCoordinates(0)     # Image nr 0


    def compare(self):
        # Check if both images contain the same stars 

        check1 = (self.starIDs_angles == self.starIDs_quat).all()
        if not check1:
            eprint("Quaternion: images do not contain the same stars")

        # Check if the pixel coordinates of the stars are equal up to 1/20 pixel

        check2 = np.abs(self.row_angles - self.row_quat).max() <= 0.05
        check3 = np.abs(self.col_angles - self.col_quat).max() <= 0.05

        if not check1 or not check2:
            eprint("Quaternion: pixel coordinates of the stars are not the same in both images")

        return check1 and check2 and check3

                        
                            


    

if __name__ == "__main__":
    mytest = Quaternion()
    print(mytest.run())
    

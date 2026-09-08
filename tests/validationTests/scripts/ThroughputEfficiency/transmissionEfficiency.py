from test import Test
import numpy as np

import h5py
import platosim.h5   as h5





""" 
This test to check the transmission efficiency. The test checks that the transmission efficiency decreases linearly during the lifetime of the
mission. The fits uses the BOL and EOL transmission efficiency obtained from the input file. The test passes of the RMS difference between the fit
and the obtained values is smaller then 0.01. 
"""


class TransmissionEfficiency(Test):

    def setNr(self):
        self.nr = "011.1"

    def setAllEffects(self):

        super().setAllEffects()
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["SubField/NumRows"]    = 1
        self.sim["SubField/NumColumns"] = 1

        self.sim["Platform/Orientation/Angles/DecPointing"] = -self.sim["Platform/Orientation/Angles/DecPointing"]


        self.sim["Telescope/TransmissionEfficiency/BOL"] = 0.9
        self.sim["Telescope/TransmissionEfficiency/EOL"] = 0.1
        self.years = 7
        self.sim["ObservingParameters/MissionDuration"]  = self.years
        self.tCycle = self.sim["ObservingParameters/CycleTime"]
 





        
    def runSimulation(self):

        # The simulation of run at equal spaced time intervales, so that there are 2 exposures in one year.
        # The obtained flux and time is then added to an np.array. 
        secondsInYear = 60 * 60 * 24 * 365
        
        time          = np.array([])
        flux          = np.array([])
        
        for exposure in range(0, int(secondsInYear * self.years / self.tCycle), int(secondsInYear / (2 * self.tCycle))):

            self.sim["ObservingParameters/BeginExposureNr"] = exposure
            simFile = self.sim.run(removeOutputFile = True)
            image  = simFile.getImage(exposure)

            hfile = h5py.File(self.outputDir + "/test" + self.nr + ".hdf5", 'r')
            #t = h5.h5get(hfile, "Time/time", verbose = False)
            t = simFile.getTime()
            time = np.append(time, t)
            flux = np.append(flux, image[0][0])

        self.time = time / secondsInYear
        self.flux = flux


        




        
    def compare(self):

        # The obtained transmission eficiency is compared versus a linear predicted curve.The test passes of the root mean square difference
        # is smaller then 0.01.
        tmeBOL   = self.sim["Telescope/TransmissionEfficiency/BOL"]
        tmeEOL   = self.sim["Telescope/TransmissionEfficiency/EOL"]
        endTime  = self.years
        tmePlato = self.flux / self.flux[0] * tmeBOL
        tmeTheo  = tmeBOL + self.time * (tmeEOL - tmeBOL) / endTime
        RMS      = np.sqrt(np.sum((1 / len(tmePlato)) * (tmePlato - tmeTheo)**2))
        return RMS < 0.01


        








if __name__ == "__main__":

    t = TransmissionEfficiency()

    print(t.run())


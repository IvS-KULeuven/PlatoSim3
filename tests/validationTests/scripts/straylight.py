from test import Test
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from plot_straylight import plot_figure
from datetime import datetime, timedelta







class Straylight(Test):

    def setAllEffects(self):

        super().setAllEffects()
        self.numExposures = 1
        self.sim["ObservingParameters/NumExposures"] = self.numExposures
        self.sim["ObservingParameters/BeginExposureNr"] = 0
        self.sim["SubField/NumColumns"] = 9
        self.sim["SubField/NumRows"] = 9
        self.sim["Sky/StrayLight/IncludeStrayLight"] = "yes"
        self.sim["Telescope/TransmissionEfficiency/EOL"] = self.sim["Telescope/TransmissionEfficiency/BOL"]
        self.sim["Sky/SkyBackground/BackgroundValue"] = 0
        self.sim["Platform/Orientation/Angles/RAPointing"] = 95.31041666666665
        self.sim["Platform/Orientation/Angles/DecPointing"] = -47.886944444444445
        self.sim["Telescope/GroupID"] = "Custom"
        self.sim["Telescope/AzimuthAngle"] = 0.
        self.sim["Telescope/TiltAngle"] = 0.


    def run(self):
        
        self.runSimulation(365)
        return self.compare()


    def runSimulation(self, iterations):

        # time = [24*60*60*(1+i) for i in range(iterations)]
        # positions = []
        # sl = []

        # for i in range(iterations):
        #     s, p = self.run_for_iteration(i)
        #     sl.append(s[0])
        #     positions.append(p)

        # plot_figure(positions, sl)

        pnts = [16, 43, 302]
        
        self.rslts = [self.run_for_iteration(i)[0] for i in pnts]

        

    def compare(self):
        vls = [0.0026, 0.01276, 0.00671]
        return all(abs(r-v) < 0.001 for r, v in zip(self.rslts, vls))


    def run_for_iteration(self, i):

        t0 = self.sim["Sky/StrayLight/Time0"]
        self.sim["ObservingParameters/BeginExposureNr"] = i*int(24*60*60 / 25)
        self.simFile = self.sim.run(removeOutputFile=True)
        s = self.simFile.getStraylight()
        p = get_positions(t0, 24*60*60*(1+i))
        return s, p

    
    def setNr(self):
        self.nr = "045"









    

def get_positions(t0, t):

    t0 = pd.to_datetime(t0, format='%Y%m%dT%H%M%S') + timedelta(seconds=t)

    fileName = os.getenv('PLATO_PROJECT_HOME') + '/inputfiles/Plato_straylight_example_index_2574.csv'
    file = pd.read_csv(fileName)
    file["T"] = pd.to_datetime(file["#Date"], format='%Y%m%dT%H%M%S')

    # Find the index of the closest time
    idx = (file['T'] - t0).abs().idxmin()

    # Get the position values
    columns_sc = ["X_SC_EME2000 [km]", "Y_SC_EME2000 [km]", "Z_SC_EME2000 [km]"]
    columns_sun = ["X_SUN_EME2000 [km]", "Y_SUN_EME2000 [km]", "Z_SUN_EME2000 [km]"]
    columns_moon = ["X_MOON_EME2000 [km]", "Y_MOON_EME2000 [km]", "Z_MOON_EME2000 [km]"]
    sc_position = file.loc[idx, columns_sc].values
    sun_position = file.loc[idx, columns_sun].values
    moon_position = file.loc[idx, columns_moon].values
    
    return [sc_position, sun_position, moon_position]

        




        










if __name__ == "__main__":

    s = Straylight()
    print(s.run())

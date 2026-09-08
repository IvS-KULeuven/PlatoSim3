
from math import radians, degrees
import numpy as np
from platosim.referenceFrames import focalPlaneToSkyCoordinates
from test import Test





""" 
Test the orientation of the camera reference frame in the (RA, dec) plane.
Cf Fig A1 of the PlatoSim paper of Jannsen et al. (2023).
The x-vector should point towards positive declination, the y-vectors should point towards negative right ascension.
"""




class FocalPlaneCoordinates(Test):

    def setNr(self):
        self.nr = "042"  


    def setAllEffects(self):
        pass                                                                                  # Nothing to set, we only use the Python modules.


    def compare(self):

        raPlatformAngles = np.deg2rad([10.0, 100.0, 200.0, 280.0])                            # [rad] Different test positions on the sky
        decPlatformAngles = np.deg2rad([-40.0, 0.0, 40.0])                                    # [rad] Different test positions on the sky
        solarPanelOrientation = 0.0                                                           # [rad]
        tiltAngle = radians(9.2)                                                              # [rad]
        azimuthAngles = np.deg2rad([45.0, 135.0, 225.0, 315.0])                               # [rad] Camera groups 1,2,3,4
        focalPlaneAngle = 0.0                                                                 # [rad] Coords in CAM-RF are now same as in FP-RF
        focalLength = 247.52                                                                  # [mm] 

        for raPlatform in raPlatformAngles:
            for decPlatform in decPlatformAngles:
                for azimuth in azimuthAngles:
                    xfp_opticalAxis = 0.0                                                     # [mm]   Focal plane coordinates of the optical axis
                    yfp_opticalAxis = 0.0                                                     # [mm]
                    xfp_unitx = 1.0                                                           # [mm]   Focal plane coords of unit x vector
                    yfp_unitx = 0.0                                                           # [mm]
                    xfp_unity = 0.0                                                           # [mm]   Focal plane coords of unit y vector
                    yfp_unity = 1.0                                                           # [mm]

                    ra_opticalAxis, dec_opticalAxis = focalPlaneToSkyCoordinates(xfp_opticalAxis, yfp_opticalAxis, raPlatform, decPlatform, solarPanelOrientation, tiltAngle, azimuth, focalPlaneAngle, focalLength)

                    ra_unitx, dec_unitx = focalPlaneToSkyCoordinates(xfp_unitx, yfp_unitx, raPlatform, decPlatform, solarPanelOrientation, tiltAngle, azimuth, focalPlaneAngle, focalLength)
                    ra_unitx = degrees(ra_unitx - ra_opticalAxis)                 # subtract the zero point of the camera pointing
                    dec_unitx = degrees(dec_unitx - dec_opticalAxis)

                    ra_unity, dec_unity = focalPlaneToSkyCoordinates(xfp_unity, yfp_unity, raPlatform, decPlatform, solarPanelOrientation, tiltAngle, azimuth, focalPlaneAngle, focalLength)
                    ra_unity = degrees(ra_unity - ra_opticalAxis)                 # subtract the zero point of the camera pointing
                    dec_unity = degrees(dec_unity - dec_opticalAxis)

                    success = True

                    # Check if the unit_x vector is always pointing towards positive declination

                    if dec_unitx < 0.0:
                        success = False

                    # Check if the ra of the unit_x vector is close to zero.
                    # The numerical accuracy seems not great, so I simply check that the ra value is considerably smaller than the dec value.

                    if np.fabs(ra_unitx) > 0.5 * np.fabs(dec_unitx):
                        success = False

                    # Check if the unit_y vector is always pointing towards negative right ascension

                    if ra_unity > 0.0:
                        success = False

                    # Check if the dec of the unit_y vector is close to zero.
                    # The numerical accuracy seems not great, so I simply check that the dec value is considerably smaller than the ra value.

                    if np.fabs(dec_unity) > 0.5 * np.fabs(ra_unity):
                        success = False

        return success





if __name__ == "__main__":
    t = FocalPlaneCoordinates()
    print(t.run())





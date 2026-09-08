""" Inserting a single point source.

The magnitude of this source is specified, together with its coordinates:

    - field angles (theta, phi) -> insertSourceFieldAngles;
    - focal-plane coordinates (x, y) -> insertSourceFP;
    - CCD coordinates (row, column) -> insertSourceCCD.

A star catalogue with only this source will be created, so you need to specify the filename you want to use for that.

Please, bear in mind that after applying these methods, you should NOT touch the configuration parameters anymore that
have to do with:

    - the platform pointing (i.e. sim["ObservingParameters/RApointing"] and sim["ObservingParameters/DecPointing"]);
    - the star catalogue (i.e. sim["ObservingParameters/StarCatalogFile"]);
    - the telescope group (i.e. sim["Telescope/GroupID"]);
    - field distortion (i.e. sim["Camera/IncludeFieldDistortion"]).

Assumed are:

    - fixed focal-plane orientation (i.e. sim["Camera/FocalPlaneOrientation/Source"] == "ConstantValue");
    - fixed focal length (i.e. sim["Camera/FocalLength/Source"] == "ConstantValue").

When applying these methods, you will see printouts that help you to choose the CCD (i.e. sim["CCD/Position"]) and the
exact location of the simulated sub-field on that CCD (i.e. sim["SubField"]).

When configuring your (CCD and) sub-field, bear in mind that jitter and/or TED may push the source out of the sub-field,
so make sure the sub-field is big enough.
"""

from math import radians, degrees

from platosim.referenceFrames import focalPlaneCoordinatesFromGnomonicRadialDistance, focalPlaneToSkyCoordinates, \
    pixelToSkyCoordinates, focalPlaneToPixelCoordinates
from platosim.simulation import Simulation




def insertSourceFieldAngles(sim: Simulation, theta: float, phi: float, magnitude: float, filename: str):
    """ Insert a single point sources at the given field angles.

    Create a star catalogue (with the given filename) with a single point source (of given magnitude), inserted at the
    given field angles.  The given Simulation object will be configured such that it uses this star catalogue.

    Args:
        - sim: Simulation object for which to insert the point source.
        - theta: Gnomonic distance from the optical axis [degrees].
        - phi: In-field angle (counter-clockwise, from the focal-plane x-axis) [degrees].
        - magnitude: Magnitude of the point source.
        - filename: Filename to use for the star catalogue.
    """

    # Switch off field distortion

    sim["Camera/IncludeFieldDistortion"] = "no"

    focalLength = sim["Camera/FocalLength/ConstantValue"] * 1000

    # Conversion: field angles -> focal-plane coordinates

    xFP, yFP = focalPlaneCoordinatesFromGnomonicRadialDistance(radians(theta), focalLength,
                                                                 inPlaneRotation=radians(phi))      # [mm]

    insertSourceFP(sim, xFP, yFP, magnitude, filename)




    
def insertSourceFP(sim: Simulation, xFP: float, yFP: float, magnitude: float, filename: str):
    """ Insert a single point sources at the given focal-plane coordinates.

    Create a star catalogue (with the given filename) with a single point source (of given magnitude), inserted at the
    given focal-plane coordinates.  The given Simulation object will be configured such that it uses this star
    catalogue.

    Args:
        - sim: Simulation object for which to insert the point source.
        - xFP: Focal-plane x-coordinate [mm].
        - yFP: Focal-plane y-coordinate [mm].
        - magnitude: Magnitude of the point source.
        - filename: Filename to use for the star catalogue.
    """

    # Switch off field distortion

    sim["Camera/IncludeFieldDistortion"] = "no"

    printCCD(sim, xFP, yFP)

    focalLength = sim["Camera/FocalLength/ConstantValue"] * 1000                           # [mm]
    raPlatform = radians(sim["ObservingParameters/RApointing"])                            # [radians]
    decPlatform = radians(sim["ObservingParameters/DecPointing"])                          # [radians]
    solarPanelOrientation = radians(sim["Platform/SolarPanelOrientation"])                 # [radians]
    angleFP = radians(sim["Camera/FocalPlaneOrientation/ConstantValue"])                   # [radians]

    telescopeGroup = sim["Telescope/GroupID"]

    if telescopeGroup == "Custom":

        tiltAngle = radians(sim["Telescope/TiltAngle"])                                    # [radians]
        azimuthAngle = radians(sim["Telescope/AzimuthAngle"])                              # [radians]

    else:

        tiltAngle = radians(sim["CameraGroups/TiltAngle"][telescopeGroup - 1])            # [radians]
        azimuthAngle = radians(sim["CameraGroups/AzimuthAngle"][telescopeGroup - 1])      # [radians]

    # Conversion focal-plane coordinates -> sky coordinates

    ra_star, dec_star = focalPlaneToSkyCoordinates(xFP, yFP, raPlatform, decPlatform, solarPanelOrientation,
                                                   tiltAngle, azimuthAngle, angleFP, focalLength)   # [radians]

    makeStarCatalog(sim, ra_star, dec_star, magnitude, filename)




    
def insertSourceCCD(sim: Simulation, row: float, column: float, ccdCode: int, magnitude: float, filename: str):
    """ Insert a single point sources at the given CCD coordinates.

    Create a star catalogue (with the given filename) with a single point source (of given magnitude), inserted at the
    given CCD coordinates.  The given Simulation object will be configured such that it uses this star catalogue.

    Args:
        - sim: Simulation object for which to insert the point source.
        - row: Row coordinate [pixels] in the CCD reference frame with the given CCD code.
        - column: Column coordinate [pixels] in the CCD reference frame with the given CCD code.
        - ccdCode: CCD code [1/2/3/4].
        - magnitude: Magnitude of the point source.
        - filename: Filename to use for the star catalogue.
    """

    # Switch off field distortion

    sim["Camera/IncludeFieldDistortion"] = "no"

    print(f"The source will be located on CCD {ccdCode} on pixel (row, column) = ({row}, {column})")

    # Conversion: pixel coordinates -> sky coordinates

    raStar, decStar = pixelToSkyCoordinates(sim, str(ccdCode), column, row)   # [radians]

    makeStarCatalog(sim, raStar, decStar, magnitude, filename)



    
def printCCD(sim: Simulation, xFP: float, yFP: float):
    """ Print out clues on where you should place the sub-field.

    Args:
        - sim: Simulation object.
        - xFP: Focal-plane x-coordinate [mm].
        - yFP: Focal-plane y-coordinate [mm].
    """

    ccdCode = None

    if xFP > 0:

        if yFP > 0:

            ccdCode = 4

        elif yFP < 0:

            ccdCode = 3

    elif xFP < 0:

        if yFP > 0:

            ccdCode = 1

        elif yFP < 0:

            ccdCode = 2

    if ccdCode is None:

        print("The source will fall in the gap between CCDs")
        return

    pixelSize = sim["CCD/PixelSize"]   # [micron]
    ccdZeroPointX = sim["CCDPositions/OriginOffsetX"][ccdCode - 1]
    ccdZeroPointY = sim["CCDPositions/OriginOffsetY"][ccdCode - 1]
    ccd_angle = radians(sim["CCDPositions/Orientation"][ccdCode - 1])

    print(ccdZeroPointX, ccdZeroPointY, ccd_angle)

    column, row = focalPlaneToPixelCoordinates(xFP, yFP, pixelSize, ccdZeroPointX, ccdZeroPointY, ccd_angle)

    print(f"The source will be located on CCD {ccdCode}, on pixel (row, column) = ({row}, {column})")



    
def makeStarCatalog(sim: Simulation, raStar: float, decStar: float, magnitude: float, filename: str):
    """ Create a star catalogue for the given Simulation object.

    Create a star catalogue (with the given filename) with a single point source, at the given sky coordinates and with
    the given magnitude. The given Simulation object is configured such that it uses this star catalogue.

    Args:
        - sim: Simulation object.
        - ra_star: Right ascension [radians] of the point source.
        - dec_star: Declination [radians] of the point source.
        - magnitude: Magnitude of the point source.
        - filename: Filename to use for the star catalogue.
    """

    raStar = degrees(raStar)
    decStar = degrees(decStar)

    starCatalog = open(filename, "w")
    starCatalog.write("# RA DEC Vmag starID\n")
    starCatalog.write("{0}  {1}  {2}  {3}\n".format(raStar, decStar, magnitude, 1))
    starCatalog.close()

    sim["ObservingParameters/StarCatalogFile"] = filename

#!/usr/bin/env python3

"""
This python module contains plot utilities used in the minimal 
PlatoSim installation and in the extra PLATOnium installation.
"""

# Built-in
import os

# PlatoSim standard
import h5py
import shapely.plotting as sp
import shapely.geometry as sg
from tqdm import tqdm
from ipywidgets import interact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib.pyplot import cm
from matplotlib.path import Path
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import constants as c
from scipy.ndimage import median_filter
from scipy.interpolate import make_interp_spline
from scipy.signal import periodogram
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# PlatoSim imports
import platosim.noise           as ns
import platosim.utilities       as ut
import platosim.referenceFrames as rf                                

# Update plots
import getpass
if getpass.getuser() == 'nicholas':
    from platosim.matplotlibrc import setup_paper; setup_paper()
else:
    from platosim.matplotlibrc import setup; setup()

# Hard-code values
aa = 0.5  # Alpha transparency
fs = 14   # Font-size
ms = 0.2  # Marker-size
lw = 0.5  # Line-width
pt = 0.1  # Percentage

# Define some nice colors
colors_sea = ['royalblue', 'lightseagreen', 'limegreen']
colors_hot = ['tomato', 'darkorange', 'gold']


#--------------------------------------------------------------#
#                         GRAPHICAL TOOLS                      #
#--------------------------------------------------------------#


def getAxesMinMax(x=None, y=None, percentage=2):

    """Automatically adjust min and max limits in plot.

    This is a small utility to automatically scale the axes of a plot
    using a default spacing in percentage (pt). The user need only to
    specify which axis a min and max limit should be returned from.
    """
    
    pt = percentage / 100.
    
    if x is not None:
        x = np.sort(x)
        axmin = x[0]  - (x[-1]-x[0])*pt
        axmax = x[-1] + (x[-1]-x[0])*pt

    if y is not None:
        y = np.sort(y)
        axmin = np.min(y) - (np.max(y)-np.min(y))*pt
        axmax = np.max(y) + (np.max(y)-np.min(y))*pt
        
    return axmin, axmax





def discretizeColorbar(cbins, cmap="coolwarm"):

    """Enforce that colorbar use discrete values.

    Parameters
    ----------
    cbins : int
        Number of bins to discretize colorbar
    cmap : str
        Matplotlib color map (see docs for possible cmaps)

    Return
    ------
    norm : matplotlib object
        Matplotlib colorbar object used to normalize the colorbar scale.
    """
        
    # define the colormap
    
    cmap = plt.get_cmap(cmap)
    
    # Extract all colors from the the choosen colormap
    
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # Create the new map
    
    cmap = colors.LinearSegmentedColormap.from_list('custom', cmaplist, cmap.N)

    # define the bins and normalize
    
    norm = colors.BoundaryNorm(cbins, cmap.N)

    return norm

    



def slider(imagePlot, images, Nimg, label="Image number"):

    """Enable the usage of a slider to show multiple images.

    Parameters
    ----------
    imagePlot : imshow object
        Matplotlib imshow object of the pixel image.
    images : ndarray
        Image cube with pixel data.
    Nimg : int
        Number of images used for slider.
    label : str
        Label for slider.
    
    Return
    ------
    None
    """
    
    # Function to update slider
    
    def update_image(n=0):
        image = images[n]
        imagePlot.set_data(image)
        fig.canvas.draw()

    # Show the image
    
    slider = IntSlider(0, 0, 10, 1, layout=Layout(width='500px'))
    interact(update_image, n=(0,Nimg-1), x=slider)





def compass(ax, x, y, size):
    
    """Add a compass to indicate the north and east directions.

    Parameters
    ----------
    x, y : float
        Position of compass vertex in axes coordinates.
    size : float
        Size of compass in axes coordinates.
    """
    xy = x, y
    scale = ax.wcs.pixel_scale_matrix
    scale /= np.sqrt(np.abs(np.linalg.det(scale)))
    self.annotate(label, xy, xy + size * n,
                  self.transAxes, self.transAxes,
                  ha='center', va='center',
                  arrowprops=dict(arrowstyle='<-', shrinkA=0.0, shrinkB=0.0))




    
def moveColorbarExponent(x_offs=0, y_offs=1, dig=0, side='left', omit_last=False):

    """Move scientific notation exponent from top to the side.
    
    Additionally, one can set the number of digits after the comma
    for the y-ticks, hence if it should state 1, 1.0, 1.00 and so forth.

    Note
    ----
    This is kind of a non-satisfying hack, which should be handled more
    properly. But it works. Functions to look at for a better implementation:
    >>> ax.ticklabel_format
    >>> ax.yaxis.major.formatter.set_offset_string

    Parameters
    ----------
    offs : float, optional; <0>
        Horizontal movement additional to default.
    dig : int, optional; <0>
        Number of decimals after the comma.
    side : string, optional; {<'left'>, 'right'}
        To choose the side of the y-axis notation.
    omit_last : bool, optional; <False>
        If True, the top y-axis-label is omitted.

    Returns
    -------
    locs : list
        List of y-tick locations.
    """

    # Get the ticks
    
    locs, _ = plt.yticks()

    # Put the last entry into a string, ensuring it is in scientific notation
    # E.g: 123456789 => '1.235e+08'
    
    llocs = '%.3e' % locs[-1]

    # Get the magnitude, hence the number after the 'e'
    # E.g: '1.235e+08' => 8
    
    yoff = int(str(llocs).split('e')[1])

    # If omit_last, remove last entry
    
    if omit_last:
        slocs = locs[:-1]
    else:
        slocs = locs

    # Set ticks to the requested precision
    
    form = r'$%.' + str(dig) + 'f$'
    plt.yticks(locs, list(map(lambda x: form % x, slocs / (10 ** yoff))))

    # Define offset depending on the side
    
    if side == 'left':
        x_offs = -.18 - x_offs  # Default left: -0.18
    elif side == 'right':
        x_offs = 1 + x_offs     # Default right: 1.0

    # Plot the exponent
    
    plt.text(x_offs, y_offs, r'$\times10^{%i}$' % yoff, transform=
    plt.gca().transAxes, verticalalignment='top')

    # That's it!
    
    return locs





#--------------------------------------------------------------#
#                         PROJECTIONS                          #
#--------------------------------------------------------------#


def drawCCDsInFocalPlane(pixelSize=18, plotCCDlabels=True, normal=True):

    """Plot the 4 CCDs in the focal plane in the FP' reference frame.
             
    May serve as a background to overplot the projected stars on the focal plane.

    Parameters
    ----------
    pixelSize : int
        Size of one PLATO pixel [micron]
    plotCCDlabels : bool
        Request to plot the CCD labels: True (including) or False (excluding)
    normal : bool
        True for the normal camera configuration, False for the fast cameras

    Return
    ------
    None
    """

    # Select the proper CCD codes depending on whether we're dealing
    # with the nominal or the fast cams

    if normal == True:
        ccdCodes = ['1', '2', '3', '4']
    else:
        ccdCodes = ['1F', '2F', '3F', '4F']

    # Set up the colors to be used to draw each CCD.
    # Different CCDs have different colors.

    color = {'1': 'b', '1F': 'b', '2': 'r', '2F': 'r',
             '3': 'g', '3F': 'g', '4': 'k', '4F': 'k'}

    # Plot each of the 4 CCDs

    fig = plt.figure(figsize = (9,9))
    ax = fig.add_subplot(111)

    for ccdCode in ccdCodes:

        # Get the corner coordinates in the FP' plane

        cornersXmm, cornersYmm = rf.computeCCDcornersInFocalPlane(ccdCode, pixelSize)

        # Repeat the coordinates of the 1st corner, to plot a nice closed loop

        x = np.append(cornersXmm, cornersXmm[0])
        y = np.append(cornersYmm, cornersYmm[0])

        ax.plot(x, y, c=color[ccdCode])

        # Overplot the row closest to the readout register with a thicker line

        ax.plot([x[0], x[1]], [y[0], y[1]], c=color[ccdCode], linewidth=4)

        # If required, also plot the CCD labels

        if plotCCDlabels:

            minX = np.min(cornersXmm)
            maxX = np.max(cornersXmm)
            minY = np.min(cornersYmm)
            maxY = np.max(cornersYmm)
            middleX = minX + (maxX - minX) / 8.
            middleY = minY + (maxY - minY) / 8.
            ax.text(middleX, middleY, ccdCode, fontsize=30, color="gray")

    ax.set_xlabel(r'$x_{\rm FP}$ [mm]', fontsize = 20)
    ax.set_ylabel(r'$y_{\rm FP}$ [mm]', fontsize = 20)





def drawCCDsInCameraFocalPlane(fig):

    """Draw the CCDs in the focal plane of a N-CAM.

    Parameters
    ----------
    fig :  A matplotlib.pyplot figure object [e.g. plt.figure(figsize=(10,10)].

    Return
    ------
    ax : matplotlib object
        Matplotlib axes object to modify the figure after use.
    """

    # Constants

    pixelSize   = 18        # [µm]
    fovDegrees  = 18.8908   # [deg]
    focalLength = 247.52    # [mm]
    ccdCodes    = ["1", "2", "3", "4"]

    # Size of the FOV

    fovMm = focalLength * np.tan(np.radians(fovDegrees))

    # Star plotting

    ax = fig.add_subplot(111)

    # Plot the camera aperture

    circ = plt.Circle((0, 0), radius=fovMm, color="none", linewidth=2, label="Telescope FOV", zorder=1)
    ax.add_patch(circ)
    circ.set_edgecolor("g")
    circ.set_facecolor("lightgray")

    # Plot location of optical axis

    ax.plot([0], [0], "rx", label="Optical axis", zorder=2)

    # Plot CCDs

    ori = [[-30, 0, 0, -30], [30, 0, 0, -30], [30, 0, 0, 30], [-30, 0, 0, 30]]

    for ccdCode in ccdCodes:

        # Fetch CCD corners

        cornersX, cornersY = rf.computeCCDcornersInFocalPlane(ccdCode, pixelSize)

        # Draw CCD names at the center of each CCD

        ax.text(np.mean(cornersX) - 10, np.mean(cornersY), "CCD " + ccdCode, fontsize=20)

        # Draw each CCD

        cornersX = np.append(cornersX, cornersX[0])
        cornersY = np.append(cornersY, cornersY[0])

        if ccdCode == "1":
            ax.plot(cornersX, cornersY, color="b", label="CCD footprint", zorder=3)
        else:
            ax.plot(cornersX, cornersY, color="b", zorder=3)

        # Plot arrays to indicate CCD origin

        ax.arrow(cornersX[0], cornersY[0], ori[int(ccdCode)-1][0], ori[int(ccdCode)-1][1],
                 head_width=3, head_length=3, fc='k', ec='k', linewidth=2, zorder=4)
        ax.arrow(cornersX[0], cornersY[0], ori[int(ccdCode)-1][2], ori[int(ccdCode)-1][3],
                 head_width=3, head_length=3, fc='k', ec='k', linewidth=2, zorder=4)

    # Settings

    plt.legend(prop={'size': 20}, bbox_to_anchor=(1.0, 1.0))
    ax.set_title('CCDs in camera focal plane', fontsize = 24)
    ax.set_xlabel(r'$x_{\mathrm{FP}}$ [mm]', fontsize = 20)
    ax.set_ylabel(r'$y_{\mathrm{FP}}$ [mm]', fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    # Finito!

    return ax





def drawSubfieldInFocalPlane(ccdCode, xCCD, yCCD, subfieldSizeX, subfieldSizeY, pixelSize):

    """Draw a subfield in the focal plane.
    
    Parameters
    ----------
    xCCD : int
        Center x coordinate of the subfield [pixels]
    yCCD : int
        Center y coordinate of the subfield [pixels]
    subfieldSizeX : int
        Size of the subfield along the x-axis [pixels]
    SubfieldSizeY : int
        Ssize of the subfield along the y-axis [pixels]
    PixelSize : int, float
        The size of a pixel in microns

    Return
    ------
    Subfield frames draw on the current plot. The subfield is drawn with respect of the 
    coordinate system of the CCD. Shown is a:
    - A blue dot indicates the lower left corner of the subfield.
    - A green dot indicates the upper right corner of the subfield.
    - A red dot indicates the center of the subfield.
    """

    # Compute the position of the subfield in pixel coordinates, for the current CCD, 
    # disregarding the physical extend of the CCD. LL = lower left, UR = upper right.

    zeroPointXmm = rf.CCD[ccdCode]["zeroPointXmm"]
    zeroPointYmm = rf.CCD[ccdCode]["zeroPointYmm"]
    ccdAngle     = rf.CCD[ccdCode]["angle"]

    xFPprime, yFPprime     = rf.pixelToFocalPlaneCoordinates(xCCD, yCCD, pixelSize,
                                                             zeroPointXmm, zeroPointYmm,
                                                             ccdAngle)
    xFPprimeLL, yFPprimeLL = rf.pixelToFocalPlaneCoordinates(xCCD - subfieldSizeX/2,
                                                             yCCD - subfieldSizeY/2, \
                                                             pixelSize,
                                                             zeroPointXmm,
                                                             zeroPointYmm,
                                                             ccdAngle)
    xFPprimeUR, yFPprimeUR = rf.pixelToFocalPlaneCoordinates(xCCD + subfieldSizeX/2,
                                                             yCCD + subfieldSizeY/2, \
                                                             pixelSize,
                                                             zeroPointXmm,
                                                             zeroPointYmm,
                                                             ccdAngle)

    verts = [(xFPprimeLL, yFPprimeLL),  # left, bottom
             (xFPprimeLL, yFPprimeUR),  # left, top
             (xFPprimeUR, yFPprimeUR),  # right, top
             (xFPprimeUR, yFPprimeLL),  # right, bottom
             (xFPprimeLL, yFPprimeLL)]  # ignored

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY]

    path = Path(verts, codes)

    # Get the current axis

    currentAxis = plt.gca()
    currentAxis.add_patch( patches.PathPatch(path, facecolor='none') )

    plt.plot(xFPprime, yFPprime, 'ro')
    plt.plot(xFPprimeLL, yFPprimeLL, 'bo')
    plt.plot(xFPprimeUR, yFPprimeUR, 'go')
    plt.draw()

    return





def drawStarInFocalPlane(sim, raStar, decStar, ccd=None):

    """Draw a star given by the equatorial coordinates in the focal plane.

    Paramaters
    ----------
    sim : class object
        Instance of simulation class (see simulation.py)
    raStar : float
        Right ascension of the star [rad]
    decStar : float
        Declination of the star [rad]

    Return
    ------
    Draw a red dot where the star is located on the CCD
    """

    normal = True  # FIXME: where can we specify that we use the fast or normal Camera

    if sim["PSF/Model"] == "MappedFromFile":
        includeFieldDistortion = True
        isMapped               = True
        pathToPsfFile          = sim["PSF/MappedFromFile/Filename"]
        distortionCoefficients = None
    if (sim["Camera/IncludeFieldDistortion"] == "yes"  or
        sim["Camera/IncludeFieldDistortion"] == "1"):
        includeFieldDistortion = True
        isMapped               = False
        pathToPsfFile          = None 
        FIELD_DISTORTION["Coeff"] = sim["Camera/FieldDistortion/Coefficients"]
        FIELD_DISTORTION["InverseCoeff"] = sim["Camera/FieldDistortion/InverseCoefficients"]
        distortionCoefficients = sim["Camera/FieldDistortion/Coefficients"]
    else:
        includeFieldDistortion = False
        distortionCoefficients = None
        pathToPsfFile          = None
        isMapped               = False 

    pixelSize             = float(sim["CCD/PixelSize"])
    raPlatform            = np.radians(float(sim["Platform/Orientation/Angles/RAPointing"]))
    decPlatform           = np.radians(float(sim["Platform/Orientation/Angles/DecPointing"]))
    solarPanelOrientation = np.deg2rad(float(sim["Platform/Orientation/Angles/SolarPanelOrientation"]))
    azimuthTelescope      = np.deg2rad(float(sim["Telescope/AzimuthAngle"]))
    tiltTelescope         = np.deg2rad(float(sim["Telescope/TiltAngle"]))
    focalPlaneAngle       = np.radians(float(sim["Camera/FocalPlaneOrientation/ConstantValue"]))
    focalLength           = float(sim["Camera/FocalLength/ConstantValue"]) * 1000.0 # [m]->[mm]
    ccdZeroPointX         = float(sim["CCD/OriginOffsetX"])
    ccdZeroPointY         = float(sim["CCD/OriginOffsetY"])
    ccdAngle              = np.radians(float(sim["CCD/Orientation"]))

    xFPmm, yFPmm = rf.skyToFocalPlaneCoordinates(raStar, decStar, raPlatform,
                                                 decPlatform, solarPanelOrientation,
                                                 tiltTelescope, azimuthTelescope,
                                                 focalPlaneAngle, focalLength, ccd)

    if includeFieldDistortion:
        if isMapped:
            xFPmm, yFPmm = rf.mappedUndistortedToDistortedFocalPlaneCoordinates(xFPmm, yFPmm,
                                                                                pathToPsfFile,
                                                                                focalLength)
        else:
            xFPmm, yFPmm = rf.undistortedToDistortedFocalPlaneCoordinates(xFPmm, yFPmm,
                                                                          distortionCoefficients)

    ccdCode, xCCD, yCCD = rf.getCCDandPixelCoordinates(raStar, decStar,
                                                       raPlatform, decPlatform,
                                                       solarPanelOrientation,
                                                       tiltTelescope, azimuthTelescope,
                                                       focalPlaneAngle, focalLength,
                                                       pixelSize, includeFieldDistortion,
                                                       normal, isMapped,
                                                       distortionCoefficients,
                                                       pathToPsfFile, ccd)

    if ccdCode == None:
        print ("Warning: DrawStarInFocalPlane(): The star doesn't fall on any of the CCDs.")
    else:
        drawPixelInFocalPlane(ccdCode, xCCD, yCCD, pixelSize, ccd)





def drawPixelInFocalPlane(ccdCode, xCCD, yCCD, pixelSize, ccd=None):

    """Plot a pixel from a particular CCD in the focal plane. 

    The actual position in millimeter is shown as a red dot, while
    the pixel itself is drawn as a rectangle with edge pixelSize.

    Parameters
    ----------
    ccdCode : str
        For nominal camera: either '1', '2', '3', or '4'
        For fast camer: either '1F', '2F', '3F', '4F'
    xCCDpix : int
        X-coordinate (column number, zero-based) of the pixel on the CCD  [pix]
    yCCDpix : int
        Y-coordinate (row number, zero-based) of the pixel on the CCD  [pix]
    pixelSize : float
        The size of a pixel in micron

    Return
    ------
    None
    """

    if ccd is None: ccd = rf.CCD()

    # Compute the position of the star in pixel coordinates, for the current CCD, 
    # disregarding the physical extend of the CCD

    zeroPointXmm = ccd[ccdCode]["zeroPointXmm"]
    zeroPointYmm = ccd[ccdCode]["zeroPointYmm"]
    ccdAngle     = ccd[ccdCode]["angle"]

    xFPmm, yFPmm = rf.pixelToFocalPlaneCoordinates(xCCD, yCCD, pixelSize,
                                                   zeroPointXmm, zeroPointYmm, ccdAngle)

    # Get the current axis

    currentAxis = plt.gca()
    currentAxis.add_patch( patches.Rectangle( (xFPmm, yFPmm),
                                              pixelSize / 1000.0,
                                              pixelSize / 1000.0, fill=False) )
    plt.plot(xFPmm, yFPmm, 'ro')
    plt.draw()
    plt.show()





def drawStarInCCDfocalPlane(fig, sim, xCCD, yCCD, refCcdCode, refGroup,
                            raPlatform, decPlatform, solarPanelOrientation):

    """Draw a star given by the CCD pixel coordinates in the CCD focal plane.

    This function plot the CCD pixel position in the CCD focal plane, together
    with the CCD- and Camera footprints. Given the reference CCD- and camera group,
    the star location is indicated by black arrows sprung from the origo of the 
    parrent CCD. To easy the eye (of this rather complicated plot) the star is
    highligthed by the parrent CCD color and the dashed line is connecting the
    star to the optical axis of the camera-group. 

    NOTE: This plot neglects pointing- and camera alignment errors.

    Parameters
    ----------
    fig : matplotlib figure object
        Example: fig = plt.figure(figsize=(10,10)))
    sim : class object
        Instance of simulation class (see simulation.py)
    xCCD : int
        Column  pixel coordinate [pixel]
    yCCD : int
        Row pixel coordinate [pixel]
    refCcdCode : int
        Reference of CCD (1, 2, 3, 4)
    refGroup : int
        Reference of camera group (1, 2, 3, 4)
    raPlatform : float
        Right Ascension of platform pointing [deg]
    decPlatform : float
        Declination of platform pointing [deg]
    solarPanelOrientation : float
        Orientation of solar panel [deg]
    titlAngle : float
        Tilt angle of camera [deg]
    azimuthAngle : float
        Azimuth angle of camera [deg] 
        
    Return
    ------
    Plot only, no axes object.
    """

    # Parameter used for the plot
    
    numGroups     = 4
    numCorners    = 4
    offset        = 4
    colors        = ['b', 'g', 'orange', 'r']
    ccdCodes      = ["1", "2", "3", "4"]
    tiltAngles    = sim['CameraGroups/TiltAngle'][:numGroups]           # [deg]
    azimuthAngles = sim['CameraGroups/AzimuthAngle'][:numGroups]        # [deg]
    fovDegrees    = sim['CCD/RelativeTransmissivity/RadiusFOV'] + 1     # [deg]
    focalLength   = sim['Camera/FocalLength/ConstantValue'] * 1e3       # [mm]
    pixelSize     = sim['CCD/PixelSize']                                # [micron]
    plateScale    = sim['Camera/PlateScale'] * pixelSize                # [arcsec]

    # Find actual FOV in pixel and mm

    fovPixels  = fovDegrees / plateScale * c.degree / c.arcsec
    fovMm      = focalLength * np.tan(np.radians(fovDegrees))

    sign = lambda x: (1, -1)[x < 0]

    xFP = np.array([])
    yFP = np.array([])
    
    # Position of the Sun
    
    raSun, decSun = rf.sunSkyCoordinatesAwayfromPlatformPointing(np.radians(raPlatform),
                                                                 np.radians(decPlatform),
                                                                 np.radians(solarPanelOrientation))

    # Telescope pointing w.r.t. platform pointing
    
    raTelescope  = []
    decTelescope = []
    for group in range(0, numGroups):
        
        # Telescope pointing (absolute) [radians]
        
        ra, dec = rf.platformToTelescopePointingCoordinates(np.radians(raPlatform),
                                                            np.radians(decPlatform),
                                                            np.radians(solarPanelOrientation),
                                                            np.radians(azimuthAngles[group]),
                                                            np.radians(tiltAngles[group]))

        # Telescope pointing w.r.t. platform pointing
        
        raTelescope.append(np.degrees(ra) - raPlatform)     # [degrees]
        decTelescope.append(np.degrees(dec) - decPlatform)  # [degrees]

    meanDist = (np.mean(np.absolute(raTelescope)) + np.mean(np.absolute(decTelescope))) / 2.0
    for group in range(numGroups):
        raTelescope[group]  = sign(raTelescope[group]) * meanDist
        decTelescope[group] = sign(decTelescope[group]) * meanDist

    # Calculate the coordinates of the telescope pointings (for the 4 telescope groups) 
    # in the focal-plane reference frame of group 1
    
    for group in range(numGroups):
        xFP1, yFP1 = rf.skyToFocalPlaneCoordinates(np.radians(raTelescope[group] + raPlatform), 
                                                   np.radians(decTelescope[group] + decPlatform), 
                                                   np.radians(raPlatform), np.radians(decPlatform),
                                                   np.radians(solarPanelOrientation),
                                                   np.radians(tiltAngles[0]), 
                                                   np.radians(azimuthAngles[0]), 
                                                   0, focalLength)
        xFP = np.append(xFP, xFP1)
        yFP = np.append(yFP, yFP1)

    # Make sure the average of the telescope pointings of the 4 telescope groups
    # is at the origin of the reference frame
    
    xAvg = np.mean(xFP)
    yAvg = np.mean(yFP)
    xFP -= xAvg
    yFP -= yAvg

    meanDist = (np.mean(np.absolute(xFP)) + np.mean(np.absolute(yFP))) / 2.0

    for group in range(numGroups):
        xFP[group] = sign(xFP[group]) * meanDist
        yFP[group] = sign(yFP[group]) * meanDist

    xPixels = np.copy(xFP)
    yPixels = np.copy(yFP)

    for group in range(numGroups):
        # [mm] -> [pixels]
        xPixels[group] = ut.mm2pixels(xPixels[group], focalLength, plateScale)  
        yPixels[group] = ut.mm2pixels(yPixels[group], focalLength, plateScale)

    index = 0

    cornersX, cornersY = rf.computeCCDcornersInFocalPlane(refCcdCode, pixelSize)
    offsetX = ut.mm2pixels(cornersX[index], focalLength, plateScale) + xPixels[refGroup - 1]
    offsetY = ut.mm2pixels(cornersY[index], focalLength, plateScale) + yPixels[refGroup - 1]

    # Correct input pixel coordinates to match orientation of CCD origin
    
    if refCcdCode == '1': xCCD, yCCD = +xCCD, +yCCD
    if refCcdCode == '2': xCCD, yCCD = -yCCD, +xCCD 
    if refCcdCode == '3': xCCD, yCCD = -xCCD, -yCCD
    if refCcdCode == '4': xCCD, yCCD = +yCCD, -xCCD
    
    # START PLOT
    
    ax = fig.add_subplot(111)

    # Add grid

    ax.grid(True)

    # Gray shade N-CAM visibility

    circles = []
    for group in range(numGroups):
        circles.append(sg.Point(-(xPixels[group] - offsetX),
                                -(yPixels[group] - offsetY)).buffer(fovPixels))
    for index in range(numCorners):
        one = circles[index].intersection(circles[index])
        ax.add_patch(sp.patch_from_polygon(one, facecolor='gray', alpha=0.08))
        two = circles[index].intersection(circles[(index + 1) % numCorners])
        ax.add_patch(sp.patch_from_polygon(two, facecolor='gray', alpha=0.08))
        three = circles[index].intersection(circles[(index + 1) % numCorners]).intersection(circles[(index + 2) % numCorners])
        ax.add_patch(sp.patch_from_polygon(three, facecolor='gray', alpha=0.08))
    four = circles[0].intersection(circles[1]).intersection(circles[2]).intersection(circles[3])
    ax.add_patch(sp.patch_from_polygon(four, facecolor='gray', alpha=0.08))        

    # Plot CCD footprint ontop

    for ccdCode in ccdCodes:
        cornersX, cornersY = rf.computeCCDcornersInFocalPlane(ccdCode, pixelSize)
        for corner in range(numCorners):
            cornersX[corner] = ut.mm2pixels(cornersX[corner], focalLength, plateScale)
            cornersY[corner] = ut.mm2pixels(cornersY[corner], focalLength, plateScale)
        cornersX = np.append(cornersX, cornersX[0])  # [mm]
        cornersY = np.append(cornersY, cornersY[0])  # [mm]
        for group in range(numGroups):
            arrayX = cornersX + xPixels[group] - offsetX
            arrayY = cornersY + yPixels[group] - offsetY
            for index in range(5):
                arrayX[index] = (arrayX[index])
                arrayY[index] = (arrayY[index])
            if group != refGroup-1:
                plt.plot(-arrayX, -arrayY, color=colors[group], zorder=1)
            else:
                plt.plot(-arrayX, -arrayY, color=colors[group], zorder=2)
            off = 4000
            if group == refGroup-1 and int(ccdCode) == int(refCcdCode):
                ax.text(-(xPixels[group] - offsetX + sign(cornersX[3]) * off),
                        -(yPixels[group] - offsetY + sign(cornersY[3]) * off),
                        ccdCode, fontsize=20, color=colors[group], weight='bold')
            else:
                ax.text(-(xPixels[group] - offsetX + sign(cornersX[3]) * off),
                        -(yPixels[group] - offsetY + sign(cornersY[3]) * off),
                        ccdCode, fontsize=15, color=colors[group])

    # Plot center of group

    for group in range(numGroups):
        plt.plot([-(xPixels[group] - offsetX)], [-(yPixels[group] - offsetY)], mfc=colors[group], marker="o", mec='k', ms=10)
        circ = plt.Circle(((-(xPixels[group] - offsetX)), (-(yPixels[group] - offsetY))),
                          radius=fovPixels, color="none", linewidth=1, label="Group " + str(group + 1))
        ax.add_patch(circ)
        circ.set_edgecolor(colors[group])
        circ.set_facecolor("none")

    # Plot arrow

    xHead, yHead = -200, -200
    if xCCD < 0: xHead *= -1
    elif xCCD < 200: xHead = abs(xCCD)
    if yCCD < 0: yHead *= -1
    elif yCCD < 200: yHead = abs(yCCD)
    ax.arrow(0, 0, xCCD+xHead, 0, head_width=150, head_length=abs(xHead), fc='k', ec='k', linewidth=3, zorder=5)
    ax.arrow(0, 0, 0, yCCD+yHead, head_width=150, head_length=abs(xHead), fc='k', ec='k', linewidth=3, zorder=5)
    plt.plot([0], [0], "ko", ms=10, label="CCD origin", zorder=6)
        
    # Plot target star

    plt.plot([-(xPixels[refGroup-1] - offsetX), xCCD], [-(yPixels[refGroup-1] - offsetY), yCCD], 'k--', zorder=6)
    plt.plot(xCCD, yCCD, '*', mfc=colors[refGroup-1], mec='k', ms=19, label='Target star', zorder=7)
    
    # Settings

    ax.set_title("CCD " + refCcdCode + " in Group " + str(refGroup), fontsize=20)
    plt.legend(prop={"size": 12}, bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("Column [pixel]", fontsize=15)
    ax.set_ylabel("Row [pixel]", fontsize=15)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    # Finito!
    
    plt.show()





def drawCCDsInSkyMollweide(raPlatform, decPlatform, solarPanelOrientation,
                           tiltAngle, azimuthAngle, focalPlaneAngle, focalLength,
                           pixelSize, normal=True, figsize=(9,5)):

    """Project and plot the 4 CCDs of 1 camera on the sky
    
    TODO: - Does not work yet for the fast cams
          - Does not take distortion into account yet

    Parameters
    ----------
    raPlatform : float
        Right Ascension of platform pointing [rad]
    decPlatform : float
        Declination of platform pointing [rad]
    titlAngle : float
        Tilt angle of camera [rad]
    azimuthAngle : float
        Azimuth angle of camera [rad] 
    solarPanelOrientation : float
        Orientation of solar panel: (0,pi/2,pi,3pi/2) for quarters (Q1,Q2,Q3,Q4) [rad]
    tiltAngle : float
        Tilt (declination) angle of the telescope w.r.t. platform z-axis [rad]
    azimuthAngle : float
        Azimuth angle of the telescope on the platform [rad]
    focalPlaneAngle: float
        Angle between the Y_TL axis and the Y_FP axis: gamma_FP rad]
    focalLength : float
        Focal length of the camera [mm]
    pixelSize : float
        Pixel size [micron]
    normal : bool
        True for the normal camera configuration, False for the fast cameras

    Return
    ------
    None
    """

    # Select the proper CCD codes depending for either normal or fast cameras

    if normal:
        ccdCodes = ['1', '2', '3', '4']
    else:
        ccdCodes = ['1F', '2F', '3F', '4F']


    # Set up the colors to be used to draw each CCD.
    # Different CCDs have different colors.

    color = {'1': 'b', '1F': 'b', '2': 'r', '2F': 'r',
             '3': 'g', '3F': 'g', '4': 'k', '4F': 'k'}

    # Set up the figure

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111, projection="mollweide")
    axes.grid(True)

    # Plot each of the 4 CCDs

    for ccdCode in ccdCodes:

        # Get the focal plane FP' coordinates of the CCD corners  [mm]

        cornersXmm, cornersYmm = rf.computeCCDcornersInFocalPlane(ccdCode, pixelSize)

        # Compute the equatorial sky coordinates [rad] from the the focal plane FP' coordinates [mm] of the corners

        ra, dec = rf.focalPlaneToSkyCoordinates(cornersXmm, cornersYmm,
                                                raPlatform, decPlatform, solarPanelOrientation,
                                                tiltAngle, azimuthAngle,
                                                focalPlaneAngle, focalLength)

        # Repeat the coordinates of the 1st corner, to plot a nice closed loop
        # Convert from radians to degrees

        ra  = np.append(ra, ra[0])
        dec = np.append(dec, dec[0])

        # The sky projection expects a longitude in [-pi, +pi] rather than [0, 2* pi]
        # Moreover, the longitude should be reversed so that East is to the left

        ra[ra>np.pi] -= 2 * np.pi
        ra = -ra

        axes.plot(ra, dec, c=color[ccdCode])

        # Overplot the row closest to the readout register with a thicker line

        axes.plot([ra[0], ra[1]], [dec[0], dec[1]], c=color[ccdCode], linewidth=3)

    # Change the tick labels so that they are 0->360, rather than -180->+180

    tickLabels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tickLabels = np.remainder(tickLabels+360, 360)
    axes.set_xticklabels(tickLabels)

    # Add axis labels

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.tight_layout()
    plt.draw()

    # That's it

    return fig, axes





def drawStarsInSkyMollweide(fig, ra, dec):

    """Project and plot the stars with the given RA and Dec on the sky.
    
    Parameters
    ----------
    ra : float
        Right ascension of the stars [deg]
    dec : float
        Declination of the the stars [deg]

    Return
    ------
    None
    """

    # Set up the figure

    axes = fig.add_subplot(111, projection="mollweide")
    axes.grid(True)

    raRadians  = []
    decRadians = []

    for index in range(len(ra)):
        raRadians.append(-ra[index] * np.pi / 180.0)
        decRadians.append(dec[index] * np.pi / 180.0)

    axes.plot(raRadians, decRadians, 'ko')

    # Change the tick labels so that they are 0->360, rather than -180->+180

    tickLabels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tickLabels = np.remainder(tickLabels+360, 360)
    axes.set_xticklabels(tickLabels)

    # Add axis labels

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.draw()

    # That's it

    return axes





def drawStarsInSkyAitoff(raStars, decStars, column=None, skymapFile=None, color='r', ms=1,
                         cbarLabel=None, cbarOrientation=None, cbarMap='rainbow',
                         title=None, fs=15, figsize=(13, 9)):

    """Project a catalog of stars on the sky in a Aitoff Galactic projection.

    This plot uses the astropy library to make the ICRS to Galactic coordinate
    transformation together with a nice Galactic background image. To show the plot
    it is necessary to introduce a "plt.show()" after the function call. This module
    scales the scatter plot markersize of the stars according to their sample size.

    Parameters
    ----------
    fig : object
        Figure matplotlib.pyplot object to define e.g. figsize
    raStars : list, array
        Right ascension of stars [deg]
    decStars : list, array
        Declination of stars [deg]
    magStars : list, array
        Magnitudes of stars
    cbarOrientation : str
        Colorbar orientation. Default 'horizontal' else 'vertical'
    cbarMap : str
        Colormap of colorbar. Default 'rainbow'
    figsize : list
        Matplotlib figsize object

    Return
    ------
    axes : object
        Axes matplotlib.pyplot handle object to be modified by the user
    """

    # Ignore warnings for this specific plot
    import warnings
    warnings.filterwarnings("ignore")
    
    # Generate figure object

    fig = plt.subplots(figsize=figsize)
    
    # Convert coordinates from ICRS to Galactic using astropy
    
    gal = SkyCoord(raStars, decStars, frame='icrs', unit=u.deg)
    gal = gal.galactic

    # Show title if requested

    if title:
        # Aitoff projection in Galactic coordinates
        plt.title(title, fontsize=fs+2, y=1.02)

    
    # Plot Aitoff projection in Galactic coordinates

    fig, ax = fig

    # Plot Galactic map as background (e.g. Gaia DR3)
    
    if skymapFile is None:
        skymap = plt.imread(f'{os.getenv("PLATO_PROJECT_HOME")}/docs/figures/gaiaDR3.png')
    else:
        skymap = plt.imread(skymapFile)
    ax.imshow(skymap)
    
    # Add the sky projection ontop as transparent layer
    
    axes = fig.add_subplot(111, projection='aitoff', facecolor='none')
        
    # Plot the targets on the sky (autumn_r, rainbow)

    if column is not None:

        if column.name in ['ncam', 'ncams']:
            # Fetch custom discrete colorbar used by matplotlib
            sep = 1
            cbins = np.arange(column.min(), column.max()+2, sep)
            ticks = cbins - 0.5
            norm  = discretizeColorbar(cbins=cbins, cmap=cbarMap)
            im = plt.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian, c=column,
                             s=ms, cmap=cbarMap, norm=norm, zorder=3)
            cbarax = fig.add_axes([0.25, 0.08, 0.525, 0.03])
            cbar = plt.colorbar(im, orientation='horizontal', cax=cbarax,
                                ticks=ticks, boundaries=cbins, format='%1i')
        else:
            im = plt.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian, c=column,
                             s=ms, cmap=cbarMap, zorder=3)
            
            # Vertical or horizontal colorbar showing magnitudes
            if cbarLabel is None:
                cbarLabel = r'PLATO passband, $P$'            

            if cbarOrientation == 'vertical':
                cbarax = fig.add_axes([0.805, 0.2, 0.02, 0.57])
                cbar = plt.colorbar(im, orientation='vertical', cax=cbarax, extend='both')
            else:
                cbarax = fig.add_axes([0.25, 0.08, 0.525, 0.03])
                cbar = plt.colorbar(im, orientation='horizontal', cax=cbarax, extend='both')

        # Set colobar for all
        cbar.set_label(cbarLabel, fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
        
    else:
        im = plt.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian, c=color,
                         s=ms, zorder=3)
        
    # Change the tick labels so that they are 0->360, rather than -180->+180
    
    tickLabels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tickLabels = np.remainder(tickLabels+360, 360)
    axes.set_xticklabels(tickLabels)

    # Change y ticks and remove last to make space for title
    
    tickLabels = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, ''])
    axes.set_yticklabels(tickLabels)

    # Change color of x tick labels
    
    axes.tick_params(axis='x', colors='w')

    # Increase x and y tick labels
    
    axes.xaxis.set_tick_params(labelsize=fs+1)
    axes.yaxis.set_tick_params(labelsize=fs)

    # Add axes labels
    
    axes.set_xlabel(r'Longitude, $l$ [deg]', fontsize=fs)
    axes.set_ylabel(r'Latitude, $b$ [deg]', fontsize=fs)

    # Set grid and remore outer ticks (if set by default)
    
    axes.grid(True, alpha=0.3)
    ax.axis('off')
    plt.draw()
    plt.tight_layout()
    
    # That's it
    
    return fig, axes





def skyProjection(fig, longitude, latitude, origin=0, projection="mollweide"):

    """Plot sky projection.
    
    This function plots the sources with coordinates longitude & latitude 
    (equatorial, galactic, etc.) in the projected sky.

    Parameters
    ----------
    longitude : int, float
        Longitude coordinate in [0, 360] [deg]
    latitude : int, float
        Latitude coordinate in [-90, +90] [deg]
    fig : matplotlib obejct
        Figure object ebing the output of plt.figure()
    origin : int, float
        Longitude value in the center of the plot [deg]
    projection : str
        Either 'mollweide', 'aitoff', 'hammer', or 'lambert'

    Return
    ------
    axes : plot object
        Matplotlib object axis handle object.
    """

    # Shift the longitude values around the origin

    longitude = np.remainder(longitude+360-origin, 360) 

    # Rescale the range from [0,360] to [-180, +180]

    longitude[longitude > 180] -= 360

    # Reverse the longitude so that East is to the left

    longitude = -longitude

    # Create the plot

    axes = fig.add_subplot(111, projection=projection)
    axes.grid(True)

    # Adapt the tick labels on the x-axis to take into account the origin shift

    tickLabels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tickLabels = np.remainder(tickLabels+360+origin, 360)
    axes.set_xticklabels(tickLabels)

    # Plot the sources

    axes.scatter(np.radians(longitude), np.radians(latitude), s=3)

    # That's it!

    return axes



            
def plotPlatoFOV(pointingField, system="galactic", fovSize=29,
                 # Settings without colorbar
                 raStars=0, decStars=0, magStars=None, ms=2, aa=1, ec='k',
                 # Settings with colorbar
                 c=None, clabel=None, cmap='Spectral', s=40, lw=0.1,
                 # Settings for add-ons
                 ncamStars=False, ncamMap='PIC210',
                 showGalactic=False, aa_gal=0.01,
                 showGroups=False, showFcamFOV=False,
                 # General settings
                 showLegend=False, title=None, fs=20, figsize=(9,9)):

    """Plot a PLATO pointing field in the sky.

    Given a pointing this function plots the PLATO FOV clearly shwoing
    the N-CAM visibility of 6, 12, 18, and 24 cameras. This function is
    used by 'picsim'.

    Parameters
    ---------
    pointingField : str
        The desired PLATO pointing field: 'SPF' or 'NPF'
    raSatrs : float
        Right Ascension of the stars [deg]
    decStars : float
        Declination of stars [deg]
    magStars : float
        Magnitudes of stars [PLATO passband]
    system : str
        Sky reference system as used by Astropy: 'icrs' or 'galactic'
    showGroups : bool
        Flag to show pointing of each camera group (default: False)
    showFcamFOV : bool
        Flag to show FOV of the F-CAMs (default: False)
    ncamStars : bool
        Flag to show N-CAM visibility of PIC stars (default: True)
    title : str
        Title for plot if requested
    fovSize : int, float
        Radius of the plotted FOV [deg] 
    fs : int
        Fontsize of the labels in the plot

    Returns
    -------
    fig : object
        Axes matplotlib.pyplot handle object to be modified by the user
        Use fig..savefig('<plot.png>', bbox_inches='tight', dpi=200)
    """

    # NOTE ligo skymap is only usable with "poetry install --with platonium"

    import ligo.skymap.plot

    # Fetch N-CAM map if requested

    if ncamMap == 'PIC210':
        if pointingField not in ['LOPS2']:
            ut.errorcode('error', 'Not valid pointing for PIC210! Choose [LOPS2]')
    elif ncamMap in ['PIC200', 'PLATO-CS']:
        if pointingField not in ['LOPS2', 'LOPN1']:
            ut.errorcode('error', f'Not valid pointing for {ncamMap}! Choose [LOPS2, LOPN1]')        
    elif ncamMap == 'PIC110':
        if pointingField not in ['SPF', 'NPF']:
            ut.errorcode('error', 'Not valid pointing for PIC110! Choose [SPF, NPF]')
    else:
        ut.errorcode('error', 'Not valid N-CAM map! Choose [PIC210, PIC200, PIC110, PLATO-CS]')
                
    # Select field [deg]

    alpha, delta, kappa = ut.getPointingField(pointingField) 
    PF = SkyCoord(alpha, delta, frame='icrs', unit='deg')
    if system == 'icrs':    
        view = 'astro'
        xPF, yFP = PF.ra.deg, PF.dec.deg
    elif system == 'galactic':
        view = system
        PF = PF.transform_to(system)
        xPF, yPF = PF.l.deg, PF.b.deg
        
    # Define figure
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=f'{view} degrees zoom', center=PF,
                  radius=f'{fovSize} deg', rotate='180 deg')
    tax = ax.get_transform('world')
    
    # Plot N-CAM visibility flower

    if ncamStars is not False:

        # Load PIC target stars
        if isinstance(ncamStars, pd.DataFrame):
            df = ncamStars
        else:
            idir = os.getenv('PLATO_PROJECT_HOME') + '/inputfiles/data_picsim'
            if ncamMap in ['PIC210', 'PIC200', 'PIC110']:
                df = pd.read_feather(f'{idir}/{ncamMap}_{pointingField}_targets.ftr')
            elif ncamMap == 'PLATO-CS':
                df = pd.read_feather(f'{idir}/PlatoCS_NCAM_{pointingField}.ftr')
                df = df.loc[::5]
            
        # NOTE Backward compatible
        try:
            ncam = df.ncam
        except AttributeError:
            ncam = df.ncams

        # Plot sources after N-CAM visibility
        PF06 = df[ncam == 6]
        PF12 = df[ncam == 12]
        PF18 = df[ncam == 18]
        PF24 = df[ncam == 24]
        starPF06 = SkyCoord(PF06.ra*u.deg, PF06.dec*u.deg, frame='icrs', unit='deg')
        starPF12 = SkyCoord(PF12.ra*u.deg, PF12.dec*u.deg, frame='icrs', unit='deg')
        starPF18 = SkyCoord(PF18.ra*u.deg, PF18.dec*u.deg, frame='icrs', unit='deg')
        starPF24 = SkyCoord(PF24.ra*u.deg, PF24.dec*u.deg, frame='icrs', unit='deg')
        if system == "icrs":
            x06, y06 = starPF06.ra.deg, starPF06.dec.deg
            x12, y12 = starPF12.ra.deg, starPF12.dec.deg
            x18, y18 = starPF18.ra.deg, starPF18.dec.deg
            x24, y24 = starPF24.ra.deg, starPF24.dec.deg
        elif system == "galactic":
            starPF06 = starPF06.transform_to(system)
            starPF12 = starPF12.transform_to(system)
            starPF18 = starPF18.transform_to(system)
            starPF24 = starPF24.transform_to(system)
            x06, y06 = starPF06.l.deg, starPF06.b.deg
            x12, y12 = starPF12.l.deg, starPF12.b.deg
            x18, y18 = starPF18.l.deg, starPF18.b.deg
            x24, y24 = starPF24.l.deg, starPF24.b.deg
        ax.plot(x06, y06, '.', c='skyblue',     transform=tax, ms=ms, zorder=1)
        ax.plot(x12, y12, '.', c='deepskyblue', transform=tax, ms=ms, zorder=1)
        ax.plot(x18, y18, '.', c='dodgerblue',  transform=tax, ms=ms, zorder=1)
        ax.plot(x24, y24, '.', c='royalblue',   transform=tax, ms=ms, zorder=1)

    # Plot transparent layer of Galactic footprint

    if showGalactic:
        skip = 5
        df = pd.read_feather(f'{idir}/PlatoCS_NCAM_{pointingField}.ftr')
        # if ncamMap == 'PLATO-CS': 
        #     df = pd.read_feather(f'{idir}/PlatoCS_NCAM_{pointingField}.ftr')
        #     skip = 5
        # else:
        #     df = pd.read_feather(f'{idir}/PlatoGO_PIC210_LOPS2.ftr')
        #     skip = 8
        df = df.loc[::skip]
        # Convert to galactic coordinates
        starPF = SkyCoord(df.ra*u.deg, df.dec*u.deg, frame='icrs', unit='deg')
        starPF = starPF.transform_to('galactic')
        ax.scatter(starPF.l.deg, starPF.b.deg, transform=ax.get_transform('world'),
                   s=2, alpha=aa_gal, marker='.', c='k', ec='none', zorder=2)
        
    # Plot stars and add legend scaled to the stellar magnitudes
    
    if magStars is not None and len(magStars) > 0:
        maxMarkerSize = 30
        dm = (max(magStars) - magStars) * maxMarkerSize
        mag_range = np.arange(min(magStars), max(magStars)).astype(int)
        dm_range  = (max(magStars) - mag_range) * maxMarkerSize/10
        mark, color = 'o', 'orange'
        handle = [plt.plot([],[], "o", c='gray', ms=dm_range[i], ls="")[0]
                  for i in range(len(dm_range))]
        ax.legend(handles=handle, labels=mag_range.tolist(), loc='upper right',
                  title=r"P [mag]", fontsize=16, title_fontsize=16)
    else:
        dm, mark, color = 20, '.', 'none'
        
    # Plot all stars

    if raStars is not None:
        starPF = SkyCoord(raStars*u.deg, decStars*u.deg, frame='icrs', unit='deg')

        # Select system
        if system == 'icrs':
            xStarPF, yStarPF = starPF.ra.deg, starPF.dec.deg
        elif system == 'galactic':
            starPF = starPF.transform_to(system)
            xStarPF, yStarPF = starPF.l.deg, starPF.b.deg

        # Plot
        if c is None:
            scatter = ax.scatter(xStarPF, yStarPF, s=dm, c=color, alpha=aa,
                                 transform=tax, marker=mark, ec=ec, lw=lw, zorder=5)
        else:
            scatter = ax.scatter(xStarPF, yStarPF, s=s, c=c, cmap=cmap,
                                 transform=tax, marker=mark, ec=ec, lw=lw, zorder=5)
            cbar = plt.colorbar(scatter, extend='both', pad=0.01, shrink=0.8)
            cbar.set_label(clabel)
            
    # Plot pointing of each camera group

    if showGroups:
        # Fetch pointing of N-CAM groups
        raGroups, decGroups = rf.getCameraGroupCoordinates(np.deg2rad(alpha),
                                                           np.deg2rad(delta),
                                                           np.deg2rad(kappa))
        ncamPointing = SkyCoord(np.rad2deg(raGroups)*u.deg,
                                np.rad2deg(decGroups)*u.deg,
                                frame='icrs', unit='deg')
        
        # Plot pointing of each camera group
        msGroups = 8
        for i, c in zip(range(4), ['b', 'limegreen', 'yellow', 'r']):
            if system == 'icrs':
                ncamX, ncamY = ncamPointing[i].ra.deg, ncamPointing[i].dec.deg
            elif system == 'galactic':
                ncamPointing = ncamPointing.transform_to(system)
                ncamX, ncamY = ncamPointing[i].l.deg, ncamPointing[i].b.deg
            ax.plot(ncamX, ncamY, 'o', c=c, mec='k', ms=msGroups,
                    transform=tax, zorder=6, label=f'Group {i+1}')
        ax.plot(xPF, yPF, '*', c='magenta', mec='k',
                ms=msGroups+8, transform=tax, zorder=6)

    # Plot F-CAM FOV as cicle 
        
    if showFcamFOV:
        ax.scatter(xPF, yPF, s=8e4, marker='o',
                   edgecolor='magenta', facecolor='none', linewidth=1.5,
                   transform=ax.get_transform(system), zorder=6)

    # Add-on's
    
    ax.scalebar((0.03, 0.03), 10 * u.deg).label()
    ax.compass(0.97, 0.03, 0.1)
    ax.grid(color='gray')
    
    # Settings

    if showLegend and showGroups:
        ax.legend(loc='upper right')    
    if title is not None:
        ax.set_title(title, fontsize=fs+2, pad=10)
    if system == 'icrs':        
        xlab, ylab = 'RA', 'Dec'
    elif system == 'galactic':
        xlab, ylab = r'Longitude, $l$', 'Latitude, $b$'
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs)
    plt.tight_layout()
    
    # Return figure
    
    return fig, ax


def plot_aitoff(df_agn, df_all=False, df_lop=False, df_best=False, NED=False):
    """Function to generate plot galactic aFetch Gaia info for each source in data frame.    
    """    
    if df_best is not False:
        df = df_best
    elif df_lop is not False:
        df = df_lop
    elif df_all is not False:
        df = df_all
    else:
        df = df_agn
    title = (f'Total: {df.shape[0]}, ' +
             f'LOPN1: {df[df.b > 0].shape[0]}, ' + 
             f'LOPS2: {df[df.b < 0].shape[0]}')
    # Plot PLATO AGNs
    fig, ax = drawStarsInSkyAitoff(
        df_agn.ra, df_agn.dec, column=df_agn.ncam, cbarMap='Blues',
        cbarLabel=r'N-CAM visibility, $n_{\rm NCAM}$',
        title=title, fs=13, figsize=(10,7))
    # Plot all candidates
    if df_all is not False:
        if NED:
            ra, dec = df_all.RA, df_all.Dec
        else:
            ra, dec = df_all.ra, df_all.dec
        gal = SkyCoord(ra, dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='k', marker='o', s=20, ec='w', lw=0.7, zorder=4)
    # Plot candidates within LOPs
    if df_lop is not False:
        gal = SkyCoord(ra, dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='purple', marker='o', s=20, ec='w', lw=0.8, zorder=5)
    # Plot best candidates within LOPs
    if df_best is not False:
        gal = SkyCoord(df_best.ra, df_best.dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='purple', marker='o', s=20, ec='w', lw=0.8, zorder=5);
    return fig, ax

#--------------------------------------------------------------#
#                     POINTING ERROR SOURCES                   #
#--------------------------------------------------------------#


def plotYawPitchRollTimeSeries(time, signals, units=["days", "arcsec"],
                               title=False, ylim=False, figsize=(9,10)):

    """Plot the time series of yaw, pitch, and roll for both AOSC jitter and thermo drift.
    
    Along with the time series plots the Root-Mean-Square (RMS) are calculated and plotted in each
    planel, respectively.

    Parameters
    ----------
    time : ndarray
        Array of time points. Make sure time units matches label.
    signals : ndarray, list-ndarray
        List or array of Yaw, Pitch, and Roll time series. Make sure that ampltide unit matches label.
    units : list-str
        List of strings of physical unit: ['time-unit', 'ampltude-unit']
    title : str (optional)
        Title in a string
    ylims : list-str (optional)
        List with ymin and ymax limits

    Return
    ------
    axes : object
        Axes matplotlib.pyplot handle object to be modified plot
    """

    # Datasets to loop over

    try:
        numData = len(signals)
    except:
        numData = 1

    # Handle yaxis limits

    if ylim is False:
        ylim = np.max(np.abs(signals))

    # Adjust linewidth after data

    if len(time) < 1e3: lw = 1
    else: lw = 0.5

    # Make plot

    fig = plt.figure(figsize=figsize)

    labels = ['Yaw', 'Pitch', 'Roll']
    colors = colors_sea

    for plot in range(numData):

        ax = fig.add_subplot(numData, 1, plot+1)

        # Make sure that time series is residuals around zero

        signals[plot] -= np.median(signals[plot])

        # Plot timeseries

        ax.plot(time, signals[plot], '-', c=colors[plot], lw=lw)

        # Add root-mean-square lines

        rms = np.sqrt(np.mean(signals[plot]**2))
        ax.axhline(+rms, c='k', ls='--', lw=0.7, label='RMS = {0:.3f} {1}'.format(rms, units[1]))
        ax.axhline(-rms, c='k', ls='--', lw=0.7)
        ax.legend(loc='upper right')

        # Latter settings

        ax.set_ylabel('{0} [{1}]'.format(labels[plot], units[1]))
        ax.set_xlim(np.min(time), np.max(time))
        ax.set_ylim(-ylim, +ylim)

        # Remove tick labels on x axis except for last plot

        if plot < numData-1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time [{0}]'.format(units[0]))

        # Title

        if title and plot == 0: ax.set_title(title)

    # Adjust layout

    plt.tight_layout()
    plt.subplots_adjust(hspace = .001)

    # Finito!

    return fig, ax





def plotYawPitchRollPSD(time, signals, scale=1e-6, carbox=144, title=False,
                        labels=False, xmin=False, ylim=[1e-1, 1e7], misreq=False,
                        figsize=(9,10)):

    """Plot Power Spectral Desity of Yaw, Pitch, and Roll angles.

    This function takes a Yaw, Pitch, and Roll time series and plots the
    Power Spectral Density (PSD) function for each. Alongside the data a
    median filter is plotted with a default carbox length of 144 time points,
    corresponding to 1 hour precision if the time series are given in seconds.

    Parameters
    ----------
    time : narray
        Time points [s]
    signals : narray, list-narray
        Either single signal array or a list of signal arrays
    carbox : int (optional)
        Length of median carbox filter. Default is 3600s/25s = 144
    title : str (optional)
        Title for plot
    labels : list-str (optinal)
        List of string labels where the first is the xlabel and the rest is ylabels
    xmin : float (optional)
        Limit for x min. The x max limit is the Nyquist frequency
    ylim : list-float (optional)
        List of y min and max limit ["y-min", "y-max"]
    misreq : bool (optional)
        If "True" the mission requirements for the AOSC will be plotted alonside the data.

    Return
    ------
    Plot or/and saved plot to PNG.
    """

    # Number of data sets

    numData = len(signals)

    # Find time step

    sampling = time[1]-time[0]

    # Make plot

    labels = ['Yaw', 'Pitch', 'Roll']
    colors = colors_hot

    fig = plt.figure(figsize=figsize)
    
    for plot in range(numData):

        # Create axes objects

        axes = fig.add_subplot(numData, 1, plot+1)

        # Find PSD and median filter

        freq, PSD = periodogram(signals[plot], 1/sampling, scaling='density')
        PSD_med   = median_filter(PSD, carbox)
        perhour   = int(carbox*sampling/3600.)
        freq *= 1e6  # [muHz]
        PSD  *= 1e6
        PSD_med *= 1e6
        
        # Plot results

        axes.plot(freq, PSD,     '-', c=colors[plot], lw=lw, label=labels[plot])
        axes.plot(freq, PSD_med, 'k-', lw=lw+1, label='1h median')

        # Plot mission requirements (from the red book)

        if misreq:
            axes.plot([3e-6*scale, 20e-6*scale], [21.4*scale, 0.23*scale],
                      c='k', linestyle='--', lw=1, label='MPE requirement')
            axes.plot([20e-6*scale, 4e-2*scale], [0.23*scale, 0.23*scale],
                      c='k', linestyle='--', lw=1)

        # Log scaling

        axes.set_xscale("log")
        axes.set_yscale("log")

        # Latter settings

        if plot == 1:
            axes.set_ylabel(r'Amplitude [arcsec$^2$ $\mu$Hz$^{-1}$]')

        # Remove tick labels on x axis except for last plot

        if plot < numData-1:
            axes.tick_params(labelbottom=False)

        # Set x and y limits

        axes.set_xlim(1e1, freq.max())
        axes.set_ylim(ylim[0], ylim[1])
        
        # Remove tick labels on x axis except for last plot

        if plot < numData-1: axes.tick_params(labelbottom=False)

        # Set legends

        axes.legend(loc='best')

        # Set title

        if title is not False and plot == 0:
            axes.set_title(title, fontsize=fs)

    # Remaining

    plt.xlabel(r'Frequency, $\nu$ [$\mu$Hz]')
    plt.tight_layout()
    plt.subplots_adjust(hspace = .001)

    # Finito!

    return fig, axes






def plotYawPitchRollJitter(time, signals, clabel, tpoint=100, lim=0.20,
                           cmap='gnuplot', plottype='short', title=False):

    """Pointing AOCS jitter illutrator. 

    This function can be used to plot a time series of spacecraft jitter.
    For time series on short time scales, the correlation between yaw, pitch,
    and roll can be illustrated using the the "plottype='short'" option. For
    visualising the entire jitter time series correlated between yaw, pitch,
    and rool use instead the "plottype='long'" option.

    Parameters
    ----------
    time : ndarray
        Array of times.
    signals : list, ndarray
        List or array of individual signal arrays, i.e. [yaw, pitch, roll]
    clabel : str
        String with the color-bar time-label.
    cmap : str
        String specifying the matplotlib color map. Default is "gnuplot"
    plottype : str
        String determine the time scale of the plot. Options are "short" and "long"
    tpoint : int
        Number time points to use for the short time scale visualition only
    title : str
        String of the main title.

    Return
    ------
    None
    """

    # Hardcode valukes
    
    labels = ['Yaw [arcsec]', 'Pitch [arcsec]', 'Roll [arcsec]']
    nticks = 5
    sms = 5
    al  = 1

    # Adjust parameters to a zero-point level

    time = time - np.min(time)
    signals[0] -= np.median(signals[0])
    signals[1] -= np.median(signals[1])
    signals[2] -= np.median(signals[2])

    # PLOT CORRELATIONS FOR SHORT TERM JITTER

    if plottype == 'short':

        fig, ax = plt.subplots(3, 3, figsize=(10, 7))

        for row in range(3):

            # Plots
            
            ax[row, 0].plot(signals[1][tpoint*row:tpoint*(row+1)],
                            signals[0][tpoint*row:tpoint*(row+1)],
                            'k-', alpha=al, lw=lw, zorder=1)
            ax[row, 1].plot(signals[2][tpoint*row:tpoint*(row+1)],
                            signals[0][tpoint*row:tpoint*(row+1)],
                            'k-', alpha=al, lw=lw, zorder=1)
            ax[row, 2].plot(signals[2][tpoint*row:tpoint*(row+1)],
                            signals[1][tpoint*row:tpoint*(row+1)],
                            'k-', alpha=al, lw=lw, zorder=1)
            im0 = ax[row, 0].scatter(signals[1][tpoint*row:tpoint*(row+1)],
                                     signals[0][tpoint*row:tpoint*(row+1)],
                                     c=time[tpoint*row:tpoint*(row+1)],
                                     s=sms, cmap=cmap, zorder=2)
            im1 = ax[row, 1].scatter(signals[2][tpoint*row:tpoint*(row+1)],
                                     signals[0][tpoint*row:tpoint*(row+1)],
                                     c=time[tpoint*row:tpoint*(row+1)],
                                     s=sms, cmap=cmap, zorder=2)
            im2 = ax[row, 2].scatter(signals[2][tpoint*row:tpoint*(row+1)],
                                     signals[1][tpoint*row:tpoint*(row+1)],
                                     c=time[tpoint*row:tpoint*(row+1)],
                                     s=sms, cmap=cmap, zorder=2)

            # Labels
            
            ax[2, 0].set_xlabel(labels[1])
            ax[2, 1].set_xlabel(labels[2])
            ax[2, 2].set_xlabel(labels[2])
            ax[row, 0].set_ylabel(labels[0])
            ax[row, 1].set_ylabel(labels[0])
            ax[row, 2].set_ylabel(labels[1])

            # Remove tick labels on x axis except for last plot
            
            if row < 2:
                ax[row, 0].tick_params(labelbottom=False)
                ax[row, 1].tick_params(labelbottom=False)
                ax[row, 2].tick_params(labelbottom=False)

            # Duplicate settings for each plotted row
            
            for col, im in zip(range(3), [im0, im1, im2]):

                # Axes limits
                
                ax[row, col].set_ylim(-lim, lim)
                ax[row, col].set_aspect('equal', 'box')

                # Force the same number of ticks
                
                ax[row, col].xaxis.set_major_locator(MaxNLocator(nticks))
                ax[row, col].yaxis.set_major_locator(MaxNLocator(nticks))

                # Set grid
                
                ax[row, col].grid(c='gray', ls='-', lw=lw, alpha=al)

                # Color bars
                
                div = make_axes_locatable(ax[row, col])
                cax = div.append_axes('right', size='10%', pad=0.1)
                cbar = plt.colorbar(im, ax=ax[row, col], cax=cax, extend='max')
                cbar.ax.invert_yaxis()
                if im == im2:
                    cbar.set_label(clabel)
                else:
                    cbar.remove()

    # PLOT CORRELATIONS FOR ENTIRE TIMESERIES

    if plottype == 'long':

        # Limits and grid
        
        lim = np.max(np.abs(signals))
        nticks = 6
        time = time/(60*60)

        fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))

        # Plot
        
        ax[0].plot(signals[1], signals[0], 'k-', alpha=al, lw=lw, zorder=1)
        ax[1].plot(signals[2], signals[0], 'k-', alpha=al, lw=lw, zorder=1)
        ax[2].plot(signals[2], signals[1], 'k-', alpha=al, lw=lw, zorder=1)
        im0 = ax[0].scatter(signals[1], signals[0], c=time, s=2, cmap='magma', zorder=2)
        im1 = ax[1].scatter(signals[2], signals[0], c=time, s=2, cmap='magma', zorder=2)
        im2 = ax[2].scatter(signals[2], signals[1], c=time, s=2, cmap='magma', zorder=2)

        # Labels
        
        ax[0].set_xlabel(labels[1])
        ax[0].set_ylabel(labels[0])
        ax[1].set_xlabel(labels[2])
        ax[1].set_ylabel(labels[0])
        ax[2].set_xlabel(labels[2])
        ax[2].set_ylabel(labels[1])

        # Duplicate settings

        for plot in range(3):

            # Adjust axes
            
            ax1[plot].set_xlim(-lim, lim)
            ax1[plot].set_ylim(-lim, lim)
            ax1[plot].set_aspect('equal', 'box')

            # Force the same number of ticks
            
            ax1[plot].xaxis.set_major_locator(MaxNLocator(nticks))
            ax1[plot].yaxis.set_major_locator(MaxNLocator(nticks))

            # Plot grid
            
            ax1[plot].set_axisbelow(False)
            ax1[plot].grid(c='gray', ls='-', lw=lw, alpha=al)

            # We put the colorbar for then remove
            # This is need to keep the same size of each subplot..
            
            div1 = make_axes_locatable(ax1[2])
            cax1 = div1.append_axes("right", size="10%", pad=0.1)
            cbar1 = plt.colorbar(im2, ax=ax1[2], cax=cax1, extend='max')
            cbar1.ax.invert_yaxis()
            if plot == 2:
                cbar1.set_label('Time [hours]')
            else:
                cbar1.remove()

    # Adjust subplot spacing

    fig.tight_layout()
    fig.subplots_adjust(hspace = .001)

    # Finito!

    return fig, ax

#--------------------------------------------------------------#
#                          PHOTOMETRY                          #
#--------------------------------------------------------------#

def plotPhotometry(df, time_unit=False, flux_unit=False, figsize=(8,5)):
    """Function normalize the input flux and change time units to days. 

    Parameters
    ----------
    df : pdarray
        Array containing at least a time and flux column (and potential flux_err)
    time_unit : bool, str
        Pass a string with the unit that should be shown for the time axis.
    flux_unit: bool, str
        Pass a string with the unit that should be shown for the flux axis.
    figsize : list
        Matplotlib 'figsize' object to select image size.

    Return
    ------
    fig, ax : objects
        Axes matplotlib.pyplot handle object to be modified by the user.

    NOTE: Function tailored to PLATOniums output format feather!
    """
    # Create matplotlib object
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the input variable source
    if "flux_err" in df.columns:
        ax.errorbar(df["time"], df["flux"], yerr=df["flux_err"],
                    fmt=".", color='k', ecolor='darkgray', elinewidth=1,
                    capsize=0, alpha=aa, label="Raw flux", zorder=1)
    else:
        ax.plot(df["time"], df["flux"], 'k.', ms=5, alpha=aa, label="Raw flux", zorder=1)

    # Plot a median filter
    if "flux_med" in df.columns:
        ax.plot(df["time"], df["flux_med"], '-', c='royalblue', lw=lw,
                label='1h mdeian', zorder=2)
    
    # Show binned mean points if requested
    if "flux_bin" in df.columns:
        binsize = 1
        ax.plot(df["time"], df["flux_bin"], 'ro', ms=8, mec='k',
                label=f'{binsize}h bins', zorder=3)
            
    # Settings
    ax.set_xlim(df["time"].iloc[0], df["time"].iloc[-1])
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel(f"Flux [{flux_unit}]")
    ax.legend(loc='best')

    # That's it!    
    return fig, ax


def plotNSRvsMagnitude(df,
                       column=False,
                       residuals=False,
                       passband='P',
                       yscale="log",
                       cmap="rainbow",
                       show_ncam_requirements=False,
                       show_ncam_noise_limits=False,
                       show_saturation_limits=False,
                       grid=False,
                       legend=False,
                       cbar_extend=None,
                       figsize=(10,6)):
    """Plot the NSR vs. Magnitude for a star catalogue.

    Parameters
    ----------
    df : pdarray
        Pandas data frame with data.
    column : str
       Column in df that should be used for the (discrete) colorbar.
    residuals : bool
       Allow to show the residuals of two NSR datasets (used in KUL-TN-21)
    yscale : str
       Allow to use either 'linear' or 'log' scale for y axis.
    cmap : str
       Matplotlib colormaps (see docs for possible cmaps).
    grid : bool
       If True a grid is added to plot.
    legend : bool
       If True a legend is added to plot.
    figsize : list
        Matplotlib 'figsize' object to select image size.
     
    Return
    ------
    axes : object
        Axes matplotlib.pyplot handle object to be modified by the user
    """
    # Create matplotlib object
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Secure a small offset in x axis to show data
    dx = (df.mag.max() - df.mag.min()) * 1e-2
    ax.set_xlim(df.mag.min()-dx, df.mag.max()+dx)

    # Define global colormap
    cmap = plt.cm.get_cmap(cmap)
    
    # Set figure labels
    if passband == 'P':
        xlabel = r'PLATO magnitude, $\mathcal{P}$'
    elif passband == 'V':
        xlabel = r'Johnson-Cousin magnitude, $V$'
    else:
        ut.errorcode('error', 'Not valid passband: options ["P", "V"]')
    ax.set_xlabel(xlabel)
    ylabel = r'NSR [ppm h$^{-1/2}$]'
    
    # Handle colorbar and make discrete
    if column in ("group", "camera", "quarter", "ncam", "ncon", "flag"):
        # Fetch custom discrete colorbar used by matplotlib
        sep =1
        cbins = np.arange(df[column].min(), df[column].max()+2, sep)
        ticks = cbins + 0.5
        norm  = discretizeColorbar(cbins=cbins, cmap=cmap)
    else:
        cbins = None
        ticks = None
        norm = None
    
    # Distinguish between the NSR or O-C plot
    if residuals in ("camera", "system"):
        ax.set_ylabel('NSR Residuals [ppm]')
        if yscale == "log":
            im = ax.scatter(df["mag"], df["res"].abs(), s=5, zorder=1,
                            c=df[column], cmap=cmap, norm=norm)
        else:
            im = ax.scatter(df["mag"], df["res"], s=5, zorder=1,
                            c=df[column], cmap=cmap, norm=norm)            
    elif column:
        im = ax.scatter(df["mag"], df["NSR"], s=3, alpha=1, zorder=1,
                        c=df[column], cmap=cmap, norm=norm)
        ax.set_ylabel(ylabel)
    else:
        im = False
        ax.plot(df["mag"], df["NSR"], 'k.', alpha=0.7, zorder=1)
        ax.set_ylabel(ylabel)
        
    # Extra settings for colorbar after image generation
    if column == 'rOA':
        column_label = r'$\vartheta_{\rm OA}$ [deg]'
    else:
        column_label = column
    
    if im and norm is None:
        cb = plt.colorbar(im, extend=cbar_extend, pad=0.01)
        cb.set_label(column_label)
    else:
        # Change label
        if column == 'ncam':
            column_label = r'$n_{\rm CAM}$'
        elif column == 'ncon':
            column_label = r'$n_{\rm contaminants}$'
        # Plot the colorbar
        cb = plt.colorbar(im, extend=cbar_extend, pad=0.01, spacing='proportional',
                          ticks=ticks, boundaries=cbins, format='%1i')
        cb.set_label(column_label)
        cb.minorticks_off()

        # Fewer tick labels for large numbers
        # if (df[column].max() - df[column].min()) > 24:
        #     for label in cb.ax.yaxis.get_ticklabels()[::2]:
        #         label.set_visible(False)
                
    # Plot saturation limits
    if show_saturation_limits:
        ax.axvline(x=8.5, color="k", alpha=0.7, linestyle=':',  zorder=0,
                   label='Onset of saturation')
        ax.axvline(x=7.4, color="k", alpha=0.5, linestyle='--', zorder=0,
                   label='Moderate saturation')

    # Plot requirements
    if show_ncam_requirements == "camera":
        ax.axhline(y=108, c="darkorange", ls="--", label="AOCS camera req.: 108 ppm", zorder=0)
        if yscale == "linear":
            ax.axhline(y=-108, c="darkorange", ls="--")
    elif show_ncam_requirements == "system":
        ax.axhline(y=9, c="red", ls="--", label="AOCS system req.: 9 ppm", zorder=0)
    elif show_ncam_requirements == "multi" and 'ncam' in df:
        for nsr, ncam, color in zip([100, 70, 58, 50],
                                    [6, 12, 18, 24],
                                    [0.0, 0.33, 0.66, 0.999]):
            ax.axhline(y=nsr, color=cmap(color), linestyle="--",
                       label=f"{nsr} ppm for "+r"$n_{\rm CAM}=\,$"+f"{ncam}", zorder=0)
        ax.axvline(x=11, color="k", lw=1, alpha=0.5, linestyle='-', zorder=0)

    # Plot noise limits
    if show_ncam_noise_limits:
        # Magnitude range
        mag = np.linspace(0, 20, 100)
        # Auto selection of level
        if show_ncam_noise_limits == 1:
            ncam  = 1
            level = 'camera'
        else:
            ncam  = 24
            level = 'instrument'
        # Jitter noise
        if legend == 'SMBHB':
            lab = None
        else:
            lab = 'Jitter noise'
        rms = 0.04
        noise_jitter = ut.getJitterNoiseLimitNSR(rms, tdur=3600, camType='normal')
        ax.axhline(y=noise_jitter, c="deeppink", ls="--", lw=1.5, zorder=2,
                       label=lab)
        # Photon noise
        ncams = show_ncam_noise_limits
        noise_photon = ut.getPhotonNoiseLimitNSR(mag, passband=passband, ncam=ncams)
        ax.plot(mag, noise_photon, '-.', c='deeppink', lw=1.5, zorder=2,
                label='Photon noise')
        # Background and readout noise
        noise_background = ut.getBackgroundNoiseLimitNSR(mag, passband=passband, ncam=ncam)
        ax.plot(mag, noise_background, ':', c='deeppink', lw=1.5, zorder=2,
                label='Sky/read noise')
        # Combine and plot
        if legend == 'SMBHB':
            lab = 'Noise model'
        else:
            lab = r"$n_{\rm CAM}=\,$"+f"{show_ncam_noise_limits} noise model"
        noise = np.sqrt(noise_jitter**2 + noise_photon**2 + noise_background**2)
        ax.plot(mag, noise, '-', c='orange', lw=2,  zorder=2,
                label=lab)
        
    # Force all yticks for log plot
    ax.set_yscale(yscale)
    if (df["NSR"].max() - df["NSR"].min()) < 900:
        subticks = [.1, .2, .3, .4, .5, .6, .7, .8, .9] 
        ax.yaxis.get_minor_locator().set_params(numticks=99, subs=subticks)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())

    # Settings
    if grid:   ax.grid(color="lightgray")
    if legend: ax.legend(loc='upper left')

    # Return axes objects
    return fig, ax


#--------------------------------------------------------------#
#                         PICSIM PLOTS                         #
#--------------------------------------------------------------#


def plotTeffvsRadius(ds, df_dK, df_dG, df_dF,
                     sg, df_sgK, df_sgG, df_sgF,
                     df, title, figsize=(8,6)):

    """Distribution of Teff vs. Radius.

    This function plots a Teff vs. R distribution of stars from the PIC,
    and this is used in picsim. The plot make a devision betweem dwarf
    and sub-giant stars, and between F, G, and K spectral type stars.

    Paramters
    ---------
    ds : ndarray
        Pandas data frame with all dwarf stars for the given PIC sample.
    df_dK : ndarray
        Pandas data frame with all K dwarfs from PIC sample.
    df_dG : ndarray
        Pandas data frame with all G dwarfs from PIC sample.
    df_dF : ndarray
        Pandas data frame with all F dwarfs from PIC sample.
    sg : ndarray
        Pandas data frame with all sub-giant stars for the given PIC sample.
    df_sgK : ndarray
        Pandas data frame with all K sub-giants from PIC sample.
    df_sgG : ndarray
        Pandas data frame with all G sub-giants from PIC sample.
    df_sgF : ndarray
        Pandas data frame with all F sub-giants from PIC sample.
    df : ndarray
        Pandas data frame with all selected stars for the given PIC sample.
    title : str
       Add a title string to figure.
    figsize : list
        Matplotlib 'figsize' object to select image size.

    Return
    ------
    fig, ax : objects
        Axes matplotlib.pyplot handle object to be modified by the user.
    """
    
    # Fonts
    
    ms = 3
    if len(ds) > 1e4: da = 0.2
    else: da = 0.0

    # Start plot

    fig, ax = plt.subplots(1,1,figsize=figsize)
    
    # Plots sub-giants
    
    ax.plot(df_sgK['Teff'], df_sgK['R'], 'o', alpha=0.5-da, ms=ms, c='orange',     label=r'K$\,$IV')
    ax.plot(df_sgG['Teff'], df_sgG['R'], 'o', alpha=0.7-da, ms=ms, c='greenyellow',label=r'G$\,$IV')
    ax.plot(df_sgF['Teff'], df_sgF['R'], 'o', alpha=0.7-da, ms=ms, c='skyblue',    label=r'F$\,$IV')

    # Plot dwarfs
    
    ax.plot(df_dK['Teff'], df_dK['R'], 'o', alpha=0.4-da, ms=ms, c='orangered', label=r'K$\,$V')
    ax.plot(df_dG['Teff'], df_dG['R'], 'o', alpha=0.4-da, ms=ms, c='limegreen', label=r'G$\,$V')
    ax.plot(df_dF['Teff'], df_dF['R'], 'o', alpha=0.4-da, ms=ms, c='royalblue', label=r'F$\,$V')

    # Plot selected targets
    
    ax.plot(df['Teff'], df['R'], 'o', mfc='none', mec='k', alpha=0.5, markersize=ms)

    # Compute main sequence devision
    
    dt = np.arange(np.min(ds['Teff']), np.max(ds['Teff']), 10)
    ax.plot(dt, ut.getMainSequenceLimit(dt), 'k-')

    # Settings
    
    ax.set_title(title)
    ax.set_xlabel(r'Effective temperature, $T_{\mathrm{eff}}$ [K]', fontsize=16)
    ax.set_ylabel(r'Stellar radius, $R$ [$R_{\odot}$]', fontsize=16)

    # Legend
    
    order = [0, 3, 1, 4, 2, 5]
    handles, labels = plt.gca().get_legend_handles_labels()
    h = [handles[idx] for idx in order]
    l = [labels[idx] for idx in order]
    ax.legend(h, l, ncol=3, loc='upper left', prop={'size':12},
               columnspacing=0.5, handletextpad=0)

    # Finito!

    return fig, ax

    



def plotStellarSampleDistributions(fig, magRange, magTar, magCon, numConPerTar, distCon):

    """Plot sample distribution of used stellar catalogue.

    This function plots 4 different stellar sample distribution plots of PIC:
    1) Magnitude distribution of PIC targets
    2) Magnitude distribution of PIC contaminants
    3) Number distribution of contaminants per target
    4) Distance distribution of contaminants

    Parameters
    ----------
    mag : list, array
        The stellar target magnitudes
    magCon : list, array
        The stellar contaminant magnitudes
    magRange : list, array
        Upper and lower magnitude limit for input sample
    numConPerTar : list, array
        The number of contaminants (integer) per target star
    distCon : list, array
        Distances of each contaminant star w.r.t. their target

    Return
    ------
    axes : object
        Axes matplotlib.pyplot handle object to be modified by the user
    """

    # Copy figure object not to overwrite it

    fig1, axes = fig

    # Prepare bins and plot magnitude distribution of targets

    magbinTar  = 0.1
    if magRange[1]-magRange[0] > 10: magbinTar = 0.2
    binsizeTar = int((magRange[1] - magRange[0]) / magbinTar) + 1
    binlistTar = np.linspace(magRange[0], magRange[1], binsizeTar)

    axes[0,0].hist(magTar, binlistTar, facecolor='b', edgecolor='b', fill=True, alpha=0.3)
    axes[0,0].set_title('Magnitude distribution of PIC targets')
    axes[0,0].set_xlabel(r'$P$ passband')
    axes[0,0].set_ylabel('Number of stars')
    axes[0,0].locator_params(axis='y', integer=True)
    axes[0,0].tick_params(axis='x', which='minor', bottom=True, top=False)
    axes[0,0].tick_params(axis='x', which='major', bottom=True, top=False)
    axes[0,0].tick_params(axis='y', which='minor', left=False, right=False)
    axes[0,0].tick_params(axis='y', which='major', left=True, right=False)
    axes[0,0].grid(axis='y', color='gray', alpha=0.3)

    # Prepare bins and plot magnitude distribution of contaminants

    magbinCon  = 0.2
    binsizeCon = int((np.max(magCon) - np.min(magCon)) / magbinCon) + 1
    binlistCon = np.linspace(round(np.min(magCon)), round(np.max(magCon)), binsizeCon)

    axes[0,1].hist(magCon, binlistCon, facecolor='m', edgecolor='m', fill=True, alpha=0.3)
    axes[0,1].set_title('Magnitude distribution of PIC contaminants')
    axes[0,1].set_xlabel(r'$P$ passband')
    axes[0,1].set_ylabel('Number of stars')
    axes[0,1].tick_params(axis='x', which='minor', bottom=True, top=False)
    axes[0,1].tick_params(axis='x', which='major', bottom=True, top=False)
    axes[0,1].tick_params(axis='y', which='minor', left=False, right=False)
    axes[0,1].tick_params(axis='y', which='major', left=True, right=False)
    axes[0,1].grid(axis='y', color='gray', alpha=0.3)

    # Prepare bins and plot number distribution of contaminants per target

    numbinCon  = 1 + int(np.max(numConPerTar)/50)
    binsizeNum = int((np.max(numConPerTar) - 0) / numbinCon) + 2
    binlistNum = np.linspace(-0.5, np.max(numConPerTar)+0.5, binsizeNum)

    axes[1,0].hist(numConPerTar, binlistNum, facecolor='g', edgecolor='g',
                   fill=True, log=True, alpha=0.3)
    axes[1,0].yaxis.set_major_formatter(ScalarFormatter())
    axes[1,0].set_title('Number distribution of contaminants per target')
    axes[1,0].set_xlabel('Number of contaminants')
    axes[1,0].set_ylabel('Number of targets')
    axes[1,0].tick_params(axis='x', which='minor', bottom=False, top=False)
    axes[1,0].tick_params(axis='x', which='major', bottom=True, top=False)
    axes[1,0].tick_params(axis='y', which='minor', left=False, right=False)
    axes[1,0].tick_params(axis='y', which='major', left=True, right=False)
    axes[1,0].grid(axis='y', color='gray', alpha=0.3)

    # Prepare bins and plot distance distribution of contaminants wrt their target star

    distbinCon  = 1.0
    binsizeDist = int((np.max(distCon) - np.min(distCon)) / distbinCon) + 2
    binlistDist = np.linspace(round(np.min(distCon)), round(np.max(distCon)), binsizeDist)

    axes[1,1].hist(distCon, binlistDist, facecolor='orange', edgecolor='orange',
                   fill=True, alpha=0.4)
    axes[1,1].set_title('Distance distribution of contaminants')
    axes[1,1].set_xlabel('Distances [arcsec]')
    axes[1,1].set_ylabel('Number of stars')
    axes[1,1].locator_params(axis='y', integer=True)
    axes[1,1].tick_params(axis='x', which='minor', bottom=True, top=False)
    axes[1,1].tick_params(axis='x', which='major', bottom=True, top=False)
    axes[1,1].tick_params(axis='y', which='minor', left=False, right=False)
    axes[1,1].tick_params(axis='y', which='major', left=True, right=False)
    axes[1,1].grid(axis='y', color='gray', alpha=0.3)

    # Layout

    plt.tight_layout()

    # Finito!

    return axes

#--------------------------------------------------------------#
#                        FREQUENCY ANALYSIS                    #
#--------------------------------------------------------------#

def plotPSD(fig, freq, psd, carbox=144, units=False, labels=False, colors=False,
            title=False, xlim=False, ylim=False, linewidth=False, misreq=False):
    """Plots the Power Spectral Density (PSD). 
    
    Alongside the data a median filter is plotted with a default carbox
    length of 144 time points, corresponding to 1 hour precision if the
    time series has a cadence of 25 seconds.

    Parameters
    ----------
    freq : narray
        Frequency points [Hz, mHz, or mizroHz]
    psd : narray, list-narray
        Either single signal array or a list of signal arrays
    carbox : int (optional)
        Length of median carbox filter. Default is 3600s/25s = 144. Also False to ignore.
    title : str (optional)
        Title for plot
    labels : list-str (optinal)
        List of string labels where the first is the xlabel and the rest is ylabels
    xmin : float (optional)
        Limit for x min. The x max limit is the Nyquist frequency
    ylim : list-float (optional)
        List of y min and max limit ["y-min", "y-max"]

    Return
    ------
    Plot or/and saved plot to PNG.
    """
    # Handle the number of input data sets
    if type(psd) == list:
        numData = len(psd)
    else:
        numData = 1
        freq = [freq]
        psd = [psd]

    # Handle axes units
    if units is False:
        units = ['$\mu$Hz', 'ppm']
        scale = 1e6
    else:
        scale = 1

    # Handle colors
    if colors is False:
        colors = ['tomato', 'darkorange', 'gold']
        if numData > 3:
            colors = cm.rainbow(np.linspace(0, 1, numData))

    # Handle linewidths
    if linewidth is False:
        lw = 1
    else:
        lw = linewidth

    # Create axes objects
    axes = fig.add_subplot()

    # Allow plotting multiple PSDs in consecutive subplots
    for plot in np.arange(numData):

        # Plot results
        if labels is False:
            plt.plot(freq[plot], psd[plot], '-', c=colors[plot], lw=lw)
        else:
            plt.plot(freq[plot], psd[plot], '-', c=colors[plot], lw=lw, label=labels[plot])

        # Plot median filter if requested
        if carbox:
            perhour = carbox*25/3600
            PSD_med = median_filter(psd[plot], carbox)
            plt.plot(freq[plot], PSD_med, 'k-', lw=lw+1, label='{0}h median '.format(perhour))

    # Plot mission requirements (from the red book)
    if misreq:
        plt.plot([3e-6*scale, 20e-6*scale], [21.4*scale, 0.23*scale],
                 c='k', linestyle='--', lw=1, label='MPE requirement')
        plt.plot([20e-6*scale, 4e-2*scale], [0.23*scale, 0.23*scale],
                 c='k', linestyle='--', lw=1)

    # Log scaling
    plt.xscale("log")
    plt.yscale("log")

    # Latter settings
    plt.ylabel(r'PSD [{}$^2$ {}'.format(units[1], units[0])+'$^{-1}$]')

    # Settings
    if xlim is not False:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not False:
        plt.ylim(ylim[0], ylim[1])
    if title is not False and plot == 0:
        plt.title(title, fontsize=fs)
    if labels is not False or misreq is True:
        plt.legend(loc='best')

    # Remaining
    plt.xlabel(r'Frequency [{}]'.format(units[0]))
    plt.tight_layout()
    plt.subplots_adjust(hspace = .001)
    plt.grid()

    # Finito!
    return axes


def plotPeriodogram(df0, dm0=None, scaling='density', lw=0.5,
                    c='k', label='Simulation',
                    cm='orange', label_model='Model',
                    show_misreq=True, noise_level=None,
                    numax=False, title=False, logx=True, logy=True,
                    figsize=(9,6)):
    """Plot Power Spectral Density (PSD) for solar-like star.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with observed light cuvre (time [s], flux [norm]). 
    dm : pd.DataFrame
        Data frame with model light curve (time [s], flux [norm]).
    scaling : str
        scipy.periodogram scaling argument {'density', 'spectrum'}.
    **kwargs

    Return
    ------
    matplotlib.pyplot handles (fig, ax)
    """
    # Convert flux to ppm
    df = df0.copy()
    df.time = df.time / 1e6
    df.flux = (df.flux - 1) * 1e6
    # Compute PSD up to Nyquist frequency
    sampling = np.diff(df.time)[0]
    freq, psd = periodogram(df.flux, 1/sampling, scaling=scaling)

    # Start plotting data and model
    fig, ax = plt.subplots(1, 1, figsize=figsize)    
    ax.plot(freq, psd, '-', c=c, lw=lw, label=label)
    if isinstance(dm0, pd.DataFrame):
        # Convert units
        dm = dm0.copy()
        dm.time = dm.time / 1e6
        dm.flux = (dm.flux - 1) * 1e6
        # Compute PSD up to Nyquist frequency
        sampling = np.diff(dm.time)[0]
        freq_model, psd_model = periodogram(dm.flux, 1/sampling, scaling=scaling)
        ax.plot(freq_model, psd_model, '-', c=cm, lw=lw, label=label_model, alpha=0.7)

    x1, x2, x3 = 3, 20, freq[-1]
    # Plot mission requirements (from the red book)
    if show_misreq:
        lw = 1.5
        var = 3600e-6 * np.array([225, 103, 71, 58, 50])**2
        y1 = var**2
        plt.plot([x1, x3], [var[0],    var[0]], linestyle='-', lw=lw, c='m', label='Random Noise')
        plt.plot([x1, x2], [var[1]+y1[1], var[1]], linestyle='-', lw=lw, label='Req. 06 N-CAM', c='b')
        plt.plot([x1, x2], [var[2]+y1[2], var[2]], linestyle='-', lw=lw, label='Req. 12 N-CAM', c='lime')
        plt.plot([x1, x2], [var[3]+y1[3], var[3]], linestyle='-', lw=lw, label='Req. 18 N-CAM', c='yellow')
        plt.plot([x1, x2], [var[4]+y1[4], var[4]], linestyle='-', lw=lw, label='Req. 24 N-CAM', c='red')
        plt.plot([x2, x3], [var[1],    var[1]], linestyle='-', lw=lw, c='royalblue')
        plt.plot([x2, x3], [var[2],    var[2]], linestyle='-', lw=lw, c='lime')
        plt.plot([x2, x3], [var[3],    var[3]], linestyle='-', lw=lw, c='yellow')
        plt.plot([x2, x3], [var[4],    var[4]], linestyle='-', lw=lw, c='red')

    # Plot noise level
    if noise_level:
        plt.plot([x1, x3], [noise_level, noise_level],
                 c='c', linestyle='--', lw=2, label='Noise level')
        
    # Settings
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.set_xlabel(r"Frequency, $\nu$ [$\mu$Hz]")
    ax.set_ylabel(r"PSD [ppm$^2$ $\mu$Hz$^{-1}$]")
    ax.legend(ncol=1, loc='best')
    plt.tight_layout()
    return fig, ax



def compute_double_sided_PSD(time_series,#: np.ndarray,
                             time_interval=25,#: Union[int, float] = 25,
                             detrend=False,#: Union[bool, str, int] = False,
                             scipy=True): #bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute Double Sided Power Spectral Density (PSD).

    PSD is simply a PS (Power Spectrum) divided by ENBW (Effective Noise Bandwidth).
    This function computes a double sided PSD, so the negative frequencies are not
    removed. Because of this the PS is not multiplied by a factor of 2 when scaling
    it. The scaling only takes into account the sampling frequency and the the loss
    of energy because of applying a window function. As the window is a boxcar,
    we can simply use the length of the time series instead.

    Args:
        time_series ([np.ndarray]): time series data
        time_interval (int, optional): Time interval of the time series in seconds. Defaults to 25.
        detrend (bool or str or int, optional): Whether to apply de-trending or not.
        scipy (bool, optional): Whether to use SciPy periodogram or NumPy FFT. Defaults to True.

    Returns:
        [tuple]: tuple containing frequency array, PSD array, and time series tuple
    """
    time = np.linspace(start=0,
                       stop=len(time_series),
                       num=len(time_series)) * time_interval

    if scipy:
        frequencies, psd = periodogram(time_series,
                                       fs=1 / time_interval,
                                       window='boxcar',
                                       nfft=None,
                                       detrend=detrend,
                                       return_onesided=False,
                                       scaling='density',
                                       axis=-1)
    else:
        if detrend:
            not_nan_ind = ~np.isnan(time_series)
            if isinstance(detrend, str) and 'linear' in detrend:
                print('Applying linear detrending...')

                m, b, _, _, _ = stats.linregress(
                    time[not_nan_ind], time_series[not_nan_ind])

                time_series = time_series - (m * time + b)
            elif isinstance(detrend, int):
                print(f'Applying polymial order {detrend} detreding...')
                model = np.polyfit(time[not_nan_ind],
                                   time_series[not_nan_ind],
                                   detrend)
                predicted = np.polyval(model, time)

                time_series = time_series - predicted
            else:
                raise NotImplementedError

        fft_of_time_series = np.fft.fft(time_series)

        # scaling - normally this would be 2 / (frequency * window_compensation)
        # here we compute a double sided PSD, so we drop the factor 2
        # instead of frequency we have time interval i.e. 1/frequency
        # because we do not use a window function, instead of having
        # a sum of squared samples of window function, we simply use the
        # lenght of the input time series. Thus, the scaling here is
        # time_interval / length of the data
        # for more details, see
        # https://dsp.stackexchange.com/questions/32187/what-should-be-the-correct-scaling-for-psd-calculation-using-tt-fft
        # https://stackoverflow.com/questions/22338415/scipy-periodogram-terminology-confusion
        scale = time_interval / time_series.shape[-1]

        # psd is the absolute ps (fft of time series) squared
        psd = scale * np.abs(fft_of_time_series)**2

        frequencies = np.fft.fftfreq(time_series.shape[-1], time_interval)

    idx = np.argsort(frequencies)
    psd = psd[idx]
    frequencies = frequencies[idx]

    return frequencies, psd



def create_performance_figure(data, EoL=False,
                              freq_break=20e-6,
                              min_freq=3e-6,
                              max_freq=40e-3,
                              residual_noise_floor=0.68e-6,
                              random_noise_level=3.0e-6,
                              residual_noise_top=50e-6):
    """

    Args:
        data ([np.asarray]):
        EoL Bool
        freq_break ([float], optional): [description]. Defaults to 20e-6.
        min_freq ([float], optional): [description]. Defaults to 3e-6.
        max_freq ([type], optional): [description]. Defaults to 40e-3.
        residual_noise_floor ([float], optional): [description]. Defaults to 0.68e-6.
        random_noise_level ([float], optional): [description]. Defaults to 3.0e-6.
        residual_noise_top ([float], optional): [description]. Defaults to 50e-6.
        output_file ([str], optional): name of the output file.
    """
    if EoL:
        output_file='mission_performance_asd_EoL.png'
    else:
        output_file='mission_performance_asd.png'

    freq, psd = data

    fig = plt.figure(figsize=(12, 8))

    # plot the ASD
    plt.plot(freq, psd, 'gray', alpha=0.5)

    # plot binned ASD
    m = 10
    p = int(freq.size / m)
    num = rebin1d(freq[0:p*m], p) / float(m)
    binned = rebin1d(psd[0:p*m], p) / float(m)

    plt.plot(num[1:], binned[1:], 'black',  lw=2)

    # residual error line
    plt.hlines(y=residual_noise_floor, xmin=freq_break, xmax=max_freq,
               colors='red', linestyles='-')
    # random noise line
    plt.hlines(y=random_noise_level, xmin=min_freq, xmax=max_freq,
               colors='magenta', linestyles='-')
    # slope line from residual to random top level
    x_values = np.linspace(min_freq, freq_break, 2)
    y_values = np.linspace(residual_noise_top, residual_noise_floor, 2)
    plt.plot(x_values, y_values, color='red', linestyle='-')

    # dashed  guide lines
    plt.vlines(x=freq_break, ymin=1e-8, ymax=residual_noise_floor,
               linestyles='dashed', colors='blue')
    plt.vlines(x=max_freq, ymin=1e-8, ymax=residual_noise_floor,
               linestyles='dashed', colors='blue')
    plt.vlines(x=min_freq, ymin=1e-8, ymax=residual_noise_top,
               linestyles='dashed', colors='blue')
    plt.hlines(y=residual_noise_floor, xmin=1e-10, xmax=freq_break,
               linestyles='dashed', colors='blue')
    plt.hlines(y=residual_noise_top, xmin=1e-8, xmax=min_freq,
               linestyles='dashed', colors='blue')

    # texts
    freq_units = r' $\frac{\mathrm{ppm}}{\sqrt{\mu\mathrm{Hz}}}$'
    top_text = f'{int(residual_noise_top*1e6)}' + freq_units
    plt.text(x=min_freq, y=residual_noise_top*1.25,
             s=top_text,
             ha='center')
    random_noise_text = f'Random Noise\n(incl. photonic stellar reference noise)\n{random_noise_level*1e6}' + freq_units
    plt.text(x=5e-4, y=random_noise_level*1.25,
             s=random_noise_text,
             ha='center')
    residual_error_text = f'Residual Errors\n{round(residual_noise_floor*1e6, 2)}' + \
        freq_units
    plt.text(x=5e-4, y=residual_noise_floor*1.25,
             s=residual_error_text,
             ha='center')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1e-6, 1e-1)
    plt.ylim(1e-7, 1e-4)

    # modify x tick points
    ticks = [1e-6, min_freq, 1e-5, freq_break,
             1e-4, 1e-3, 1e-2, max_freq, 1e-1]
    labels = ['$10^{-6}$', str(int(min_freq*1e6)) + '$\mu$Hz', '$10^{-5}$',
              str(int(freq_break*1e6)) + '$\mu$Hz',
              '$10^{-4}$', '$10^{-3}$', '$10^{-2}$',
              str(int(max_freq*1e3)) + 'mHz', '$10^{-1}$']
    plt.xticks(ticks=ticks, labels=labels)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Amplitude Spectral Density $(\mu\mathrm{Hz})^{-\frac{1}{2}}$')

    plt.tight_layout()
    #plt.savefig(output_file, dpi=150)
    

def rebin1d(array, n):
    nr = int(float(array.shape[0]) / float(n))
    return (np.reshape(array, (n, nr))).sum(1)

#--------------------------------------------------------------#
#                        VARSIM PLOTS                          #
#--------------------------------------------------------------#


def plotSED(wvl, wvl1_in, wvl2_in, wvl_equi,
            flux, flux1_in, flux2_in, flux_equi,
            Teff, Teff_upper, Teff_lower):

    """Plot synthetic SED for best model fit.
    """

    fig, ax = plt.subplots(2, 1, figsize=(8,7))

    # Teff labels
    
    Teff0 = r'$T_{\mathrm{eff}}$' + f' = {int(Teff_upper)} K'
    Teff1 = r'$T_{\mathrm{eff}}$' + f' = {int(Teff)} K'
    Teff2 = r'$T_{\mathrm{eff}}$' + f' = {int(Teff_lower)} K'
    ymax0 = flux2_in.max() + flux2_in.max() * 0.1
    cc = ['tomato', 'k', 'orange', 'royalblue']
    
    # Plot the interpolation of the grid

    ax[0].plot(wvl2_in, flux2_in, c=cc[0], lw=1, alpha=1.0, label=Teff0)
    ax[0].plot(wvl,     flux,     c=cc[1], lw=1, alpha=1.0, label=Teff1)
    ax[0].plot(wvl1_in, flux1_in, c=cc[2], lw=1, alpha=1.0, label=Teff2)
    ax[0].fill_between((0,      wvl_equi[0]), 0, ymax0, facecolor='gray', alpha=0.2)
    ax[0].fill_between((wvl_equi[-1], 13000), 0, ymax0, facecolor='gray', alpha=0.2)
    ax[0].set_xlim(1000, 13000)
    ax[0].set_ylim(0, ymax0)
    ax[0].legend()

    # Plot the final equidistant grid used for further calculations

    ymax1 = flux_equi.max() + flux_equi.max() * 0.3
    ax[1].plot(wvl,      flux,      c=cc[1], lw=0.5, alpha=0.9,
               label=f'Interpolated grid: {Teff1}')
    ax[1].plot(wvl_equi, flux_equi, c=cc[3], lw=2.0, alpha=1.0,
               label=r'Equidistant grid in passband')
    ax[1].fill_between((0,      wvl_equi[0]), 0, ymax0, facecolor='gray', alpha=0.2)
    ax[1].fill_between((wvl_equi[-1], 13000), 0, ymax0, facecolor='gray', alpha=0.2)
    ax[1].set_xlim(wvl_equi[0]-1000, wvl_equi[-1]+1000)
    ax[1].set_ylim(0, ymax1)
    ax[1].legend()

    # Settings

    ax[1].set_xlabel(r'Wavelength, $\lambda$ [\AA]')
    laby = r'Flux, $F_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$ sr$^{-1}$]'
    fig.text(-0.01, 0.5, laby, va='center', rotation='vertical')
    plt.tight_layout(h_pad=0.15)
    plt.show()

    # Finito!
    
    return fig, ax





def plotGranOscTimeSeries(time, signal_gran, signal_puls):

    """Plot bolometric luminosity amplitude timeseries.
    """

    signal_total = signal_gran + signal_puls

    # Correct time points from Ms to days

    time = time * 1e6 / 86400
    
    # Plot

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 10), sharex=True)
    ax1.plot(time, signal_gran,  colors_hot[0], linewidth=lw, label = 'Granulation')
    ax2.plot(time, signal_puls,  colors_hot[1], linewidth=lw, label = 'Pulsations')
    ax3.plot(time, signal_total, 'k',           linewidth=lw, label = 'Combined')

    # Limits

    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(signal_gran.min() - signal_gran.std(), signal_gran.max() +signal_gran.std())
    ax2.set_ylim(signal_puls.min() - signal_puls.std(), signal_puls.max() +signal_puls.std())
    ax3.set_ylim(signal_total.min()-signal_total.std(), signal_total.max()+signal_total.std())

    # Labels

    ax1.set_title('Bolometric amplitude time series', fontsize = fs)
    ax3.set_xlabel('Time [days]',           fontsize = fs-2)
    ax1.set_ylabel('Granulation [ppm]',     fontsize = fs-2)
    ax2.set_ylabel('Pulsation [ppm]',       fontsize = fs-2)
    ax3.set_ylabel('Total Amplitude [ppm]', fontsize = fs-2)

    # Extra settings

    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.tight_layout()
    plt.show()





def plotGranOscAmplitudeSpectrum(df, numax, title=False):

    """Plot Power Spectral Density (PSD) for solar-like star.

    Parameters
    ----------
    time : narray
        Time points [s]
    signal_gran : ndarray, pdseries 
        Amplitudes for granulation noise [ppm]
    signal_puls : ndarray, pdseries 
        Amplitudes for stochastic oscillations [ppm]
    title : str (optional)
        Title for plot

    Return
    ------
    matplotlib figure handles
    """

    # Compute frequencies uptil the Nyquist frequency

    sampling = np.diff(df.time)[0]

    # Compute PSD of granulation and oscillations
    freq_gran, psd_gran = periodogram(df.gran, 1/sampling, scaling='density')
    freq_puls, psd_puls = periodogram(df.puls, 1/sampling, scaling='density')
    freq_gran *= 1e6  # [muHz]
    freq_puls *= 1e6  # [muHz]
    psd = psd_gran + psd_puls
    
    # PLOT SEPERATE

    fig, ax = plt.subplots(2, 1, figsize=(9,8))
    
    # Plot global model
    freq_min = 1e2
    freq_max = np.max(freq_gran)
    dex_min = ut.findNearestIndex(freq_puls, freq_min)
    dex_max = ut.findNearestIndex(freq_puls, freq_max)
    psd_min = np.median(psd_puls[dex_min:dex_max])
    psd_max = np.max(psd[dex_min:dex_max]) * 2
    ax[0].plot(freq_gran, psd_gran, "-", c='royalblue', lw=0.3, label="Granulation")
    ax[0].plot(freq_puls, psd_puls, "-", c='orange', lw=0.3, label="Pulsations")
    ax[0].set_xlim(freq_min, freq_max)
    ax[0].set_ylim(psd_min, psd_max)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r"Frequency, $\nu$ [$\mu$Hz]")
    ax[0].set_ylabel(r"PSD [ppm$^2$ $\mu$Hz$^{-1}$]")
    ax[0].legend(ncol=1, loc='upper right', fontsize=16)

    # Plot zoom in on p-modes

    lab1 = "Stochastic oscillations"
    dx = numax * 0.2
    freq_min = numax-dx
    freq_max = numax+dx
    if freq_min < 0: freq_min = 0
    dex_min = ut.findNearestIndex(freq_puls, freq_min)
    dex_max = ut.findNearestIndex(freq_puls, freq_max)
    psd_min = np.median(psd[dex_min:dex_max])
    psd_max = np.max(psd[dex_min:dex_max]) * 2
    ax[1].plot(freq_puls, psd, "-", c='k', lw=0.3, label=lab1)
    ax[1].set_xlim(freq_min, freq_max)
    ax[1].set_ylim(psd_min, psd_max)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r"Frequency, $\nu$ [$\mu$Hz]")
    ax[1].set_ylabel(r"PSD [ppm$^2$ $\mu$Hz$^{-1}$]")
    ax[1].legend(ncol=1, loc='upper right', fontsize=16)

    # Settings
    
    plt.tight_layout()
    plt.show()




def plotPassbandLDC(wvl_int_plato, tran_int_plato, grid_no,
                    mu_trunc, intensity_VTA_trunc, LD_values, ldc):

    """Plot the limb darkening coefficients of the PLATO passband.
    """

    # Import TESS passband

    wvl_tess = np.loadtxt(os.getcwd() + '/data/Passbands/response_tess.txt')[:,0]*10.  # [Å]
    tra_tess = np.loadtxt(os.getcwd() + '/data/Passbands/response_tess.txt')[:,1]      # Norm.
    wvl_int_tess  = np.linspace(wvl_tess[0], wvl_tess[-1], grid_no)
    passband_tess = make_interp_spline(wvl_tess, tra_tess, k=3)
    tran_int_tess = passband_tess(wvl_int_tess)

    # Import Kepler passband

    wvl_kepler  = np.loadtxt(os.getcwd() + '/data/Passbands/response_kepler.txt')[:,0]*10.
    tran_kepler = np.loadtxt(os.getcwd() + '/data/Passbands/response_kepler.txt')[:,1]
    wvl_int_kepler  = np.linspace(wvl_kepler[0], wvl_kepler[-1], grid_no)
    passband_kepler = make_interp_spline(wvl_kepler, tran_kepler, k=3)
    tran_int_kepler = passband_kepler(wvl_int_kepler)

    # Create the plot

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))#13,6))

    # Response functions:

    ax[0].plot(wvl_int_plato,  tran_int_plato,  'r-', label='PLATO')
    ax[0].plot(wvl_int_tess,   tran_int_tess,   'g--', label='TESS')
    ax[0].plot(wvl_int_kepler, tran_int_kepler, 'b:', label='Kepler')
    ax[0].set_xlabel(r'Wavelength, $\lambda$ [Å]')
    ax[0].set_ylabel(r'Norm. Spectral Response, $S_{\lambda}$')
    ax[0].set_title('Bandpass')
    ax[0].legend(fontsize=12)

    # Quadratic LD coeffients fitting:

    lab = 'Model:'+'\n'+r'$u_1$ = %.5s'%ldc[0]+'\n'+r'$u_2$ = %.5s'%ldc[1]
    ax[1].plot(mu_trunc, intensity_VTA_trunc, 'ko', alpha=0.2, label='Data')
    ax[1].plot(mu_trunc, LD_values.model, 'r-', label=lab)
    ax[1].set_xlabel(r'Norm. Wavelength, $\lambda$')
    ax2.set_ylabel(r'Norm. Intensity, $I_{\lambda}$')
    ax2.set_title('Quadratic LD coeffients fitting')
    ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # That's it!
    
    return fig, ax




    
def plotOrbitalPhaseCurve(time, lc_tra, lc_occ, lc_beam, lc_elli,
                          t0, P, dt_c, t_tra_cen, t_tra_tot, t_occ_cen, t_occ_tot,
                          A_beam, A_elli, colors=None, figsize=(13, 10)):

    """Plot a orbital phase curve of exoplanet.
    """

    # Input parameters

    pp = 0.05
    if colors is None:
        colors = ['royalblue', 'limegreen', 'darkorange', 'tomato', 'gold']

    # Convert units
    
    lc_tra  = (lc_tra  - 1) * 1e6
    lc_occ  = (lc_occ  - 1) * 1e6
    lc_beam = (lc_beam - 1) * 1e6
    lc_elli = (lc_elli - 1) * 1e6
    lc_exo  = lc_tra + lc_occ + lc_beam + lc_elli
        
    # Phase-fold light curve
    # TODO Implement ourselves
    
    from PyAstronomy import pyasl
    phase = pyasl.foldAt(time, P, T0=t0)

    # Locate first transit
    
    t_tra_min = t_tra_cen - t_tra_tot
    t_tra_max = t_tra_cen + t_tra_tot
    dex_tra = (time >= t_tra_min) * (time < t_tra_max)

    # Locate first occultation
    
    t_occ_min = t_occ_cen - t_occ_tot
    t_occ_max = t_occ_cen + t_occ_tot
    dex_occ = (time >= t_occ_min) * (time < t_occ_max)

    # Setup

    fig = plt.figure(figsize=figsize)
    
    plt.subplots_adjust(wspace=0.15, hspace=0.20)
    fig.text(0.5, 0.92, 'Time [days]', ha='center', fontsize=fs)
    fig.text(0.5, 0.06, 'Phase', ha='center', fontsize=fs)
    fig.text(0.05, 0.4, 'Relative Flux [ppm]', va='center', rotation='vertical', fontsize=fs)

    # Final light curve in time

    ax0 = fig.add_subplot(4,2,(1,2))
    ax0.axvline(t0,      color='gray', linestyle='--')
    ax0.axvline(t0+dt_c, color='gray', linestyle=':')
    ax0.plot(time, lc_exo, 'k-')
    ax0.xaxis.set_label_position('top')
    ax0.xaxis.tick_top()
    ymin, ymax = getAxesMinMax(y=lc_exo)
    ax0.set_ylim(ymin, ymax)
    ax0.set_xlim(time[0], time[-1])
    ax0.fill_between((t_tra_min, t_tra_max), ymin, ymax, facecolor=colors[0], alpha=0.3)
    ax0.fill_between((t_occ_min, t_occ_max), ymin, ymax, facecolor=colors[1], alpha=0.2)
    ax0.ticklabel_format(style='plain', useOffset=False)
    ax0.set_ylabel('Relative Flux')

    # Transit

    ax1 = fig.add_subplot(4,2,3)
    ax1.plot(time[dex_tra], lc_tra[dex_tra], '-', c=colors[0], label='Transit')
    ax1.legend(loc='upper center')
    x_pos     = t_tra_max - t_tra_tot*1.3
    delta_tra = np.max(lc_tra[dex_tra]) - np.min(lc_tra[dex_tra])
    y_pos     = np.max(lc_tra[dex_tra]) - delta_tra/2.
    ax1.text(x_pos, y_pos, r'$\delta_{\mathrm{tra}}=%.1f$ ppm' % delta_tra, fontsize=fs-4)
    ax1.set_xlim(time[dex_tra][0], time[dex_tra][-1])
    ax1.set_ylim(getAxesMinMax(y=lc_tra))
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    # Occultation and phase curve

    ax2 = fig.add_subplot(4,2,4)
    ax2.plot(time[dex_occ], lc_occ[dex_occ], '-', c=colors[1], label='Occultation')
    ax2.legend(loc='upper center')
    x_pos     = t_occ_max - t_occ_tot*1.3
    delta_occ = np.max(lc_occ[dex_occ]) - np.min(lc_occ[dex_occ])
    y_pos     = np.max(lc_occ[dex_occ]) - delta_occ/2.
    ax2.text(x_pos, y_pos, r'$\delta_{\mathrm{occ}}=%.1f$ ppm' % delta_occ, fontsize=fs-4)
    ax2.set_xlim(time[dex_occ][0], time[dex_occ][-1])
    ax2.set_ylim(getAxesMinMax(y=lc_occ))
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    # Beaming and Ellipsoidal

    sort = np.argsort(phase)
    ax3 = fig.add_subplot(4,2,(5,6))
    ax3.axhline(0.00, color='gray', linestyle='--')
    ax3.axvline(0.00, color='gray', linestyle='--')
    ax3.axvline(0.25, color='gray', linestyle=':')
    ax3.axvline(0.50, color='gray', linestyle='-.')
    ax3.axvline(0.75, color='gray', linestyle=':')
    ax3.axvline(1.00, color='gray', linestyle='--')
    ax3.plot(phase[sort], lc_beam[sort], '-', c=colors[2], ms=ms, label='Beaming')
    ax3.plot(phase[sort], lc_elli[sort], '-', c=colors[3], ms=ms, label='Ellipsoidal')
    ypos_max   = np.max([lc_beam, lc_elli])
    ypos_text0 = ypos_max - ypos_max*2.0*pt
    ypos_text1 = ypos_max - ypos_max*5.0*pt
    ax3.text(0.347, ypos_text0, r'$A_{\mathrm{beam}}=%.3f$ ppm' % A_beam, fontsize=fs-4)
    ax3.text(0.360, ypos_text1, r'$A_{\mathrm{elli}}=%.3f$ ppm' % A_elli, fontsize=fs-4)
    ax3.legend(loc='upper left')
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.set_xlim(0-pp, 1+pp)

    # Combined model

    ax4 = fig.add_subplot(4,2,(7,8))
    ax4.axvline(0.00, color='gray', linestyle='--', zorder=0)
    ax4.axvline(0.25, color='gray', linestyle=':',  zorder=1)
    ax4.axvline(0.50, color='gray', linestyle='-.', zorder=2)
    ax4.axvline(0.75, color='gray', linestyle=':',  zorder=3)
    ax4.axvline(1.00, color='gray', linestyle='--', zorder=4)
    ymin, ymax = getAxesMinMax(y=lc_exo - lc_tra)
    ydif = (ymax-ymin)*pp
    ypos_text = ymax + ydif + ymax*pt
    ax4.text(0.00-0.02, ypos_text, 'Transit',     fontsize=fs-5)
    ax4.text(1.00-0.02, ypos_text, 'Transit',     fontsize=fs-5)
    ax4.text(0.50-0.03, ypos_text, 'Occultation', fontsize=fs-5)
    ax4.text(0.25-0.03, ypos_text, 'Quadrature',  fontsize=fs-5)
    ax4.text(0.75-0.03, ypos_text, 'Quadrature',  fontsize=fs-5)
    yy = lc_exo/(np.max(lc_exo)+np.max(lc_exo*2*pt))
    ax4.plot(phase[sort], lc_exo[sort], 'k-', zorder=5, label='Combined Model')
    ax4.scatter(phase, lc_exo, marker='o', s=5, c=cm.hot(yy), ec='None', zorder=6)
    ax4.set_ylim(ymin-ydif, ymax+ydif)
    ax4.set_xlim(0-pp, 1+pp)
    ax4.legend(loc='upper left')

    # Finito!
    
    return fig, [ax0, ax1, ax2, ax4]



    
    
def plotVarsimLC(lc, figsize=False):

    """Plot noise-less light curve from file produced with varsim.
    """

    # Handle time units
    time = lc.time / 86400.
    colors = ['purple', 'royalblue', 'lightseagreen', 'limegreen',
              'orange', 'tomato', 'deeppink']

    # Check for planet model
    if 'tran' in lc:
        if 'occu' in lc:
            lc.tran += (lc.occu - 1)
            lc = lc.drop(columns=['occu'])
        if 'beam' in lc:
            lc.tran += (lc.beam - 1)
            lc = lc.drop(columns=['beam'])
        if 'elli' in lc:
            lc.tran += (lc.elli - 1)
            lc = lc.drop(columns=['elli'])
    
    # Auto adjust figure
    n = lc.shape[1] - 1
    if not figsize: figsize = (11, 2*n)

    # Start plotting
    fig, ax = plt.subplots(n, 1, figsize=figsize, sharex=True)

    if lc.flux.abs().max() > 0.01:
        norm = 1e3
        lab  = 'ppt'
    else:
        norm = 1e6
        lab  = 'ppm'
    
    for i,signal in zip(range(n), lc.columns[1:]):
        flux = (lc[signal] - 1) * norm
        if signal == 'flux':
            ax[i].plot(time, flux, '-', c='k', label='Combined')
        else:
            ax[i].plot(time, flux, '-', c=colors[i], label=signal)
        ax[i].set_xlim(time.iloc[0], time.iloc[-1])
        ax[i].legend(loc="upper right")

    plt.xlabel('Time [days]')
    fig.text(0.01, 0.5, f'Relative flux [{lab}]', va='center', rotation='vertical')    
    plt.tight_layout(h_pad=0.0, w_pad=1)
    
    return fig, ax


#--------------------------------------------------------------#
#                     PLOTS FOR CORE SCIENCE                   #
#--------------------------------------------------------------#


def plotTimesPlanetsHZ():


    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    # Select range of temperatures

    w = 0.
    e = 0.
    i = (90. * u.deg).to('rad').value

    # Value for F0-M5 main-sequence dwarf stars
    s = ['F0', 'F5', 'G0', 'G2', 'G5', 'K0', 'K5', 'M0', 'M5']
    Teff = np.array([7240, 6420, 5920, 5780, 5610, 5240, 4410, 3800, 3120]) * u.K
    L = np.array([6.0, 2.5, 1.26, 1.0, 0.79, 0.40, 0.16, 0.072, 0.0027]) * u.L_sun
    M = np.array([1.7, 1.3, 1.10, 1.0, 0.93, 0.78, 0.69, 0.60,  0.15]) * u.M_sun
    R = np.array([1.3, 1.2, 1.05, 1.0, 0.93, 0.85, 0.74, 0.51,  0.18]) * u.R_sun

    def timeHZ(Rp):

        # Find effective stellar flux
        Teff_sun = 5780. * u.K
        T = (Teff - Teff_sun).value

        c_inn = [8.1774e-5, 1.7063e-9, -4.3241e-12, -6.6462e-16]
        c_out = [5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16]

        Seff_inn = 1.0140 + c_inn[0]*T + c_inn[1]*T**2 + c_inn[2]*T**3 + c_inn[3]*T**4
        Seff_out = 0.3438 + c_out[0]*T + c_out[1]*T**2 + c_out[2]*T**3 + c_out[3]*T**4

        # Semi-major axis

        a_inn = ((L.value/Seff_inn)**0.5 * u.AU).to('m')   # [AU] => [m]
        a_out = ((L.value/Seff_out)**0.5 * u.AU).to('m')   # [AU] => [m]

        # Orbital period (K3)

        P_inn = np.sqrt(4 * np.pi**2 * a_inn**3 / (c.G * M.to('kg')))  # [s]
        P_out = np.sqrt(4 * np.pi**2 * a_out**3 / (c.G * M.to('kg')))  # [s]


        # fig = plt.figure(figsize=(12,5))
        # # Plot Teff vs. Seff
        # ax1 = fig.add_subplot(131)
        # ax1.plot(Seff_inn, Teff, 'r-')
        # ax1.plot(Seff_out, Teff, 'b-')
        # ax1.set_xlabel(r'$S_{eff}$ ($S_{\odot}$)')
        # ax1.set_ylabel(r'$T_{eff}$ (K)')
        # ax1.invert_xaxis()
        # # Plot Mass vs. Distance
        # ax2 = fig.add_subplot(132)
        # ax2.plot(a_inn.to('AU'), M, 'r-')
        # ax2.plot(a_out.to('AU'), M, 'b-')
        # ax2.set_xscale('log')
        # ax2.set_yscale('log')
        # ax2.set_xlabel(r'Distance, $D$ (AU)')
        # ax2.set_ylabel(r'Stellar mass, $M$ ($M_{\odot}$)')
        # # Plot Mass vs. Distance
        # ax3 = fig.add_subplot(133)
        # ax3.plot(P_inn.to('d'), Teff, 'r-')
        # ax3.plot(P_out.to('d'), Teff, 'b-')
        # ax3.set_xlabel(r'Orbital Period, $P$ (days)')
        # ax3.set_ylabel(r'$T_{eff}$ (K)')
        # # Plot
        # plt.tight_layout()
        # plt.show()
        # exit()

        # Impact parameter: Winn (2014) Eq. 7 & 8

        b_tra_inn = 0#a_inn.to('m')*np.cos(i)/R.to('m') * (1 - e**2)/(1 + e*np.sin(w))
        b_tra_out = 0#a_inn.to('m')*np.cos(i)/R.to('m') * (1 - e**2)/(1 + e*np.sin(w))

        #b_occ_inn = a_out*np.cos(i)/R * (1 - e**2)/(1 - e*np.sin(w))
        #b_occ_out = a_out*np.cos(i)/R * (1 - e**2)/(1 - e*np.sin(w))

        # Transit depth first approximation

        k = Rp.to('R_sun')/R.to('R_sun')

        # Transit times: Winn (2014) Eq. 14, 15 & 16
        # NOTE on circular orbits the transit and occultation times are equal

        x  = np.sqrt(1 - e**2)   # Optimization constant
        e_tra = x/(1 + e*np.sin(w))
        e_occ = x/(1 - e*np.sin(w))

        t_tra_tot_inn = P_inn.to('s')/np.pi * np.arcsin( R.to('m')/a_inn.to('m') * np.sqrt((1 + k)**2 - b_tra_inn**2)/np.sin(i) ) * e_tra / u.rad
        t_tra_tot_out = P_out.to('s')/np.pi * np.arcsin( R.to('m')/a_out.to('m') * np.sqrt((1 + k)**2 - b_tra_out**2)/np.sin(i) ) * e_tra / u.rad

        t_tra_ful_inn = P_inn.to('s')/np.pi * np.arcsin( R.to('m')/a_inn.to('m') * np.sqrt((1 - k)**2 - b_tra_inn**2)/np.sin(i) ) * e_tra / u.rad
        t_tra_ful_out = P_out.to('s')/np.pi * np.arcsin( R.to('m')/a_out.to('m') * np.sqrt((1 - k)**2 - b_tra_out**2)/np.sin(i) ) * e_tra / u.rad

        tau_tra_inn = (t_tra_tot_inn - t_tra_ful_inn)/2.
        tau_tra_out = (t_tra_tot_out - t_tra_ful_out)/2.

        # Occultation times

        t_occ_tot_inn = P_inn.to('s')/np.pi * np.arcsin( R.to('m')/a_inn.to('m') * np.sqrt((1 + k)**2 - b_tra_inn**2)/np.sin(i) ) * e_occ / u.rad
        t_occ_tot_out = P_out.to('s')/np.pi * np.arcsin( R.to('m')/a_out.to('m') * np.sqrt((1 + k)**2 - b_tra_out**2)/np.sin(i) ) * e_occ / u.rad

        t_occ_ful_inn = P_inn.to('s')/np.pi * np.arcsin( R.to('m')/a_inn.to('m') * np.sqrt((1 - k)**2 - b_tra_inn**2)/np.sin(i) ) * e_occ / u.rad
        t_occ_ful_out = P_out.to('s')/np.pi * np.arcsin( R.to('m')/a_out.to('m') * np.sqrt((1 - k)**2 - b_tra_out**2)/np.sin(i) ) * e_occ / u.rad

        tau_occ_inn = (t_occ_tot_inn - t_occ_ful_inn)/2.
        tau_occ_out = (t_occ_tot_out - t_occ_ful_out)/2.

        # Finito!

        return ([t_tra_tot_inn, t_tra_tot_out, tau_tra_inn, tau_tra_out],
                [t_occ_tot_inn, t_occ_tot_out, tau_occ_inn, tau_occ_out],
                [P_inn, P_out])



    # Compute HZ for different planet radii

    #mars = timeHZ(0.5*u.R_earth)
    earth = timeHZ(1.0*u.R_earth)
    neptune = timeHZ(5.0*u.R_earth)
    jupiter = timeHZ(10.0*u.R_earth)

    # Make a pretty table for an Earth analog of transit times

    p = jupiter
    t = PrettyTable(['Type', 'T_tra1 [h]', 'T_tra2 [h]', 'tau_tra1 [min]', 'tau_tra2 [min]', 'P_inn [days]', 'P_out [days]'])
    for i in range(len(s)):
        t.add_row([str(s[i]),
                   '%.2f' % p[0][0].to('h').value[i], '%.2f' % p[0][1].to('h').value[i],
                   '%.2f' % p[0][2].to('min').value[i], '%.2f' % p[0][3].to('min').value[i],
                   '%.2f' % p[2][0].to('d').value[i], '%.2f' % p[2][1].to('d').value[i]])

    # Plot figure showing results

    fig = plt.figure(figsize=(4.3,5))
    ax1 = fig.add_subplot(111)

    for i in range(len(Teff)):
        ax1.hlines(y=Teff.value[i], xmin=0, xmax=27, color='k', linestyle=':', lw=1)
        ax1.text(27.2, Teff.value[i]-50, s[i], fontsize=12)

    #ax1.plot(mars[0][0].to('h'), Teff, 'r-', label=r'$0.5 R_{\oplus}$')
    #ax1.plot(mars[0][1].to('h'), Teff, 'r-')

    ax1.plot(earth[0][0].to('h'), Teff, 'b-', label=r'$1 R_{\oplus}$')
    ax1.plot(earth[0][1].to('h'), Teff, 'b-')

    ax1.plot(neptune[0][0].to('h'), Teff, 'c-', label=r'$5 R_{\oplus}$')
    ax1.plot(neptune[0][1].to('h'), Teff, 'c-')

    ax1.plot(jupiter[0][0].to('h'), Teff, '-', c='orange', label=r'$10 R_{\oplus}$')
    ax1.plot(jupiter[0][1].to('h'), Teff, '-', c='orange')

    ax1.set_xlabel('Eclipse times (h)')
    ax1.set_ylabel(r'$T_{\text{eff}}$ (K)')

    ax1.set_xlim(0,30)
    ax1.legend()
    plt.tight_layout()
    plt.show()




def plotDetectedPlanets():


    # Load NASA exoplanet file
    # https://exoplanetarchive.ipac.caltech.edu/

    filename = os.getcwd() + '/NASA_archive/nasa_exoplanets.csv'
    planetID = np.loadtxt(filename, delimiter=',', usecols=[0], dtype=str)
    data     = np.genfromtxt(filename, delimiter=',')

    # Sort out planets that do not contain {P, a, R, i}={2, 6, 10, 22}:

    cols = [2, 6, 10, 22]
    dex0 = np.ones_like(data[:,0], dtype=bool)

    for row in range(len(data)):
        for col in cols:
            dex0[row] *= ~np.isnan(data[row,col])

    # Check if either {a, Mp*sini}={6, 14}

    cols = [6, 14]
    dex1 = np.zeros_like(data[:,0], dtype=bool)
    for row in range(len(data)):
        for col in cols:
            dex1[row] += ~np.isnan(data[row,col])

    dex = dex0 * dex1

    # Ommit targets tabulated several times

    d = {'ID': planetID[dex],  'P': data[:,2][dex], 'a': data[:,6][dex],
         'R': data[:,10][dex], 'M': data[:,14][dex],
         'e': data[:,18][dex], 'i': data[:,22][dex]}

    # Create a data frame for easy handling

    df = pd.DataFrame(d, columns=['ID', 'R', 'M', 'P', 'a', 'i', 'e'])

    # Fetch columns of interest

    df = df.drop_duplicates(subset=['ID'])

    # Convert upper mass limit to actual mass

    df['M'] /= np.sin(np.deg2rad(df['i']))

    # Save data frame into ascii

    #df.to_csv('cat.txt')

    # PLOT DISTRIBUTION OF PLANETS

    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    sc = ax[0].scatter(df['M'], df['R'], c=df['e'], edgecolor='w', cmap='coolwarm')
    ax[0].set_xscale('log')
    ax[0].set_title('Radius vs. Mass')
    ax[0].set_xlabel(r'Mass [$M_{\oplus}$]')
    ax[0].set_ylabel(r'Radius [$R_{\oplus}$]')

    ax[1].scatter(df['P'], df['R'], c=df['e'], edgecolor='w', cmap='coolwarm')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('Radius vs. Period')
    ax[1].set_xlabel('Period [days]')
    ax[1].set_ylabel('Radius [$R_{\oplus}$]')

    ax[2].scatter(df['a'], df['M'], c=df['e'], edgecolor='w', cmap='coolwarm')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_title('Mass vs. Semimajor axis')
    ax[2].set_xlabel('Semimajor axis [AU]')
    ax[2].set_ylabel(r'Mass [$M_{\oplus}$]')

    ax[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[2].xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax[2].yaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.tight_layout(h_pad=1.0)
    cbar_ax = fig.add_axes([0.07, 0.58, 0.015, 0.3])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Eccentricity')
    plt.show()



    


def plotHistogramSED(df, title=False, figsize=(8,15)):

    # Compute ranges
    P_min, P_max = df.Pmag.min(), df.Pmag.max()
    M_min, M_max = df.M.min(), df.M.max()
    R_min, R_max = df.R.min(), df.R.max()
    Teff_min, Teff_max = df.Teff.min(), df.Teff.max()
    logg_min, logg_max = df.logg.min(), df.logg.max()
    Z_min, Z_max = df.Z.min(), df.Z.max()
    # Seperate stars
    df06 = df[df.ncams ==  6]
    df12 = df[df.ncams == 12]
    df18 = df[df.ncams == 18]
    df24 = df[df.ncams == 24]

    lw = 1.1
    c = ['royalblue', 'green', 'orange', 'orangered']

    # Inspect magnitude counts
    fig, ax = plt.subplots(6, 1, figsize=figsize)

    N = 50
    ax[0].hist(df06.Pmag, bins=N, range=(P_min, P_max), histtype='step', ec=c[0], lw=lw)
    ax[0].hist(df12.Pmag, bins=N, range=(P_min, P_max), histtype='step', ec=c[1], lw=lw)
    ax[0].hist(df18.Pmag, bins=N, range=(P_min, P_max), histtype='step', ec=c[2], lw=lw)
    ax[0].hist(df24.Pmag, bins=N, range=(P_min, P_max), histtype='step', ec=c[3], lw=lw)
    ax[0].set_xlabel(r'PLATO magnitude, $\mathcal{P}$')
    if title: ax[0].set_title(title, fontsize=20, pad=10)
    
    ax[1].hist(df06.M, bins=N, range=(M_min, M_max), histtype='step', ec=c[0], lw=lw)
    ax[1].hist(df12.M, bins=N, range=(M_min, M_max), histtype='step', ec=c[1], lw=lw)
    ax[1].hist(df18.M, bins=N, range=(M_min, M_max), histtype='step', ec=c[2], lw=lw)
    ax[1].hist(df24.M, bins=N, range=(M_min, M_max), histtype='step', ec=c[3], lw=lw)
    ax[1].set_xlabel(r'Mass, $M$ [$M_{\odot}$]')

    ax[2].hist(df06.R, bins=N, range=(R_min, R_max), histtype='step', ec=c[0], lw=lw)
    ax[2].hist(df12.R, bins=N, range=(R_min, R_max), histtype='step', ec=c[1], lw=lw)
    ax[2].hist(df18.R, bins=N, range=(R_min, R_max), histtype='step', ec=c[2], lw=lw)
    ax[2].hist(df24.R, bins=N, range=(R_min, R_max), histtype='step', ec=c[3], lw=lw)
    ax[2].set_xlabel(r'Radius, $R$ [$R_{\odot}$]')

    ax[3].hist(df06.Teff, bins=N, range=(Teff_min, Teff_max), histtype='step', ec=c[0], lw=lw)
    ax[3].hist(df12.Teff, bins=N, range=(Teff_min, Teff_max), histtype='step', ec=c[1], lw=lw)
    ax[3].hist(df18.Teff, bins=N, range=(Teff_min, Teff_max), histtype='step', ec=c[2], lw=lw)
    ax[3].hist(df24.Teff, bins=N, range=(Teff_min, Teff_max), histtype='step', ec=c[3], lw=lw)
    ax[3].set_xlabel(r'Effective temperature, $T_{\rm eff}$ [K]')

    ax[4].hist(df06.logg, bins=N, range=(logg_min, logg_max), histtype='step', ec=c[0], lw=lw)
    ax[4].hist(df12.logg, bins=N, range=(logg_min, logg_max), histtype='step', ec=c[1], lw=lw)
    ax[4].hist(df18.logg, bins=N, range=(logg_min, logg_max), histtype='step', ec=c[2], lw=lw)
    ax[4].hist(df24.logg, bins=N, range=(logg_min, logg_max), histtype='step', ec=c[3], lw=lw)
    ax[4].set_xlabel(r'Surface gravity, log $g$')

    ax[5].hist(df06.Z, bins=N, range=(Z_min, Z_max), histtype='step', label='6',  ec=c[0], lw=lw)
    ax[5].hist(df12.Z, bins=N, range=(Z_min, Z_max), histtype='step', label='12', ec=c[1], lw=lw)
    ax[5].hist(df18.Z, bins=N, range=(Z_min, Z_max), histtype='step', label='18', ec=c[2], lw=lw)
    ax[5].hist(df24.Z, bins=N, range=(Z_min, Z_max), histtype='step', label='24', ec=c[3], lw=lw)
    ax[5].set_xlabel(r'Metallicity, [Fe/H]')
    ax[5].legend(loc='upper left')

    for i in range(6):
        ax[i].set_ylabel('Count')
        ax[i].get_yaxis().set_label_coords(-0.07, 0.5)
    plt.tight_layout(pad=0.5)

    return fig, ax

    

#--------------------------------------------------------------#
#                          ANIMATIONS                          #
#--------------------------------------------------------------#

    
def plotSubfieldAnimation(filename, outputFileName=False, cadence=25,
                          frameRate=50, dpi=100, skipNimages=None, numImages=False,
                          colorMap="cubehelix", clipPercentile=8.0, 
                          showStarPositions=False, showPointLikeGhostPositions=False,
                          minVmag=None, maxVmag=None,
                          showStarIDs=False, showMaskOfStarID=None, tarMarkerSize=200,
                          useTitle=True, showGrid=True, fontSize=15, figsize=(6,6)):

    """Create and plot an animation of a set of subfields.

    Parameters
    ----------
    filename : str
        Filename of the HDF5 file.
    outputFileName : str
        Pass an output filename to save the animation.
    skipNimages : int
        Limit the animation and only show every 'skipNimages' for faster computation.
    numImages : int
        Pass the number of images to be used, else False means all images in file will be used.
    colorMap : str
        Matplotlib colormap (see docs for possible cmaps)
    clipPercentile : float
        Clip percentage used by the function simulation.showImage() to set contrast.
    showStarPositions : bool, str
        Show star positions with True. Use 'PIC' if you have a small subfield with contaminats.
    showPointLikeGhostPositions : bool
        Show the positions of the point-like ghosts.
    minVmag : float
        Lower V magnitude cut to show stellar coordinates. 
    maxVmag : float
        Upper V magnitude cut to show stellar coordinates. 
    showStarIDs : bool
        Show the star ID if requested.
    showMaskOfStarID : int
        If saved to file, show the photometric mask of the requested star.
    useTitle : str
        Use default elapsed time string or pass a title string.
    showGrid : bool
        If requested show a pixel grid.
    figsize : list
        Matplotlib 'figsize' object to select image size.

    Return
    ------
    Animation and/or output GIF file.
    """

    # Fetch file with simulated subfields
    print('Creating GIF animation:')
    from platosim.simfile import SimFile
    f = h5py.File(filename, "r")
    simfile = SimFile(filename)

    # Fetch the image names
    imgNames   = list(f['Images'].keys())
    imgNumbers = list(range(int(imgNames[0][5:]), int(imgNames[-1][5:])))
    N          = len(imgNames)

    # Skip N images to make animation faster
    if skipNimages is not None:
        imgNames   = imgNames[0::skipNimages]
        imgNumbers = imgNumbers[0::skipNimages]

    # Fetch StarPosition keys (i.e. "Exposure000000", etc)
    # TODO remove time column: -1 because Time is last column
    exposureGroupNames = list(f['StarPositions'].keys())[:-1]

    # Plot the image. Note that pixel coordinates start at the left bottom side of each pixel.

    ims = []
    fig, ax = plt.subplots(1,1,figsize=figsize)

    for imgNumber, imgName in zip(tqdm(imgNumbers, bar_format=ut.tqdmBar()), imgNames):

        # Fetch each pixel image
        
        image = f["Images/{0}".format(imgName)]

        # Fetch image dimention first time only
        
        if not ims:
            Nrows, Ncols = image.shape

        # Make image plot
        
        imagePlot = ax.imshow(image, cmap=colorMap, interpolation="nearest",
                              origin='lower', extent=[0,Nrows,0,Ncols], animated=True)

        # The large dynamic range of the pixel values often results in images where only
        # the brightest stars are visible. To improve the contrast, clip the color mapping.
        
        imagePlot.set_clim(np.percentile(image, clipPercentile),
                           np.percentile(image, 100-clipPercentile))

        # Add either default title or defined by user
        
        if useTitle:
            time  = cadence/86400 * imgNumber
            title = ax.text(0.5, 1.05, f"Elapsed time: {time:.2f} days",
                            horizontalalignment='center', transform=ax.transAxes)
        elif isinstance(useTitle, str) and int(imgNumber) == 0:
            ax.set_title(useTitle, fontsize=fontSize)

        # If requiered, overplot a gray semi-transparent grid
        # Note: this is only meaningsful for smaller imagettes

        if showGrid is True:
            ax.grid(c='gray', ls='-', alpha=0.3)

        # Overplot rectangles over those pixels that are part of the mask
        # NOTE: imshow reverses rows and columns

        if showMaskOfStarID is not None:
            rowIndices, colIndices, _, _, _, _ = simfile.getApertureMask(showMaskOfStarID,
                                                                         imgNumber)
            for k in range(len(rowIndices)):
                rect = patches.Rectangle((colIndices[k], rowIndices[k]), 1, 1, linewidth=2.0,
                                         edgecolor='royalblue', facecolor='none', hatch="/",
                                         zorder=2)
                ax.add_patch(rect)
                
        # If required, overplot the true averaged star positions

        if showStarPositions:

            ID, row, col, Xmm, Ymm, flux = simfile.getStarCoordinates(imgNumber-imgNumbers[0])

            # Set linewidth of marker

            lw = 0.1 * fontSize
            
            # Allow differentiating between a target and its contaminants
            
            if showStarPositions == 'PIC':

                mag = -2.5*np.log10(flux) + 25
                coor_tar = ax.scatter(col[0], row[0], s=tarMarkerSize, marker='o', c='lime',
                            edgecolor='k', linewidth=lw, zorder=4)

                # Scale contaminant circle with area
                
                if len(col) > 1:
                    # Scale contaminant circle with area
                    conDeltaMag   = mag[1:] - mag[0]
                    conMarkerSize = tarMarkerSize * (mag[0]/mag[1:])**2
                    coor_con = ax.scatter(col[1:], row[1:], s=conMarkerSize, marker='o', c='orange',
                                          edgecolor='k', linewidth=lw, zorder=4)

                # Add magnitude label above star position

                ax.annotate(f'{mag[0]:.1f}', xy=(col[0]-0.25, row[0]+0.30), color='green')
                for m,i,j in zip(mag[1:], col[1:], row[1:]):
                    ax.annotate(f'{m:.1f}', xy=(i-0.25, j+0.30), color='darkorange')
                    
            # Or hightligth all stars the same
            
            else:
                ax.scatter(col, row, s=int(tarMarkerSize/3), marker='o',
                           facecolors='royalblue', edgecolors='k',
                           linewidth=lw, zorder=4)

                
            # If requested, add star IDs to plot
            
            # if showStarIDs:
            #     for k in range(len(ID)):
            #         label = "{0}".format(ID[k])
            #         ax.annotate(label, (col[k], row[k]), fontsize='small',
            #                     fontweight='extra bold', color="black")
                    
        # Ensure that the axis limits are properly set
        
        ax.set_xlim(0, Ncols)
        ax.set_ylim(0, Nrows)

        # By default, matplotlib only shows the (x,y) coordinates of each pixel but not
        # the pixel value itself. Change this by redefining the axis.format_coord
        
        def format_coord(x, y):
            col = int(x)
            row = int(y)
            if col >= 0 and col < Ncols and row >= 0 and row < Nrows:
                z = image[row,col]
                return "x={:.1f}, y={:.1f}, z={:.1f}".format(x, y, z)
            else:
                return "x={:.1f}, y={:.1f}".format(x, y)
        ax.format_coord = format_coord

        # Show all ticks for smaller subfields or otherwise 10
        
        if Ncols < 10 and Nrows < 10:
            plt.xticks(np.arange(0, Nrows+1))
            plt.yticks(np.arange(0, Ncols+1))
        else:
            plt.xticks(np.arange(0, Nrows, 10))
            plt.yticks(np.arange(0, Ncols, 10))
            plt.tight_layout()
            
        # Add x and y axis labels
        
        plt.xlabel(r'Pixel column, $i$', fontsize=15)
        plt.ylabel(r'Pixel row, $j$',    fontsize=15)
            
        # Append images to list

        if showStarPositions == 'PIC':
            if len(col) > 1:
                ims.append([imagePlot, title, coor_tar, coor_con])
            else:
                ims.append([imagePlot, title, coor_tar])
        else:
            ims.append([imagePlot, title])

    # CREATE ANIMATION
    
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=0)
    
    # Save animation (fps=50 and dpi=100 seems like good settings)
    
    if outputFileName is not False:
        print('Saving animation, be patient..')
        ani.save(f'{outputFileName}.gif', fps=frameRate, dpi=dpi)

    # Show animation

    os.system(f'xdg-open {outputFileName}.gif')

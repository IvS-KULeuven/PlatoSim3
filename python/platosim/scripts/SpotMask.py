"""
Apply OGSE spot mask for camera tests.

A set of sources (point + extended) is injected in the given simulation object,
following the description of the OGSE spot masks for camera tests (PLATO-DLR-PL-TN-0069).
"""

##########
# Imports
##########

from platosim.simulation import Simulation
from math import pow, log10, radians, sin, cos, pi
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MASK_COORDINATES_SMALL_SPOTS = \
    {
        "1": {"x": -1.845, "y": 0.000},
        "2": {"x": -1.350, "y": 1.800},
        "3": {"x": 1.845, "y": 1.305},
        "4": {"x": 1.350, "y": -0.495},
    }   # [mm]

MASK_DIAMETER_SMALL_SPOTS = 0.05    # [mm]

MASK_COORDINATES_LARGE_SPOTS = \
    {
        "1": {"x": 0.000, "y": -2.000},
    }   # [mm]

MASK_DIAMETER_LARGE_SPOTS = 1.5     # [mm]

PROJECTION_RATIO = 5.0


#####################
# Auxiliary functions
#####################

def magnitude2flux(sim, magnitude):
    """ Conversion from magnitude to flux.

    Args:
        - magnitude: Magnitude to convert to flux.

    Return:
        - Flux [photons] corresponding to the given magnitude.
    """

    flux_v0 = sim["ObservingParameters/Fluxm0"]
    throughput_bandwidth = sim["Camera/ThroughputBandwidth"]
    transmission_efficiency = sim["Telescope/TransmissionEfficiency/BOL"]
    light_collecting_area = sim["Telescope/LightCollectingArea"] * 1e-4

    flux_factor = flux_v0 * throughput_bandwidth * transmission_efficiency * light_collecting_area

    exposure_time = sim["ObservingParameters/CycleTime"] - sim.getReadoutTime()[0]

    return flux_factor * pow(10, -0.4 * magnitude) * exposure_time


def flux2magnitude(sim, flux):
    """ Conversion from flux to magnitude.

    Args:
        - flux: Flux [photons] to convert to magnitude.

    Return:
        - Magnitude corresponding to the given flux.
    """

    flux_v0 = sim["ObservingParameters/Fluxm0"]
    throughput_bandwidth = sim["Camera/ThroughputBandwidth"]
    transmission_efficiency = sim["Telescope/TransmissionEfficiency/BOL"]
    light_collecting_area = sim["Telescope/LightCollectingArea"] * 1e-4

    flux_factor = flux_v0 * throughput_bandwidth * transmission_efficiency * light_collecting_area

    exposure_time = sim["ObservingParameters/CycleTime"] - sim.getReadoutTime()[0]

    return -log10(flux / flux_factor / exposure_time) / 0.4


def mask_to_ccd_coordinates(sim, x_mask, y_mask, rotation_angle_mask):
    """ Conversion from mask coordinates to  CCD coordinates.

    Args:
        - sim: Simulation in which to insert the sources.
        - x_mask: x-coordinate [mm] in the mask reference frame (before projection).
        - y_mask: y_coordinate [mm] in the mask reference frame (before projection).
        - rotation_angle_mask: Rotation angle of the mask reference frame w.r.t.
                               the CCD reference frame [radians].

    Return:
        - x_ccd: x-coordinate [pixels] in the CCD reference frame.
        - y_ccd: y-coordinate [pixels] in the CCD reference frame.
    """

    # After projection [µm] (in the projected mask reference frame)

    x_mask, y_mask = x_mask / PROJECTION_RATIO * 1000, y_mask / PROJECTION_RATIO * 1000

    # After projection [pixels] (in the projected mask reference frame)

    pixel_size = sim["CCD/PixelSize"]  # [µm]
    x_mask, y_mask = x_mask / pixel_size, y_mask / pixel_size

    # In the CCD reference frame [pixels]
    # -> account for the following:
    #       - rotation of the mask w.r.t. the CCD reference frame;
    #       - the projected centre of the mask is at the centre of the sub-field;
    #       - sub-field zeropoint.

    x_ccd = x_mask * cos(rotation_angle_mask) - y_mask * sin(rotation_angle_mask)
    y_ccd = x_mask * sin(rotation_angle_mask) + y_mask * cos(rotation_angle_mask)

    x_ccd += + sim["SubField/NumColumns"] / 2
    y_ccd += sim["SubField/NumRows"] / 2

    x_ccd += sim["SubField/ZeroPointColumn"]
    y_ccd += sim["SubField/ZeroPointRow"]

    return x_ccd, y_ccd


def insert_ogse_spot_mask(sim, magnitude_point_source, rotation_angle_mask=30, include_large_spot=True):
    """ Apply OGSE spot mask for camera tests.

    A set of sources (point + extended) is injected in the given simulation object,
    following the description of the OGSE spot masks for camera tests
    (PLATO-DLR-PL-TN-0069).

    All small spots are point sources of the given magnitude.  The flux of
    extended source should be scaled according to the number of illuminated pixels
    (in absence of a PSF, all illuminated pixels should receive the same flux).

    As the mask can be rotated in the OGSE, the orientation angle of the mask
    reference frame w.r.t. the CCD reference frame is given.

    The following assumptions are made:
        - use of a pre-computed PSF (from file);
        - readout mode configured (nominal vs. partial readout);
        - position and dimensions of sub-field configured.

    Args:
        - sim: Simulation in which to insert the sources.
        - magnitude_point_source: Magnitude of a point source (the total flux of
                                  extended sources should be scaled according to
                                  the number of illuminated pixels).
        - rotation_angle_mask: Rotation angle of the mask reference frame w.r.t.
                               the CCD reference frame [degrees].
    """

    rotation_angle_mask_radians = radians(rotation_angle_mask)

    flux_point_source = magnitude2flux(sim, magnitude_point_source)

    #############################
    # Small spots (point sources)
    #############################

    radius_small_source = MASK_DIAMETER_SMALL_SPOTS / PROJECTION_RATIO * 1000 / sim["CCD/PixelSize"] / 2  # [pixels]

    small_spot_rows = np.array([])
    small_spot_columns = np.array([])

    for spot_id, spot_coordinates in MASK_COORDINATES_SMALL_SPOTS.items():

        x_mask, y_mask = spot_coordinates["x"], spot_coordinates["y"]
        x_ccd, y_ccd = mask_to_ccd_coordinates(sim, x_mask, y_mask, rotation_angle_mask_radians)

        small_spot_rows = np.append(small_spot_rows, y_ccd)
        small_spot_columns = np.append(small_spot_columns, x_ccd)

    flux_small_spot = flux_point_source * pi * pow(radius_small_source, 2)
    
    small_spot_magnitudes = np.empty(len(MASK_COORDINATES_SMALL_SPOTS))
    small_spot_magnitudes.fill(flux2magnitude(sim, flux_small_spot))

    ###########################################
    # Large sources (circular extended sources)
    ###########################################

    radius_large_source = MASK_DIAMETER_LARGE_SPOTS / PROJECTION_RATIO * 1000 / sim["CCD/PixelSize"] / 2  # [pixels]
    num_subpixels = sim["SubField/SubPixels"]  # Sub-pixel resolution

    large_spot_rows = np.array([])
    large_spot_columns = np.array([])
    large_spot_magnitudes = np.array([])

    if include_large_spot:

        for spot_id, spot_coordinates in MASK_COORDINATES_LARGE_SPOTS.items():

            x_mask, y_mask = spot_coordinates["x"], spot_coordinates["y"]
            x_ccd, y_ccd = mask_to_ccd_coordinates(sim, x_mask, y_mask, rotation_angle_mask_radians)

            # The extended source is approximated by placing proportionally fainter stars
            # in each of the sub-pixels, seen by this source.

            for row in np.arange(y_ccd - radius_large_source, y_ccd + radius_large_source + 1, 1 / num_subpixels):

                for column in np.arange(x_ccd - radius_large_source, x_ccd + radius_large_source + 1, 1 / num_subpixels):

                    if pow(row - y_ccd, 2) + pow(column - x_ccd, 2) <= pow(radius_large_source, 2):

                        large_spot_rows = np.append(large_spot_rows, row)
                        large_spot_columns = np.append(large_spot_columns, column)

        # Calculate the magnitude to insert in the sub-pixels to simulate the extended source:
        #   - magnitude of point source -> total flux in single pixel (in absence of PSF);
        #   - distribute flux evenly over the sub-pixels;
        #   - flux per sub-pixel -> magnitude

        flux_sub_pixel = flux_point_source / pow(num_subpixels, 2)

        magnitude_sub_pixel = flux2magnitude(sim, flux_sub_pixel)

        num_subpixels_extended_source = len(large_spot_rows)
        large_spot_magnitudes = np.empty(num_subpixels_extended_source)
        large_spot_magnitudes.fill(magnitude_sub_pixel)

    ########################################################
    # Concatenate the small and large spots in one catalogue
    ########################################################

    rows = np.concatenate([small_spot_rows, large_spot_rows])
    columns = np.concatenate([small_spot_columns, large_spot_columns])
    magnitudes = np.concatenate([small_spot_magnitudes, large_spot_magnitudes])
    ids = np.arange(len(magnitudes)) + 1

    sim.createStarCatalogFileFromPixelCoordinates(rows, columns, magnitudes, ids,
                                                  os.environ["PLATO_WORKDIR"] + "ogse-spot-mask.txt")

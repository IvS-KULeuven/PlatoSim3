#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <array>
#include <tuple>

#include "armadillo"

#include "ArrayOperations.h"
#include "ConfigurationParameters.h"
#include "Constants.h"
#include "Detector.h"
#include "HDF5File.h"
#include "HDF5Writer.h"
#include "Heartbeat.h"
#include "Logger.h"
#include "Exceptions.h"
#include "Polynomial1D.h"
#include "Sky.h"
#include "StringUtilities.h"
#include "Platform.h"
#include "Telescope.h"
#include "Units.h"
#include "Parameter.h"


using namespace std;


class Detector;  // forward declaration

typedef map<unsigned int, array<double, 6>>::iterator starInfoIterator;


class Camera : public HDF5Writer
{
    public:

        Camera(ConfigurationParameters &configParam, HDF5File &hdf5File, Platform &platform, Telescope &telescope, Sky &sky);
        virtual ~Camera();

        virtual void configure(ConfigurationParameters &configParam);
        virtual void exposeDetectorWithStars(Detector &detector, double startTime, double exposureTime);
        virtual void exposeDetectorWithSkyBackground(Detector &detector, double startTime, double exposureTime, double readoutTimeBeforeNextExposure);
        virtual void updateParameters(double time);
        double getTransmissionEfficiency(double startTime);
        virtual void initHDF5Groups() override;
        virtual void flushOutput() override;

        pair<double, double> skyToFocalPlaneCoordinates(double raStar, double decStar, bool useInitialOrientation=false);
        pair<double, double> focalPlaneToSkyCoordinates(double xFP, double yFP, bool useInitialOrientation=false);
  arma::colvec telescopeToSkyCoordinates(arma::vec vecTL, bool useInitialOrientation=false);

        pair<double, double> undistortedToDistortedFocalPlaneCoordinates(double xFPmm, double yFPmm);
        pair<double, double> distortedToUndistortedFocalPlaneCoordinates(double xFPdist, double yFPdist);

        double getGnomonicRadialDistanceFromOpticalAxis(double xFP, double yFP);
        void addSkybackgroundAndTransmissionEfficiency(double skyBackground, double transmissionEfficiency);
        set<unsigned int> getAllStarIDs();

        tuple<double, double, double, double, double, double> getInfoForTheMostRecentExposureForStar(int starID);
        pair<starInfoIterator, starInfoIterator> getInfoForTheMostRecentExposureForAllStars();

        double getTotalSkyBackground();
        double getFocalLength();
    double getBackgroundFlux(double xFP, double yFP, Detector &detector, double startTime, double exposureTime, double readoutTimeBeforeNextExposure);

    protected:

        virtual tuple<unsigned long, unsigned long> makeStarCatalogSelection(Detector &detector, double startTime, double exposureTime);

        int beginExposureNr;                 // Sequential number of first exposure. useful for slurm parallellisation
        int numExposures;                    // Number of exposures

        Platform &platform;
        Telescope &telescope;
        Sky &sky;

        Parameter<double> *focalLength;       // [mm]
        Parameter<double> *focalPlaneAngle;   // Orientation of the focal plane, as an angle around the optical axis  [rad]
        Parameter<double, 7> *distortionCoef; // distortion coefficients to map undistorted to distorted coordinates.
        Parameter<double, 7> *inverseDistortionCoef; // inverse distortion coefficient to map distorted to undistorted coordinates.

        string distortionModel;               // The model used to compute the distortion
        double plateScale;                    // [arcsec/micron]
        double throughputBandwidth;           // FWHM of the throughput passband [nm]
        double throughputLambdaC;             // Central wavelength of the throughput passband [nm]

        double internalTime;

        bool includeAberrationCorrection; // Whether or not (differential) aberration correction should be included
        string aberrationCorrectionType;  // [differential or absolute]

        bool isMapped;                    // Whether or not the PSF is mapped from a file or not
        bool includeFieldDistortion;      // Whether or not field distortion should be included

        bool useConstantSkyBackground;
        double userGivenSkyBackground;    // User-set zodiacal + stellar sky background.                          [phot/pix/s]
                                          // If negative, computed by the Sky class
        double fluxOfV0Star;              // Photon flux of a V=0 (G2V) star                                      [phot/s/m^2/nm]

        double raSun;                     // Right ascension of the direction of the sun shield during the run    [rad]
        double decSun;                    // Declination of the direction of the sun shield during the run        [rad]

        bool writeStarPositions;          // Whether or not the star positions should be written to the output HDF5 file
        bool groupByExposure;             // Whether or not we should group the starpositions by exposure or by StarID
        bool writeGhostPositions;         // Whether or not the ghost positions should be written to the output HDF5 file
        bool writeTransmissionEfficiency;

        // detectedStarInfo[startTime][starID] contains the values (xFPmean, yFPmean, rowPixMean, colPixmean, sumFlux, Ndetections)

        map<double, map<unsigned int, array<double, 6>>> detectedStarInfo;
        map<double, map<unsigned int, array<double, 7>>> detectedExtendedGhostInfo;
        map<double, map<unsigned int, array<double, 6>>> detectedPointLikeGhostInfo;
        vector<double> skyBackgroundValues;
        vector<double> transmissionEfficiencyValues;
        vector<unsigned int> starIDsInSubfield;
        double totalSkyBackground;          // Total sky background [photons / pixel / exposure]

        bool includePointLikeGhosts;                                // Whether or not to include pointlike ghosts
        bool includeExtendedGhosts;                                 // Whether or not to include extended ghosts
        double distanceCutOffPointLikeGhosts;                       // Beyond this distance from the optical axis [degrees], sources don't produce point-like ghosts anymore
        double fluxRatioOnAxisPointLikeGhosts;                      // Flux ratio between the point-like ghost and the originating source on-axis [%] -> linear decrease
        double distanceRatioExtendedGhosts;                         // For a star at FP-coordinates (x, y), the centre of the extended ghost will be at (distanceRatio * x, distanceRatio * y)
        double fluxRatioExtendedGhosts;                             // Flux ratio between the extended ghost and the originating source [%]
        Parameter<double, 3> *extendedGhostRadiusCoefficients;      // Coefficients of the 2nd-degree polynomial (in distance from the optical axis), describing the radius of the (circular) extended source

    private:

        
};



#endif

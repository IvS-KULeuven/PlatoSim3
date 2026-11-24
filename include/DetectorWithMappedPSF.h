#ifndef DETECTORWITHMAPPEDPSF_H
#define DETECTORWITHMAPPEDPSF_H

#include <string>
#include <cmath>
#include <random>
#include <functional>

#include "armadillo"

#include "Detector.h"
#include "PointSpreadFunction.h"
#include "ArrayOperations.h"

using namespace std;



class DetectorWithMappedPSF : public Detector
{
    public:

        DetectorWithMappedPSF(ConfigurationParameters &configParam, HDF5File &hdf5file, Camera &camera, TemperatureGenerator &feeTemperatureGenerator, TemperatureGenerator &detectorTemperatureGenerator, double readoutTimeBeforeNextExposure, double readoutTimeDuringNextExposure);
        ~DetectorWithMappedPSF();

        double takeExposure(int exposureNr, double startTime, double exposureTime) override;

        tuple<bool, double, double> addFlux(double xFP, double yFP, double flux) override;
        void addFlux(double flux) override;
        tuple<bool, double, double> addExtendedGhost(double xFP, double yFP, double radius, double flux) override;

        void configure(ConfigurationParameters &configParam);
        void writeDiffusedPSFToHDF5(PointSpreadFunction *psf);
        void applyDiffusionKernelOnPSF(double subpixRow, double subpixColumn, double flux, arma::fmat& psf, int numberOfPsfSubpixelsPerPixel);
        void applyDistortion(double &x, double &y) override;
        void applyInverseDistortion(double &x, double &y) override;
        double getMaxMappedFieldDistortion();
        void generateThroughputMap() override;

    protected:

        bool areColinear(std::array<std::array<double, 2>, 3>);
        void reset() override;
        void initHDF5Groups() override;
        void integrateLight(int exposureNr, double startTime, double exposureTime) override;
        bool isInSubPixelMap(double row, double column);
        void applyFlatfieldAtSubpixel();
        void applyDiffusionKernel(arma::Mat<float> &map, double row, double column, double flux);
        void addStarToColumnMap(double xFP, double yFP, double row, double column, double flux) override;
        void addStarToColumnProfile(double xFP, double yFP, double row, double column, double flux) override;
        void initialiseColumnStarStamp();
        void generateFlatfieldMap();
        void readInFlatfieldMap();
        void generateDiffusionKernel(double kernelWidth);
        void rebin();
        void setPsfForSubfield();
        void convolveWithPsf();
        void addBackgroundMapToSubpixelMap(Camera &camera, double startTime);
        void fillBackgroundSubpixelMap(Camera &camera, double startTime, double exposureTime);
        void fillBackgroundMap(Camera &camera, double startTime, double exposureTime) ;

        arma::Mat<float> subPixelBackgroundMap;
        arma::Mat<float> subPixelMap;           // Sub-pixel map, incl. edge pixels
        arma::Mat<float> psfMap;                // The PSF map that will be used for convolving
        arma::Mat<float> columnStarStamp;       // Sub-pixel stamp in which a star along the columns is convolved with the PSF
        Convolver columnStarConvolver;          // Convolver for columnStarStamp
        unsigned int columnStarStampSize = 0;   // Size of columnStarStamp, 0 if columnStarConvolver still needs to be initialised [pixels]
        arma::Row<float> columnStarPsfProfile;  // Sum over the rows of psfMap, to convolve the column profile of a star with
        arma::Mat<float> columnStarKernelStrip; // Sub-pixel strip, just high enough for the diffusion kernel of one star
        arma::Mat<float> flatfieldSubpixelMap;          // Intra-pixel flatfield map
        arma::Mat<float> unDistortedX;
        arma::Mat<float> unDistortedY;
        vector<std::array<double, 4>> distortionMap;
        string flatfieldFilePath;
        string flatfieldSource;

        double chargeDiffusionStrength;		// Strength of the charge diffusion (width of the Gaussian diffusion kernel) [pixels]
        bool includeChargeDiffusion;		// Whether or not to include charge diffusion
        bool includeJitterSmoothing;            // Whether or not to include jitter smoothing
        double flatfieldNoiseRMS;               // Peak-to-peak noise amplitude
        bool includeFlatfield;                  // Whether or not to include flat fielding
        bool writeFlatfieldMap;                 // Whether or not to write the flatfield map to the HDF5 file
        bool writeSubPixelImagesToHDF5;         // Write subpixel maps to HDF5 as well
        bool includeConvolution;                // Whether or not to convolve the subPixelMap with the PSF
        bool writeDiffusedPSF;                  // Whether or not to write the diffused PSF to the output HDF5 file

        arma::Mat<float> diffusionKernel;                   // Diffusion kernel image
        IntegralOfAnalyticSignalResponse signalResponse;    // Signal response
        double diffusionKernelWidth;                        // Width (sigma) of the Gaussian diffusion kernel [sub-pixels]
        int diffusionKernelImageSize;                       // Size of the diffusion kernel image [sub-pixels]
  
        unsigned int numRowsSubPixelMap;        // Nr of subpixel rows in the subfield incl. edge pixels (= size in the y-direction) [subpixels]
        unsigned int numColumnsSubPixelMap;     // Nr of subpixel columns in the subfield incl. edge pixels (= size in the x-direction = readout direction) [subpixels]
        unsigned int numSubPixelsPerPixel;      // Nr of sub-pixels per pixel
        PointSpreadFunction *psf;
        long flatfieldSeed;

        Convolver convolver;
};

#endif

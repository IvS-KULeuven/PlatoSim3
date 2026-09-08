#ifndef DETECTORWITHANALYTICNONGAUSSIANPSF_H
#define DETECTORWITHANALYTICNONGAUSSIANPSF_H

#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <complex>
#include <string>

#include "armadillo"

#include "Constants.h"
#include "Units.h"
#include "Detector.h"
#include "Camera.h"
#include "Parameter.h"



using namespace std;



class DetectorWithAnalyticNonGaussianPSF: public Detector 
{
    public:

        DetectorWithAnalyticNonGaussianPSF(ConfigurationParameters &configParam, HDF5File &hdf5File, Camera &camera, TemperatureGenerator &feeTemperatureGenerator, TemperatureGenerator &detectorTemperatureGenerator, double readoutTimeBeforeNextExposure, double readoutTimeDuringNextExposure);
        ~DetectorWithAnalyticNonGaussianPSF();

        double takeExposure(int exposureNr, double startTime, double exposureTime) override;

        void configure(ConfigurationParameters &configParam);
        void updateParameters(double time) override;
        bool addFluxToMap(arma::Mat<float>& map, double row0, double col0, double r, double p, double flux);

        tuple<bool, double, double> addFlux(double xFP, double yFP, double flux) override;
        void addFlux(double flux) override;
        tuple<bool, double, double> addExtendedGhost(double xFP, double yFP, double radius, double flux) override;

        void integrateAnalyticPSF(IntegralOfAnalyticSignalResponse&, double, double, double, double, double, int = 1);
        void makeHighResolutionPSF(arma::Mat<float> &highResMap, bool includeDiffusion, int Npixels, int Nsubpixels);
        void flushOutput() override;

        // Photometry
  
        void applyPhotometry(const unsigned int exposureNr);

    protected:

        void integrateLight(int exposureNr, double startTime, double exposureTime) override;
        void applyFlatfield() override;
        void generateFlatfieldMap();
        void readInFlatfieldMap();

        Parameter<double> *sigma;           // Width of the analytic PSF, equal to sigma for a Gaussian PSF
        vector<vector<double>> params;      // Table of analytic PSF parameters
        arma::Mat<float> flatfieldMap;      // Pixel flatfield map
        unsigned int numExposures;          // Number of exposures
        unsigned int beginExposureNr;       // Exposure nr of the first exposure in the time series
        double cycleTime;                   // Image cycle time (exposure + readout before next exposure starts)  [s]
        double chargeDiffusionStrength;	    // Strength of the charge diffusion (width of the Gaussian diffusion kernel) [pixels]
        bool includeChargeDiffusion;	    // Whether or not to include charge diffusion
        double flatfieldNoiseRMS;           // Peak-to-peak noise amplitude
        bool includeFlatfield;              // Whether or not to include flat fielding        
        long flatfieldSeed;                 // Seed dedicated to generate a random flatfield map
        bool writeFlatfieldMap;             // Whether or not to write the flatfield map to the HDF5 file
        string flatfieldFilePath;            // Path of the flatfield file
        string flatfieldSource;              // Determines how the flatfield gets determined 
        bool writeHighResolutionPSF;        // Wheter or not to write the high resosultion PSF to the HDF5 file
        bool writeDiffusedPSF;              // Wheter or not to write the high resosultion PSF to the HDF5 file 

        // Photometry
  
        bool includePhotometry;             // Whether or not to include on-the-fly photometry
        int contaminationRadius;            // Stars outside the radius are never considered a contaminant of the main target [pix] 
        double maskUpdateInterval;          // All photometric masks will be updated every xx days  [days]
        vector<unsigned int> photStarIDs;   // Star IDs for which you need photometry

        // The following maps contain for each star ID, a vector with the photometry information for each exposure.
        // E.g. maskSizeTarget[1234][10] contains the nr of pixels of the mask of target 1234 for exposure 10.

        map<unsigned int, vector<unsigned int>> exposureNrOfMaskUpdate;   // Photometric mask is updated once in a while
        map<unsigned int, vector<unsigned int>> maskSizeTarget;           // Nr of pixels within the mask for each target
        map<unsigned int, vector<double>> inputFluxTarget;                // Flux of the target as computed from the (variable) Vmag
        map<unsigned int, vector<double>> estimatedFluxTarget;            // Estimated flux for each target
        map<unsigned int, vector<double>> varFluxTarget;                  // Variance of the flux for each target
        map<unsigned int, vector<double>> NSRtarget;                      // Noise/Signal ratio of the flux of each target
        map<unsigned int, vector<double>> aggregatedSPR;                  // Stellar pollution ratio = aggregated contaminant flux / aggregated total flux 
    
        // The indices of the masks are stored as [starID][exposureNr][index]
  
        map<unsigned int, map<unsigned int, vector<unsigned int>>> rowIndexOfMaskOfTarget;  // The row indices of all mask pixels, for each target
        map<unsigned int, map<unsigned int, vector<unsigned int>>> colIndexOfMaskOfTarget;  // The column indices of all mask pixels, for each target

};


#endif

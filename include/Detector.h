#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <cmath>
#include <random>
#include <functional>
#include <valarray>

#include "armadillo"

#include "Faddeeva.hh"

#include "StrayLight.h"
#include "Constants.h"
#include "ArrayOperations.h"
#include "Mathematics.h"
#include "FileUtilities.h"
#include "Random.h"
#include "Camera.h"
#include "FrontEndElectronics.h"
#include "TemperatureGenerator.h"
#include "ConfigurationParameters.h"
#include "Convolver.h"
#include "HDF5File.h"
#include "HDF5Writer.h"
#include "Logger.h"
#include "Units.h"
#include "Parameter.h"


using namespace std;

class StrayLight;

class Camera;      // forward declaration



class IntegralOfAnalyticSignalResponse
{
    public:

        IntegralOfAnalyticSignalResponse() : size(0), n(0.) {};
        IntegralOfAnalyticSignalResponse(size_t s, double d = 0.) : size(s), n(0.), dsigma(d) {}
        virtual ~IntegralOfAnalyticSignalResponse(){};
        IntegralOfAnalyticSignalResponse& addPart(double, double, double, double, double = 0., double = 0., double = 0.);
        double operator()(unsigned, unsigned, bool = true);

    private:

        size_t size;                              // number of (sub)pixels in one dimension
        double n;                                 // normalization factor
        double dsigma;                            // Gaussian diffusion kernel width
        vector<valarray<double>> erfxr;           // evaluated error functions for x
        vector<valarray<double>> erfyr;           // evaluated error functions for y
        vector<valarray<complex<double>>> erfxc;  // evaluated complex error functions for x
        vector<valarray<complex<double>>> erfyc;  // evaluated complex error functions for y
};










class Detector: public HDF5Writer
{
    public:

        Detector(ConfigurationParameters &configParam, HDF5File &hdf5File, Camera &camera, TemperatureGenerator &feeTemperatureGenerator, TemperatureGenerator &detectorTemperatureGenerator, double readoutTimeBeforeNextExposure, double readoutTimeDuringNextExposure);
        virtual ~Detector();

        virtual double takeExposure(int exposureNr, double startTime, double exposureTime);
        virtual void updateParameters(double time);
        void configure(ConfigurationParameters &configParam);

        pair<double, double> pixelToFocalPlaneCoordinates(double row, double column);
        pair<double, double> focalPlaneToPixelCoordinates(double xFP, double yFP);

        pair<double, double> getFocalPlaneCoordinatesOfSubfieldCenter();
        tuple<double, double, double, double, double, double, double, double> getFocalPlaneCoordinatesOfSubfieldCorners();

        double getSolidAngleOfOnePixel(double plateScale);

        virtual tuple<bool, double, double> addFlux(double xFP, double yFP, double flux) = 0;
        virtual void addFlux(double flux) = 0;
        virtual tuple<bool, double, double> addExtendedGhost(double xFP, double yFP, double radius, double flux) = 0;

        bool isInPixelMap(double row, double column);
        bool isInSubfield(double xFPmm, double yFPmm);
        bool constantSkyBackground;

        double getReadoutTimeBeforeNextExposure();
        virtual void applyDistortion(double &, double &){};
        virtual void applyInverseDistortion(double &, double &){};
        void addBackgroundMapToPixelMap(Camera &camera, double startTime);
        void addStraylightToPixelMap(double startTime);

    protected:

        virtual void reset();
        virtual void integrateLight(int exposureNr, double startTime, double exposureTime) = 0;

        virtual void generateThroughputMap();
        virtual void checkGain();
        virtual void readBfeCoefficients(string filename);

        virtual void applyFlatfield() = 0;
        virtual void applyThroughputEfficiency();
        virtual void applyBFE();
        virtual void addDarkSignal(float exposureTime);
        virtual void configureMetallicShield(ConfigurationParameters &configParam);

        virtual void readOut(float exposureTime);
        virtual void addPhotonNoise();
        virtual void addCosmics(float exposureTime);
        virtual void addCosmics(float exposureTime, arma::Mat<float> &map, vector<unsigned int> &rowsCos, vector<unsigned int> &columnsCos, vector<double> &fluxCos, int numRows, int numColumns, string area);
        virtual void applyFullWellSaturation();
        virtual void applyCTI();
        virtual void applyChargeInjection();
        virtual void setInitialNumberOfOccupiedTraps(arma::Mat<float> &numberOfOccupiedTraps);
        virtual void applyOpenShutterSmearing(float exposureTime);
        virtual void addReadoutNoise();
        virtual void applyQuantisation();
        virtual void applyGain();
        virtual void addElectronicOffset();
        virtual void applyDigitalSaturation();
        virtual void applyOverAndUnderShoot();

        void generateBadPixelMap();
        void applyBadPixelMap();

        RaggedArray subFieldCosmics;
        RaggedArray smearingCosmics;
        RaggedArray leftBiasCosmics;
        RaggedArray rightBiasCosmics;

        StrayLight *straylight;
        void applySimpleCTImodel();
        void applyShort2013CTImodel(string map);
        void makeSubGroupForCosmics(string field, int exposureNr);

        void applyParticulateContamination();
        void applyMolecularContamination();

        void setSubfield(const arma::Mat<float> &subfield);
        arma::Mat<float> getSubfield();

        virtual void initHDF5Groups() override;
        // TODO implement writing of time here

        virtual void writePixelMapsToHDF5(int exposureNr);

        virtual void writeCosmicHitsToHDF5(int exposureNr);
        virtual void writeBackgroundMapToHDF5();
        virtual void writeCTIToHDF5();
        void writeBadPixelMapToHDF5();

        double getRowEdgeFOV(int column);

        virtual double getTemperature();

        vector<unsigned int> rowsOfCosmicsInSubField;
        vector<unsigned int> columnsOfCosmicsInSubField;
        vector<double> fluxOfCosmicsInSubField;

        vector<unsigned int> rowsOfCosmicsInSmearingMap;
        vector<unsigned int> columnsOfCosmicsInSmearingMap;
        vector<double> fluxOfCosmicsInSmearingMap;

        vector<double> straylightValues;
        vector<unsigned int> rowsOfCosmicsInBiasMapLeft;
        vector<unsigned int> columnsOfCosmicsInBiasMapLeft;
        vector<double> fluxOfCosmicsInBiasMapLeft;

        vector<unsigned int> rowsOfCosmicsInBiasMapRight;
        vector<unsigned int> columnsOfCosmicsInBiasMapRight;
        vector<double> fluxOfCosmicsInBiasMapRight;
        void fillBackgroundMap(Camera &camera, double startTime, double exposureTime);

        int coveredLeft, coveredRight;            // Amount of pixels in the subfield that are blocked off from light because of the metallic schield
        int coveredBottom, coveredTop;

        arma::Mat<float> pixelMap;               // Pixel map, excl. edge pixels
        arma::Mat<float> backgroundMap;          // Sub-pixel map, incl. edge pixels
        arma::Mat<float> smearingMap;            // Smearing map (i.e. over-scan strip)
        arma::Mat<float> biasMapLeft;            // Bias map (i.e. pre-scan strip) for the left detector half
        arma::Mat<float> biasMapRight;           // Bias map (i.e. pre-scan strip) for the right detector half
        arma::Mat<float> throughputMap;          // Throughput efficiency map, due to
                                                 // vignetting, particulate & molecular
                                                 // contamination, and quantum efficiency
        arma::Mat<int> badPixelMap;
    

        double missionDuration;                  // Duration of the PLATO Mission, used for degrading parameters      [s]

        arma::Mat<int> mechanicalVignettingMask; // Mask for the sub-field showing which pixels are within the FOV (1) and which aren't (0)
        arma::Row<int> numExposedRowsInFOV;      // How many pixels in the exposed part of the detector for each column are within the FOV (only for columns showing overlap with the sub-field)

        unsigned int numRows;                    // Nr of rows of the detector (= size in y-direction) including non-exposed ones [pixels]
        unsigned int numColumns;                 // Nr of columns of the detector (= size in x-direction = readout direction) [pixels]
        unsigned int numRowsPixelMap;            // Nr of rows in the subfield excl. edge pixels (= size the y-direction) [pixels]
        unsigned int numColumnsPixelMap;         // Nr of columns in the subfield excl. edge pixels (= size in the x-direction = readout direction) [pixels]
        unsigned int numRowsSmearingMap;         // Nr of rows in the smearing over-scan strip [pixels]
        unsigned int numRowsBiasMap;             // Nr of rows in the bias pre-scan strip [pixels]
        unsigned int numColumnsBiasMap;          // Nr of columns in the bias pre-scan strip [pixels]

        unsigned int firstRowExposed;            // Index of the first row that is exposed to light (different for the Fast and Normal camera's) [pixels]

        string ccdPosition;
        Parameter<double, 12> *ccdPositions;      // Pre-defined CCD positions (either fixed or time-dependent (from file))
        double customOriginOffsetY;              // Y-coordinate of the detector origin from the centre of the optical plane [mm]
        double customOriginOffsetX;              // X-coordinate of the detector origin from the centre of the optical plane [mm]
        unsigned int subFieldZeroPointRow;       // Position of the subfield zeropoint w.r.t. the complete detector in the row direction [pixels]
        unsigned int subFieldZeroPointColumn;    // Position of the subfield zeropoint w.r.t. the complete detector in the column direction [pixels]
        double customOrientationAngle;           // Orientation angle of the detector w.r.t. the orientation of the focal plane, measured counterclockwise [radians]
        double rotationAnglePsf;                 // Angle over which to rotate the PSF

        double pixelSize;                        // Pixel size [microns]
        unsigned int numEdgePixels;              // Nr of pixels to extend the subfield on each side, to account for the edge effect

        arma::Cube<float> bfeCoefficients;       // Coefficients a^X_ij for the BFE in Sect. 6.1 in Guyonnet et al. 2015
        int bfeNeighbors[4][2];                  // Neighbours X for the BFE in Sect. 5.2 in Guyonnet et al. 2015
        int bfeRange;                            // How far pixels can be apart and still influence each other [pixels] (use window with dimensions 2 * range + 1)

        bool includeCosmicsInSubField;           // Whether or not to include cosmic hits
                                                 // in the subfield
        bool includeStraylight;                  // Wheter or not to include straylight
                                                 // in the subfield
        bool includeBadPixelMap; 
        bool includeCosmicsInSmearingMap;        // Whether or not to include cosmic hits in the (physical) overscan region
        bool includeCosmicsInBiasMap;            // Whether or not to include cosmic hits in the (virtual) prescan region
        bool groupByExposure;                    
        double cosmicHitRate;				     // Cosmic hit rate [events / cm^2 / s]
        vector<double> cosmicTrailLengthParams;  // Distribution parameters of the length of the cosmic trails          [pixels]
        vector<double> cosmicIntensityParams;    // Skew-Normal distribution parameters fo the intensity of the intensities of the cosmics

        vector<unsigned int> cosmicEntryRowSubfield;     // rows in the subfield where the cosmic hit the CCD                 [pixel]
        vector<unsigned int> cosmicEntryColSubfield;     // columns in the subfield where the cosmic hit the CCD              [pix]
        vector<double> cosmicsTrailsSubfield;            // length of the trails of the cosmics that hit the subfield         [pix]
        vector<double> cosmicsAnglesSubfield;            // angle at which the cosmic hits the CCD in the subfield            [rad]
        vector<double> cosmicsIntensitiesSubfield;       // total number of electrons the cosmic will release over its trail  [e-]

        vector<unsigned int> cosmicEntryRowSmearingMap;  // rows in the smearing map where the cosmic hit the CCD             [pixel]
        vector<unsigned int> cosmicEntryColSmearingMap;  // columns in the smearing map where the cosmic hit the CCD          [pix]
        vector<double> cosmicsTrailsSmearingMap;         // length of the trails of the cosmics that hit the smearing map     [pix]
        vector<double> cosmicsAnglesSmearingMap;         // angle at which the cosmic hits the CCD in the smearing map        [rad]
        vector<double> cosmicsIntensitiesSmearingMap;    // total number of electrons the cosmic will release over its trail  [e-]

        vector<unsigned int> cosmicEntryRowBiasMapLeft;  // rows in the left bias map where the cosmic hit the CCD            [pixel]
        vector<unsigned int> cosmicEntryColBiasMapLeft;  // columns in the left bias map where the cosmic hit the CCD         [pix]
        vector<double> cosmicsTrailsBiasMapLeft;         // length of the trails of the cosmics that hit the left bias map    [pix]
        vector<double> cosmicsAnglesBiasMapLeft;         // angle at which the cosmic hits the CCD in the left bias map       [rad]
        vector<double> cosmicsIntensitiesBiasMapLeft;    // total number of electrons the cosmic will release over its trail  [e-]

        vector<unsigned int> cosmicEntryRowBiasMapRight; // rows in the righ bias map where the cosmic hit the CCD            [pixel]
        vector<unsigned int> cosmicEntryColBiasMapRight; // columns in the righ bias map where the cosmic hit the CCD         [pix]
        vector<double> cosmicsTrailsBiasMapRight;        // length of the trails of the cosmics that hit the righ bias map    [pix]
        vector<double> cosmicsAnglesBiasMapRight;        // angle at which the cosmic hits the CCD in the righ bias map       [rad]
        vector<double> cosmicsIntensitiesBiasMapRight;   // total number of electrons the cosmic will release over its trail  [e-]
        int cosmicSubgroupIndex=-1;                      // Keeps track of which subgroup in HDF5 file is being filled when saving the cosmics

        vector<double> relTransmissivityCoefVector;      // To take into account the transmissivity, including the vignetting
        vector<double> gainNonlinearityCoefficients;     // To take into account the non-linear behaviour of the gain
        double radiusFOV;                        // Radius of the FOV [radians]
        double expectedValueRelativeTransmissivity; // Expected value of the relative transmissivity for the sub-field
        double expectedValuePolarization;        // Expected value of the throughput efficiency due to polarisation
        double particulateContaminationEfficiency;  // Efficiency of particulate contamination (in [0,1])
        double molecularContaminationEfficiency;    // Efficiency of molecular contamination (in [0,1])
        double meanQE;							 // Mean QE (over all wavelengths)
        double meanAngleDependencyQE;			 // Mean (over all pixels) of the relative efficiency due to the angle dependency of the QE
        double serialTransferTime;				 // Time to shift the content of the readout register by one pixel [s]
        double parallelTransferTime;			 // Time to shift the charges one row down in case the readout register will be read out [s]
        double parallelTransferTimeFast;	     // Time to shift the charges one row down in case the readout register will not be read out [s]
        bool isFastCamera;                       // Indicates whether or not the camera is a fast camera
        bool includeShield;                      // Indicates whether or not to include a metacllic shield around CCD
        int firstRowPartialReadout;			     // First row that will be read out by the FEE in partial readout mode
        int numRowsPartialReadout;			     // Number of rows that will be read out by the FEE, starting at firstRowReadout, in partial readout mode
        double readoutNoise;                     // Mean readout noise [electrons]
        double refValueGainLeft;                 // Reference value for the gain on the ACD reading the left-hand side of the detector [µV/e-]
        double refValueGainRight;                // Reference value for the gain on the ACD reading the right-hand side of the detector [µV/e-]
        double gainStability;                    // Gain stability [µV/e-]
        double gainAllowedDifference;            // Allowed difference in gain between the left and the right half of the detector [% of the reference values]
        double combinedGainLeft;                 // Combined (CCD + FEE) gain for the left half of the CCD      [ADU / e-]
        double combinedGainRight;                // Combined (CCD + FEE) gain for the right half of the CCD     [ADU / e-]
        unsigned long fullWellSaturationLimit;   // Full-well saturation limit [electrons/pixel]
        unsigned int electronicOffset;           // Bias or electronic offset [ADU]
        unsigned long digitalSaturationLimit;    // Digital saturation limit [ADU / pixel]
        double darkCurrent;						 // Dark current [e- / s]
        double dsnu;							 // Dark signal non-uniformity
        double darkCurrentStability;             // Temperature stability of the dark current [e / K / s]
        bool writePixelMaps;                     // Whether or not to write the pixel maps of the subfield to the HDF5 file, for each exposure
        bool writeBiasMaps;                      // Whether or not to write the bias maps (left and right) to the HDF5 file, for each exposure
        bool writeBackgroundMap;                  // Wheter or not to write the background map to the HDF5 file.
        bool writeSmearingMaps;                  // Whether or not to write the smearing maps to the HDF5 file, for each exposure
        bool writeThroughputMaps;                // Whether or not to write the throughput maps to the HDF5 file, for each exposure
        bool writeFlatfieldMap;                  // Whether or not to write the flatfield map to the HDF5 file, for each exposure  
        bool writeCosmics;                       // Whether or not to write the cosmics row, column and flux to the HDF5 file, for each exposure
        bool writeCTI;                           // Whether or not to write the BOL/EOL trap density maps
                                                 // for each species to the HDF5 file
        bool writeBadPixelMap;                   // Wheter or not to write the bad pixel map

        string CTImodel;
        double meanCte;                          // Mean charge-transfer efficiency  (in [0,1])
        double beta;                             // Beta exponent in Short et al., MNRAS 430, 3078-3085 (2010).
        double temperature;                      // Temperature of the detector
        unsigned int numTrapSpecies;             // Number of different trap species included in the Short2010 model
        vector<double> meanTrapDensityBOL;       // mean (averaged over entire CCD) trap density of each trap species at Beginning-Of-Live [traps/pixel]
        vector<double> meanTrapDensityEOL;       // mean (averaged over entire CCD) trap density of each trap species at End-Of-Live [traps/pixel]
        vector<double> trapCaptureCrossSection;  // For each trap species: the trap capture cross section [m^2]
        vector<double> releaseTime;              // For each trap species: the electron release time [s]

        HDF5File CTIFile;                        // Input CTI file with the trap density maps, contains the info below.
        arma::Mat<float> radiationMap;           // Normalized radiation map for the subfield under consideration [p+ / s]
        arma::Mat<float> radiationSmearingMap;   // Normalized radiation map for the smearing map under consideration [p+ / s]
        arma::Mat<float> numberOfOccupiedTrapsPixelMap;      // Number of occupied traps in the Short2013 CTI model, when specified in ...
        arma::Mat<float> numberOfOccupiedTrapsSmearingMap;   // ... a file, to allow a for spatially varying trap density.

        double chargeInjectionLevel;             // Percentage of the full well to be filled by charge injection [0-100]
        int injectionRowInterval;                // Charge will be injected every XX CCD row [integer: in 1 - numrows] starting from firstInjectedRow.
        int firstInjectedRow;                    // First CCD row that will be injected. 0 is the row closest to the readout register.

        string readoutMode;                      // Readout mode (Nominal / Partial)
        double readoutTimeBeforeNextExposure;    // Duration of the readout before the next exposure can start [s]
        double readoutTimeDuringNextExposure;    // Duration of the readout when the next exposure has already started [s]

        bool includeBFE;                         // Whether or not to include the BFE
        bool includeDarkSignal;                  // Whether or not to include dark
        bool includeFieldDistortion;             // Whether or not to include field distortion
        bool includePhotonNoise;                 // Whether or not to include photon noise
        bool includeReadoutNoise;                // Include readout noise [yes or no]
        bool includeCTIeffects;                  // Include CTI effects [yes or no]
        bool includeChargeInjection;             // Include charge injection to mitigate the CTI [yes or no]
        bool includeOpenShutterSmearing;         // Include trails due reading out with an open shutter
        bool includeQuantumEfficiency;           // Include loss of throughput due to quantum efficiency
        bool includeRelativeTransmissivity;      // Include overall relative transmissivity
        bool includePolarization;                // Include loss of throughput due to polarisation
        bool includeParticulateContamination;    // Include loss of throughput due to particulate contamination
        bool includeMolecularContamination;      // Include loss of throughput due to molecular contamination
        bool includeFullWellSaturation;          // Whether or not full well saturation should be applied
        bool includeDigitalSaturation;           // Whether or not digital saturation should be applied
        bool includeQuantisation;                // Whether or not to include quantisation
        bool includeGainNonlinearity;            // Whether or not to include a non-linear gain (e- -> ADU)
        double transmissionEfficiencyBOS;        // Transmission efficiency at the begining of the simulation.

        int beginExposureNr;                     // Sequential number of the very first exposure. See yaml input file.
        int finalExposureNr;                     // Sequential number of the very last exposure: beginExposureNr + numExposures

        double badPixelParameter;
        double nominalOperatingTemperature;
        double internalTime;

        long darkSignalSeed;
        long readoutNoiseSeed;
        long photonNoiseSeed;
        long cosmicSeed;
        long badPixelSeed;

        mt19937 darkSignalGenerator;
        mt19937 darkNoiseGenerator;
        mt19937 photonNoiseGenerator;
        mt19937 readoutNoiseGenerator;
        mt19937 cosmicHitRateGenerator;
        mt19937 cosmicEntryRowGenerator;
        mt19937 cosmicEntryColumnGenerator;
        mt19937 cosmicEntryAngleGenerator;
        mt19937 cosmicTrailLengthGenerator;
        mt19937 cosmicIntensityGenerator;
        mt19937 decimalNumCosmicHitsGenerator;
        mt19937 badPixelGenerator;
    

        normal_distribution<double> darkSignalDistribution;
        normal_distribution<double> darkNoiseDistribution;
        poisson_distribution<long> photonNoiseDistribution;
        normal_distribution<double> readoutNoiseDistribution;
        poisson_distribution<long> cosmicHitRateDistribution;
        uniform_real_distribution<double> cosmicEntryRowDistribution;
        uniform_real_distribution<double> cosmicEntryColumnDistribution;
        uniform_real_distribution<double> cosmicEntryAngleDistribution;
        uniform_real_distribution<double> cosmicTrailLengthDistribution;
        skew_normal_distribution cosmicIntensityDistribution;
        uniform_real_distribution<double> decimalNumCosmicHitsDistribution;
        bernoulli_distribution badPixelDistribution;

        Camera &camera;
        FrontEndElectronics *frontEndElectronics;

    private:

        TemperatureGenerator &temperatureGenerator;

        void readCTIinputFile(string ctiInputFile);
};

arma::Mat<uint16_t> floatToUint(arma::Mat<float> &A, string mapName);

#endif

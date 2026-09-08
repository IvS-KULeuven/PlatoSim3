#ifndef DETECTORWITHANALYTICGAUSSIANPSF_H
#define DETECTORWITHANALYTICGAUSSIANPSF_H

#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

#include "armadillo"

#include "Constants.h"
#include "Units.h"
#include "Detector.h"

using namespace std;



class DetectorWithAnalyticGaussianPSF: public Detector
{
    public:

        DetectorWithAnalyticGaussianPSF(ConfigurationParameters &configParam, HDF5File &hdf5File, Camera &camera, TemperatureGenerator &feeTemperatureGenerator, TemperatureGenerator &detectorTemperatureGenerator, double readoutTimeBeforeNextExposure, double readoutTimeDuringNextExposure);
        virtual ~DetectorWithAnalyticGaussianPSF();

        virtual double takeExposure(int exposureNr, double startTime, double exposureTime) override;

        void configure(ConfigurationParameters &configParam);

        virtual tuple<bool, double, double> addFlux(double xFP, double yFP, double flux) override;
        virtual void addFlux(double flux) override;
        virtual tuple<bool, double, double> addExtendedGhost(double xFP, double yFP, double radius, double flux) override;

    protected:

        virtual void integrateLight(int exposureNr, double startTime, double exposureTime) override;
        virtual void applyFlatfield() override;
        virtual void generateFlatfieldMap();
        virtual void readInFlatfieldMap();

        arma::Mat<float> flatfieldMap;      // Pixel flatfield map

        double sigma00;                     // Stdev of Gaussian PSF in x- and y-direction at the optical axis      [pix]
        double sigmaX18;                    // Stdev of Gaussian PSF in x-direction at 18 deg from the optical axis [pix]
        double sigmaY18;                    // Stdev of Gaussian PSF in y-direction at 18 deg from the optical axis [pix]

        double flatfieldNoiseRMS;     // Peak-to-peak noise amplitude

        bool includeFlatfield;              // Whether or not to include flat fielding
        long flatfieldSeed;
        bool writeFlatfieldMap;             // Whether or not to write the flatfield map to the HDF5 file
        string flatfieldFilePath;            // Path of the flatfield file
        string flatfieldSource;              // Determines how the flatfield gets determined 

    private:

};


#endif

#ifndef POINTSPREADFUNCTION_H
#define POINTSPREADFUNCTION_H

#include <string>
#include <array>
#include <vector>
#include <algorithm>

#include "ArrayOperations.h"
#include "ConfigurationParameters.h"
#include "Parameter.h"
#include "HDF5File.h"
#include "HDF5Writer.h"


using namespace std;





class PointSpreadFunction : public HDF5Writer
{
    public:

        PointSpreadFunction(ConfigurationParameters &configParam, HDF5File &hdf5File);
        ~PointSpreadFunction();

        void initHDF5Groups() override;
        void flushOutput() override;
        void select(double xFP, double yFP);
        void rotate(double angle);

        int getNumSubPixelsPerPixel(){return numberOfSubPixelsPerPixel;};
        arma::fmat rebinToSubPixels(unsigned int targetSubPixels);
        arma::fmat getOriginalPSF();
        vector<std::array<double, 4>> getDistortionMap();
        bool writeHighResolutionPSF;
        bool writeDiffusedPSF;
        bool includeChargeDiffusion;
        vector<std::array<double, 4>> distortionMap;

    protected:

        void configure(ConfigurationParameters &configParam);

        arma::fmat rebinToPixels();
        arma::fmat getGaussianPsf();

        // Indicates whether a PSF has been selected

        bool isSelected = false;

        // Indicates whether the PSF has been rotated

        bool isRotated = false;

        // Indicates whether this PSF has been rebinned

        bool isRebinned = false;

        // Absolute path to the HDF5 file with the PSFs

        string absolutePath;

        // The HDF5 file that holds the PSFs

        HDF5File psfFile;

        // The selected psf is copied into this array

        arma::Mat<float> psfMap;

        // Number of sub-pixels per pixel used to generate the PSFs

        unsigned int numberOfSubPixelsPerPixel;

        // Number of pixels in the field that holds the PSFs

        unsigned int numberOfPixels;

        // Focal length
        double focalLengthValue;

        // Actual rotation angle of the PSF with respect to the x-axis orientation of the focal plane

        double rotationAngle = 0.0;    // [radians]

        void initializeDistortionMap();
        void readDistortionmapFromFile();
        void generateDistortionMap();
};

#endif

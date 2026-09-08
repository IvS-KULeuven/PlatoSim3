#include "DetectorWithAnalyticGaussianPSF.h"

/**
 * \brief Constructor.
 *
 * \details
 *
 * The constructor initializes the groups in the HDF5 file where the different maps (i.e. pixel map,
 * bias register map, smearing map, etc.) will be saved.
 *
 * The following maps are initialized to zero (partly through the base class Detector):
 *
 * pixelMap
 * subPixelMap
 * biasMap
 * smearingMap
 * flatfieldMap
 * throughputMap
 * cteMap
 *
 * The flatfieldMap, throughputMap and cteMap are filled at pixel level.
 *
 * \param configParam    Configuration parameters for the detector.
 * \param hdf5file       HFD5 file to write the detector images to.
 * \param camera         Camera to which to attach the detector.
 * \param readoutTimeBeforeNextExposure Duration of the readout that takes place before the next exposure can start.
 */

DetectorWithAnalyticGaussianPSF::DetectorWithAnalyticGaussianPSF(ConfigurationParameters &configParam, HDF5File &hdf5file, Camera &camera, TemperatureGenerator &feeTemperatureGenerator, TemperatureGenerator &detectorTemperatureGenerator, double readoutTimeBeforeNextExposure, double readoutTimeDuringNextExposure)
: Detector(configParam, hdf5file, camera, feeTemperatureGenerator, detectorTemperatureGenerator, readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure)
{
    // Parse the parameters from the configuration file.

    configure(configParam);

    // Allocate memory for the flatfield map

    flatfieldMap.ones(numRowsPixelMap, numColumnsPixelMap);

    if(includeFlatfield)
    {
        if (flatfieldSource == "FromRedNoise")
        {
            // Generate the flatfield map

            generateFlatfieldMap();
        }
        else if (flatfieldSource == "FromFile")
        {
            // Read in the flatfield map

            readInFlatfieldMap();
        }      
    }
}








/**
 * Destructor.
 *
 */
DetectorWithAnalyticGaussianPSF::~DetectorWithAnalyticGaussianPSF()
{
    flushOutput();
}










/**
 * \brief Configure the DetectorWithAnalyticGaussianPSF object using the ConfigurationParameters
 *
 * \param configParam: the configuration parameters
 **/

 void DetectorWithAnalyticGaussianPSF::configure(ConfigurationParameters &configParam)
 {
    // Treat the specific configurations for a analytical Gaussian PSF

    sigma00 = configParam.getDouble("PSF/AnalyticGaussian/Sigma00");
    sigmaX18 = configParam.getDouble("PSF/AnalyticGaussian/SigmaX18");
    sigmaY18 = configParam.getDouble("PSF/AnalyticGaussian/SigmaY18");

    // Get the configuration parameters for the PRNU

    includeFlatfield          = configParam.getBoolean("CCD/IncludeFlatfield");
    if (includeFlatfield)
    {
        flatfieldSource = configParam.getString("CCD/Flatfield/Source");
        if (flatfieldSource == "FromFile")
        {
            flatfieldFilePath = configParam.getAbsoluteFilename("CCD/Flatfield/FromFile/FilePath");
        }
        else if (flatfieldSource == "FromRedNoise")
        {
            flatfieldNoiseRMS = configParam.getDouble("CCD/Flatfield/FromRedNoise/FlatfieldNoiseRMS");
            flatfieldSeed     = configParam.getLong("RandomSeeds/FlatFieldSeed");
        }
    }

    // The configuration for the HDF5 contents

    writeFlatfieldMap = configParam.getBoolean("ControlHDF5Content/WriteFlatfieldMap");
 }









/**
 * \brief: Read in the flatfield variations from a file.
 *
 */

void DetectorWithAnalyticGaussianPSF::readInFlatfieldMap()
{
    HDF5File prnuFile;
    Log.info("Detector: reading in flatfield map.");

    // Prepare the prnu map by performing some basic checks

    if (!FileUtilities::fileExists(flatfieldFilePath))
    {
      throw FileException("Detectors: trying to load flatfield file (" + flatfieldFilePath + "), but file doesn't exist.");
    }

    try
    {
        prnuFile.open(flatfieldFilePath);
    }
    catch (H5::FileIException ex)
    {
        Log.error("H5::FileIException: " + string(ex.getCDetailMsg()));
        throw H5FileException("Detector: Could not open flatfield HDF5 file: " + flatfieldFilePath);
    }

    // Select the correct subregion from our prnuFile
    arma::Mat<float> prnuMap;
    prnuFile.readArray("/", "PRNU", prnuMap);

    flatfieldMap = prnuMap.submat(subFieldZeroPointRow, subFieldZeroPointColumn,
                                   subFieldZeroPointRow+numRowsPixelMap-1,
                                   subFieldZeroPointColumn+numColumnsPixelMap-1);


    prnuFile.close();
    if (writeFlatfieldMap)
    {
        Log.debug("Detector: writing PRNU to HDF5");
        hdf5File.createGroup("/Flatfield");
        hdf5File.writeArray("/Flatfield", "PRNU", flatfieldMap);
    }
}










 /**
 * \brief: Generate the (random) flatfield variations. This map is generated
 *         at pixel level but without the edge pixels.
 *
 * https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py#L108
 */

void DetectorWithAnalyticGaussianPSF::generateFlatfieldMap()
{

    Log.info("Detector: generating flatfield map.");

    // Random number generation

    mt19937 flatfieldGenerator(flatfieldSeed);
    normal_distribution<double> flatfieldDistribution(0.0, 1.0);

    // Double the dimensions (this is necessary because of the behaviour of the Fourier transforms)

    int unsigned Nrows    = 2 * numRowsPixelMap;
    int unsigned Ncolumns = 2 * numColumnsPixelMap;

    arma::cx_fmat evenMap = arma::cx_fmat(Nrows, Ncolumns);

    for(unsigned int row = 0; row < Nrows; row++)
    {
        for(unsigned int column = 0; column < Ncolumns; column++)
        {
            // Fourier space: generate white noise and include 1/f dependency
            // (Note: see https://en.wikipedia.org/wiki/Pink_noise#Generalization_to_more_than_one_dimension)

            evenMap(row, column) = flatfieldDistribution(flatfieldGenerator) / (pow(row, 2) + std::pow(column, 2) + 1);
        }
    }

    // Take the real part of the inverse Fourier transform

    evenMap = arma::ifft2(evenMap);
    arma::fmat realMap = arma::real(evenMap);

    // Cut out the appropriate part

    unsigned int numRowsFlatfield = Nrows / 2;
    unsigned int numColumnsFlatfield = Ncolumns / 2;

    flatfieldMap(arma::span::all, arma::span::all) = realMap(arma::span(0, numRowsFlatfield - 1), arma::span(0, numColumnsFlatfield - 1));
    flatfieldMap.reshape(numRowsFlatfield * numColumnsFlatfield, 1);

    // Normalisation
    //  - divide by mean and subtract 1.0 -> mean = 0.0
    //  - scale such that std.dev. = flatfield RMS and mean = 0.0
    //  - add 1.0

    flatfieldMap /= arma::mean(flatfieldMap.col(0));
    flatfieldMap -= 1;
    double scale = flatfieldNoiseRMS / arma::stddev(flatfieldMap.col(0));
    flatfieldMap *= scale;
    flatfieldMap += 1;

    flatfieldMap.reshape(numRowsFlatfield, numColumnsFlatfield);

    // Write the result to the HDF5 output file

    if (writeFlatfieldMap)
    {
        Log.debug("Detector: writing PRNU to HDF5");
        hdf5File.createGroup("/Flatfield");
        hdf5File.writeArray("/Flatfield", "PRNU", flatfieldMap);
    }
}












/**
 * \brief: Take an exposure with the detector starting at the given time.
 *         The light is integrated during the given exposure time, during which
 *         the detector experiences the effects of jitter and thermo-elastic telescope
 *         drift. The background is assumed uniform for the whole subfield.
 *         Afterwards, the collected light is read out, and various noise effects are added.
 *
 * \param exposureNr:   Sequential number of the exposure
 * \param startTime:    Starting time of the exposure [s].
 * \param exposureTime: Duration of the exposure [s].
 *
 * \return endTime:     Time after the exposure (startTime + exposureTime + readoutTime)
 *
 * \pre Pixel, bias register, and smearing map filled with values from previous exposure.
 *
 * \post Pixel unit in the pixel, bias register, and smearing maps: [ADU]
 */

double DetectorWithAnalyticGaussianPSF::takeExposure(int exposureNr, double startTime, double exposureTime)
{
    // Advance the internal clock until the given start time

    internalTime = startTime;

    // Clear all arrays

    reset();

    // Integration of point sources and background, taking into account jitter + drift.

    if (!constantSkyBackground && (exposureNr == beginExposureNr))
    {
        fillBackgroundMap(camera, startTime, exposureTime);
    }

    Log.info("Detector: Integrating light for exposure " + to_string(exposureNr) + " with exposure time = " + to_string(exposureTime));

    integrateLight(exposureNr, startTime, exposureTime);

    // If this is the first exposure, we should initialize the number of occupied traps.
    // This can only be done after the detector
    // has been exposed to the skybackground.
    // => Check if CTI is included && we use the Short2013 model

    if (exposureNr == beginExposureNr) {
        if (includeCTIeffects && (CTImodel == "Short2013" || CTImodel == "Short2013FromFile")) {
            setInitialNumberOfOccupiedTraps(numberOfOccupiedTrapsPixelMap);
        }
    }

    // Include noise effects like readout noise, photon noise, full well saturation, etc.
    // Note: readOut() needs the exposure time to compute the open shutter smearing.

    Log.info("Detector: Adding noise effects to exposure " + to_string(exposureNr));
    readOut(exposureTime);

    // Write the CCD subfield, the bias map, and the smearing map to the HDF5 file

    Log.debug("Detector: Writing PixelMap, smearing map, and bias map #" + to_string(exposureNr) + " to HDF5 file.");
    writePixelMapsToHDF5(exposureNr);

    // Write the cosmic hits to the HDF5 file

    Log.debug("Detector: Writing Cosmics of the PixelMap, smearing map, bias map #" + to_string(exposureNr) + " to HDF5 file.");

    if (writeCosmics)
    {
        writeCosmicHitsToHDF5(exposureNr);
    }

    // Advance the internal clock

    internalTime += exposureTime + readoutTimeBeforeNextExposure;

    return internalTime;
}












/**
 * \brief: During an exposure, this method makes the detector integrate the light
 *         in small steps. During each step the slight change of star positions due
 *         to spacecraft jitter is taken into account.
 *
 *  \details  Besides jitter, also the sky background, and the flatfield is taken into
 *            account. The sub-pixel map is rebinned in a pixel map.  After rebinning,
 *            vignetting and polarisation are applied (if applicable).
 *
 * \param exposureNr:   Sequential number of the exposure
 * \param startTime:    Starting time of the exposure [s].
 * \param exposureTime: Duration of the exposure [s].
 *
 * \pre Pixel, bias register, and smearing map filled with values from previous exposure.
 *
 * \post Pixel unit of the sub-pixel map: [photons].
 * \post Pixel, bias register, and smearing map filled with zeroes.
 */

void DetectorWithAnalyticGaussianPSF::integrateLight(int exposureNr, double startTime, double exposureTime)
{

    // Integration (incl. jitter): point sources + background
    // PixelMap units after: [photons]

    camera.exposeDetectorWithStars(*this, startTime, exposureTime);
    if (constantSkyBackground)
    {
        camera.exposeDetectorWithSkyBackground(*this, startTime, exposureTime, readoutTimeBeforeNextExposure);
    }
    else
    {
        addBackgroundMapToPixelMap(camera, startTime);
    }

    // Apply throughput efficiency on the pixel map.
    // This takes into account the QE, vignetting, polarisation, and particulate & molecular contamination.
    // PixelMap units change from [photons] to [electrons]

    applyThroughputEfficiency();

    // Apply the charge injection which will mitigate the CTI. The injection happens in electrons,
    // so the throughput efficiency should already have been applied. The injected charges do feel the PRNU,
    // so applying the flatfied should happen afterwards.

    if (includeChargeInjection)
    {
        Log.debug("Detector: applying charge injection");
        applyChargeInjection();
    }

    // Apply flatfield (at pixel level)
    // PixelMap units after: [photons]

    if (includeFlatfield)
    {
        Log.debug("Detector: applying Flatfield.");

        applyFlatfield();
    }
    else
    {
        Log.debug("Detector: no flatfield applied.");
    }

    // Apply the effects of readout smearing due to an open shutter. Because there is no shutter,
    // the pixels are still receiving photons from the sky, while they are being transfered towards
    // the readout register.
    // Pixel units before: [electrons]
    // Pixel units after: [electrons]

    if (includeOpenShutterSmearing)
    {
        Log.debug("Detector: applying open shutter smearing.");
        applyOpenShutterSmearing(exposureTime);
    }
    else
    {
         Log.debug("Detector: no open shutter smearing applied.");
    }

    // Apply poisson distributed photon noise
    // Pixel units before: [electrons]
    // Pixel units after: [electrons]

    if (includePhotonNoise)
    {
        Log.debug("Detector: adding photon noise.");
        addPhotonNoise();
    }
    else
    {
        Log.debug("Detector: no photon noise added.");
    }

    // Add dark current

    if(includeDarkSignal)
    {
        Log.debug("Detector: adding dark current");

       	addDarkSignal(exposureTime);
    }
    else
    {
        Log.debug("Detector: no dark current added");
    }
}













/**
 * \brief: Add the PSF of the star with given focal plane coordinates and flux level to the pixel map.
 *         Return the pixel coordinates of the barycenter of the PSF. As PSF we use a Gaussian which
 *         is symmetric at the optical axis, and becomes more asymmetric and fatter towards the edge
 *         of the FOV (although the user can set the parameters such that the effect can be opposite).
 *
 * \param xFP   X-coordinate of the (fractional) pixel in the focal plane in the FP reference frame [mm].
 * \param yFP   Y-coordinate of the (fractional) pixel in the focal plane in the FP reference frame [mm].
 * \param flux  Flux to add to the pixel map [photons].
 *
 * \return           (isInSubfield, row, col)
 *                   isInSubfield: True if (xFP, yFP) are on the subfield, false otherwise.
 *                   row:          subfield (not CCD) row number of the barycenter of the PSF.
 *                   col:          subfield (not CCD) column number of the barycenter of the PSF.
 *
 * \note In the code below, we use pixelMap(i,j) that involves a boundary check. To increase speed,
 *       switch to pixelMap.at(i,j) that does not involve a boundary check.
 */

tuple<bool, double, double> DetectorWithAnalyticGaussianPSF::addFlux(double xFP, double yFP, double flux)
{
    // Convert the FP coordinates of the PSF barycenter to (real-valued) CCD pixel coordinates
    // Then convert from CCD pixel coordinates to subfield pixel coordinates.

    double row0, column0;
    tie(row0, column0) = focalPlaneToPixelCoordinates(xFP, yFP);
    row0 -= subFieldZeroPointRow;
    column0 -= subFieldZeroPointColumn;

    // Check if the star falls in the subfield. If not, don't add any flux, but simply return.

    if (!isInPixelMap(row0, column0))
    {
        return make_tuple(false, row0, column0);
    }

    // Depending on the angular distance from the optical axis, the PSF increases in size.
    // Determine the standard deviations in both directions.
    // Recall:
    //    sigma00:    Stdev of Gaussian PSF in x- and y-direction at the optical axis      [pix]
    //    sigmaX18:   Stdev of Gaussian PSF in x-direction at 18 deg from the optical axis [pix]
    //    sigmaY18;   Stdev of Gaussian PSF in y-direction at 18 deg from the optical axis [pix]

    const double theta = rad2deg(camera.getGnomonicRadialDistanceFromOpticalAxis(xFP, yFP));
    const double sigmaX = sigma00 + theta/18.0  * (sigmaX18 - sigma00);
    const double sigmaY = sigma00 + theta/18.0  * (sigmaY18 - sigma00);

    // Construct the covariance matrix of the rotated non-spherial gaussian
    // Note: sigmaX corresponds to the column direction, sigmaY to the row direction.

    const double angle = Constants::PI - atan2(yFP, xFP);

    arma::mat R(2,2);
    R << cos(angle) << -sin(angle) << arma::endr
      << sin(angle) <<  cos(angle) << arma::endr;

    arma::mat invS(2,2);
    invS << 1./(sigmaY*sigmaY) <<         0          << arma::endr
         <<          0         << 1./(sigmaX*sigmaX) << arma::endr;

    const arma::mat invCov = R * invS * R.t();                      // Inverse of the covariance matrix

    // Compute the prefactor of the 2D gaussian, that contains the determinant of the cov matrix

    const double detCov = 1./arma::det(invCov);
    double preFactor = 1./(2*Constants::PI * sqrt(detCov));

    // Take care that the total flux is conserved over the pixel range that we're computing the PSF.
    // Compute the integral (sum) over the total range (not only the visible range).

    const arma::colvec mu = {row0, column0};
    double sum = 0.0;

    int range = max(6*floor(sigmaX), 6*floor(sigmaY));

    for (int row = int(floor(row0))-range; row < int(floor(row0))+range+1; row++)
    {
        for (int col = int(floor(column0))-range; col < int(floor(column0))+range+1; col++)
        {
            const arma::colvec x = {double(row), double(col)};
            sum += preFactor * arma::exp(-0.5 * (x - mu).t() * invCov * (x-mu)).at(0,0);
        }
    }

    preFactor = preFactor / sum * flux;

    // Paste the 2D gaussian PSF into the pixelMap

    for (int row = int(floor(row0))-range; row < int(floor(row0))+range+1; row++)
    {
        if ((row < 0) || (row >= numRowsPixelMap)) continue;

        for (int col = int(floor(column0))-range; col < int(floor(column0))+range+1; col++)
        {
            if ((col < 0) || (col >= numColumnsPixelMap)) continue;

            arma::colvec x = {double(row), double(col)};
            pixelMap(row, col) += preFactor * arma::exp(-0.5 * (x - mu).t() * invCov * (x-mu)).at(0,0);
        }
    }

    // That's it!

    return make_tuple(true, row0, column0);
}


/**
 * \brief Insert the extended ghost with the given radius and flux at the given focal-plane position.
 *
 * Note that the extended source will not be convolved with the PSF, for practical reasons (but since the
 * extended ghosts are so large, the influence of the PSF is negligible).
 *
 * \param x0: Focal-plane x-coordinate of the centre of the extended ghost [mm].
 * \param y0: Focal-plane y-coordinate of the centre of the extended ghost [mm].
 * \param radius: Radius of the extended ghost [mm].
 * \param flux: Flux of the extended ghost [photons].
 *
 * \return: Whether or not the extended source falls (at least partially) on the sub-field, and the
 *          (row, column) coordinates of the centre of the extended ghost in the pixel map.
 */

tuple<bool, double, double> DetectorWithAnalyticGaussianPSF::addExtendedGhost(double x0, double y0, double radius, double flux)
{
    // Calculate the number of pixels in the extended ghost

    double radiusPixels = radius * 1000 / pixelSize;    // Radius [pixels]
    double radiusPixelsSquared = pow(radiusPixels, 2);  // Squared radius [pixels^2]

    double numPixels = PI * pow(radiusPixels, 2);       // Area of the extended ghost [pixels]
    double fluxPerPixel = flux / numPixels;             // Flux [photons / pixel]

    // Calculate the (row, column) coordinates of the centre of the extended source in the pixel map

    double row0, column0;
    tie(row0, column0) = focalPlaneToPixelCoordinates(x0, y0);
    row0 -= subFieldZeroPointRow;
    column0 -= subFieldZeroPointColumn;

    bool ghostInPixelMap = false;

    // Try to add flux to all pixels covered by the extended ghosts

    for(int row = row0 - radiusPixels; row <= row0 + radiusPixels; row++)
    {
        for(int column = column0 - radiusPixels; column <= column0 + radiusPixels; column++)
        {
            if (isInPixelMap(row, column) && pow(column - column0, 2) + pow(row - row0, 2) <= radiusPixelsSquared)
            {
                ghostInPixelMap = true;
                pixelMap(row, column) += fluxPerPixel;
            }
        }
    }

    return  make_tuple(ghostInPixelMap, row0, column0);
}








/**
 * \brief: Add the given flux value to (all sub-pixels of) the sub-pixel map.
 *
 * \param flux: Flux to add to the sub-pixel map [photons/pixel].
 *
 */

void DetectorWithAnalyticGaussianPSF::addFlux(double flux)
{
    bool subFieldIsBlockedOff = (coveredBottom + coveredTop >= numRowsPixelMap) || (coveredLeft + coveredRight >= numColumnsPixelMap);
    if (!subFieldIsBlockedOff)
    {
      pixelMap.submat(coveredBottom, coveredLeft,
                      numRowsPixelMap - coveredTop - 1,
                      numColumnsPixelMap - coveredRight - 1) += flux;
    }
}










/**
 * \brief: Multiply the sub-pixel map with the flatfield.
 *
 * NOTE: The sub-pixel map contains extra edge pixels, but the flatfield
 *       map does not. These edge pixels are excluded from this flatfield
 *       multiplication.
 *
 * \pre Unit of the sub-pixels: [photons].
 * \pre Flatfield map at sub-pixel level, excl. edge pixels.
 * \pre Pixel, bias register, and smearing maps filled with zeroes.
 *
 * \post Pixel value in the sub-pixel map: [photons].
 * \post Pixel, bias, and smearing maps filled with zeroes.
 */

void DetectorWithAnalyticGaussianPSF::applyFlatfield()
{
    const unsigned int beginRow = numEdgePixels;
    const unsigned int beginCol = numEdgePixels;
    const unsigned int endRow = numRowsPixelMap - numEdgePixels - 1;
    const unsigned int endCol = numColumnsPixelMap - numEdgePixels - 1;

    pixelMap.submat(beginRow, beginCol, endRow, endCol) = pixelMap.submat(beginRow, beginCol, endRow, endCol) % flatfieldMap;
}


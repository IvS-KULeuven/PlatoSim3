
#include "Detector.h"

/**
 * \brief Add a part to the definite integral of the analytic PSF and adjust the normalization factor accordingly.
 *
 * \param ox:    relative x offset
 * \param oy:    relative y offset
 * \param h:     relative height
 * \param sigma: width of Gaussian term
 * \param r:     strength and type of periodic term
 * \param rho:   distance between Gaussian term and periodic term
 * \param phi:   orientation of periodic term
 *
 * \return reference to itself
 **/
IntegralOfAnalyticSignalResponse& IntegralOfAnalyticSignalResponse::addPart(double ox, double oy, double h, double sigma, double r, double rho, double phi)
{
    using Faddeeva::erf;

    ox += (size - (size & 1)) / 2.;
    oy += (size - (size & 1)) / 2.;

    erfxr.emplace_back(size + 1);
    erfyr.emplace_back(size + 1);
    double sr = 1. / sqrt(2.) / sigma;
    double fr1 = sqrt(M_PI * fabs(h) * sigma * sigma / (r != 0.? 4.: 2.));
    double fr2 = h < 0.? -fr1: fr1;

    // dsigma is the Gaussian diffusion kernel width

    if (dsigma != 0.)
        sr /= sqrt(2. * sr * sr * dsigma * dsigma + 1.);

    erfxr.back()[0] = erf(-sr * ox);
    erfyr.back()[0] = erf(-sr * oy);
    for (unsigned i = 0; i < size; i++)
    {
        erfxr.back()[i] = ((erfxr.back()[i + 1] = erf(sr * (i + 1. - ox))) - erfxr.back()[i]) * fr1;
        erfyr.back()[i] = ((erfyr.back()[i + 1] = erf(sr * (i + 1. - oy))) - erfyr.back()[i]) * fr2;
    }
    n += 4. * fr1 * fr2;

    if (r != 0.)
    {
        erfxc.emplace_back(size + 1);
        erfyc.emplace_back(size + 1);
        double delta = 2. * M_PI * sigma * sigma / r / r;
        complex<double> sc = 1. / sqrt(2.) / sigma * sqrt(complex<double>(1., delta));
        complex<double> xc = complex<double>(0., M_PI / fabs(r) * sqrt(rho) * cos(phi)) / sc;
        complex<double> yc = complex<double>(0., M_PI / fabs(r) * sqrt(rho) * sin(phi)) / sc;
        complex<double> fc = sqrt((r < 0.? -fr1: fr1) * fr2 * exp(-M_PI * rho / (1. + delta * delta) * complex<double>(delta, 1.)) / complex<double>(1., delta));

        if (dsigma != 0.) {
            complex<double> dc = sqrt(2. * sc * sc * dsigma * dsigma + 1.);
            sc /= dc;
            xc /= dc;
            yc /= dc;
        }

        erfxc.back()[0] = erf(xc - sc * ox);
        erfyc.back()[0] = erf(yc - sc * oy);
        for (unsigned i = 0; i < size; i++)
        {
            erfxc.back()[i] = ((erfxc.back()[i + 1] = erf(xc + sc * (i + 1. - ox))) - erfxc.back()[i]) * fc;
            erfyc.back()[i] = ((erfyc.back()[i + 1] = erf(yc + sc * (i + 1. - oy))) - erfyc.back()[i]) * fc;
        }
        n -= 4. * (fc.real() * fc.real() - fc.imag() * fc.imag());
    }
    return *this;
}









/**
 * \brief Get integral of analytic PSF for a (sub)pixel.
 *
 * \param i:    x direction
 * \param j:    y direction
 * \param norm: apply normalization
 *
 * \return normalized or unnormalized integral of analytic PSF for (sub)pixel (i, j)
 **/

double IntegralOfAnalyticSignalResponse::operator()(unsigned i, unsigned j, bool norm)
{
    if (i >= size || j >= size)
    {
        return 0.;
    }

    double ret = 0.;
    for (unsigned k = 0; k < erfxr.size() && k < erfyr.size(); k++)
    {
        ret += erfxr[k][i] * erfyr[k][j];
    }

    for (unsigned k = 0; k < erfxc.size() && k < erfyc.size(); k++)
    {
        ret -= erfxc[k][i].real() * erfyc[k][j].real() - erfxc[k][i].imag() * erfyc[k][j].imag();
    }

    return norm? ret / n: ret;
}






/**
 * \brief Constructor.
 *
 * \details
 *
 * The constructor initializes the groups in the HDF5 file where the different maps (i.e. pixel map,
 * bias register map, smearing map, etc.) will be saved.
 *
 * The following maps are initialized:
 *
 * pixelMap
 * subPixelMap
 * biasMap
 * smearingMap
 * flatfieldMap
 * throughputMap
 * cteMap
 *
 * The flatfieldMap is filled at sub-pixel level, the throughputMap and cteMap are filled at pixel level.
 *
 * \param configParam    Configuration parameters for the detector.
 * \param hdf5file       HFD5 file to write the detector images to.
 * \param camera         Camera to which to attach the detector.
 * \param readoutTimeBeforeNextExposure Duration of the readout that takes place before the next exposure can start.
 */

Detector::Detector(ConfigurationParameters &configParam,
		   HDF5File &hdf5file,
		   Camera &camera,
		   TemperatureGenerator &feeTemperatureGenerator,
		   TemperatureGenerator &detectorTemperatureGenerator,
		   double readoutTimeBeforeNextExposure,
		   double readoutTimeDuringNextExposure)
: HDF5Writer(hdf5file),
  includeCosmicsInSubField(true),
  includeCosmicsInSmearingMap(true),
  includeCosmicsInBiasMap(true),
  includeBFE(true),
  includeDarkSignal(true),
  includePhotonNoise(true),
  includeReadoutNoise(true),
  includeCTIeffects(true),
  includeOpenShutterSmearing(true),
  includeQuantumEfficiency(true),
  includeRelativeTransmissivity(true),
  includeParticulateContamination(true),
  includeMolecularContamination(true),
  includeFullWellSaturation(true),
  includeDigitalSaturation(true),
  internalTime(0.0), camera(camera),
  temperatureGenerator(detectorTemperatureGenerator)
{
    // Parse the parameters from the configuration file.

    configure(configParam);

    this->readoutTimeBeforeNextExposure = readoutTimeBeforeNextExposure;
    this->readoutTimeDuringNextExposure = readoutTimeDuringNextExposure;

    // Create the groups in the HDF5 file where the different maps (i.e. pixel map,
    // bias register map, smearing map, etc.) will be saved. This needs to be done
    // BEFORE other methods write arrays to HDF5.

    initHDF5Groups();

    // Front-end electronics

    frontEndElectronics = new FrontEndElectronics(configParam, hdf5file, feeTemperatureGenerator);

    // Allocate memory for the different maps

    pixelMap.zeros(numRowsPixelMap, numColumnsPixelMap);
    biasMapLeft.zeros(numRowsBiasMap, numColumnsBiasMap);
    biasMapRight.zeros(numRowsBiasMap, numColumnsBiasMap);

    smearingMap.zeros(numRowsSmearingMap, numColumnsPixelMap);
    throughputMap.ones(numRowsPixelMap, numColumnsPixelMap);

    badPixelMap.ones(numRows, numColumns);

    if (!constantSkyBackground)
    {
        // Initialize the background map
        backgroundMap.zeros(numRowsPixelMap, numColumnsPixelMap);
    }

    if (includeStraylight)
    {
        // Include straylight
        straylight = new StrayLight(configParam, hdf5file, camera);
    }

    // If we are going to apply open-shutter smearing, we have to know which pixels are within
    // the FOV (relevant only in case of mechanical vignetting).  When mechanical vignetting is
    // disabled, all pixels of the detector are inside the FOV.

    if(includeOpenShutterSmearing)
    {
        // Mechanical vignetting map:
        //  - no mechanical vignetting: all pixels of the sub-field inside FOV -> all values set to one
        //  - mechanical vignetting: set value of the pixels in the sub-field outside FOV to zero (others should be one) -> on creation of the throughput map

        mechanicalVignettingMask.ones(numRowsPixelMap, numColumnsPixelMap);

        // Number of exposed rows in each column:
        // - no mechanical vignetting: all exposed rows inside FOV (numRows - firstRowExposed)
        // - mechanical vignetting: count the exposed rows (i.e. from firstRowExposed) that are inside FOV

        numExposedRowsInFOV.zeros(numColumnsPixelMap);

        if(!includeRelativeTransmissivity)
            numExposedRowsInFOV.fill(numRows - firstRowExposed);
    }

    // Check whether the gain values for left- and right-hand side of the CCD are not too far apart

    checkGain();

    // Set the seeds of the random number generators

    darkSignalGenerator.seed(darkSignalSeed);
    darkNoiseGenerator.seed(darkSignalSeed + 1);

    photonNoiseGenerator.seed(photonNoiseSeed);
    readoutNoiseGenerator.seed(readoutNoiseSeed);

    cosmicHitRateGenerator.seed(cosmicSeed);
    cosmicEntryRowGenerator.seed(cosmicSeed + 1);
    cosmicEntryColumnGenerator.seed(cosmicSeed + 2);
    cosmicEntryAngleGenerator.seed(cosmicSeed + 3);
    cosmicTrailLengthGenerator.seed(cosmicSeed + 4);
    cosmicIntensityGenerator.seed(cosmicSeed + 5);
    decimalNumCosmicHitsGenerator.seed(cosmicSeed + 6);

    badPixelGenerator.seed(badPixelSeed);

    decimalNumCosmicHitsDistribution = uniform_real_distribution<double>(0, 1);

    badPixelDistribution = bernoulli_distribution(1-badPixelParameter);
    generateBadPixelMap();
    
}











/**
 * \brief: Adds the background map to the pixelMap. This function is only used
 *         is we are dealing with a non-constant background map.
 *
 * \param camera: camera object
 * \param startTime: startTime of current exposure [s]
 *
 */
void Detector::addBackgroundMapToPixelMap(Camera &camera, double startTime)
{
    double transmissionEfficiency = camera.getTransmissionEfficiency(startTime);
    double meanBackground = arma::mean(arma::mean( backgroundMap*transmissionEfficiency));

    pixelMap += backgroundMap*transmissionEfficiency;
    camera.addSkybackgroundAndTransmissionEfficiency(meanBackground, transmissionEfficiency);

}










/**
 * Destructor.
 *
 */
Detector::~Detector()
{
    writeCTIToHDF5();
    if (!constantSkyBackground)
    {
        writeBackgroundMapToHDF5();
    }

    if (includeStraylight)
    {
        hdf5File.writeStraylight(straylightValues);
        delete straylight;
    }

    if (writeBadPixelMap)
    {
        writeBadPixelMapToHDF5();
    }

    if (writeCosmics)
    {
      subFieldCosmics.offsetsDataset.write(subFieldCosmics.offsets.data(), H5::PredType::NATIVE_UINT64);
      smearingCosmics.offsetsDataset.write(smearingCosmics.offsets.data(), H5::PredType::NATIVE_UINT64);
      leftBiasCosmics.offsetsDataset.write(leftBiasCosmics.offsets.data(), H5::PredType::NATIVE_UINT64);
      rightBiasCosmics.offsetsDataset.write(rightBiasCosmics.offsets.data(), H5::PredType::NATIVE_UINT64);

    }

    flushOutput();
    delete frontEndElectronics;
}







/**
 * \brief Update the time dependent parameters of the Detector to their
 *        value at the given time point
 *
 * \param time: current time
 */

void Detector::updateParameters(double time)
{
    if (ccdPosition != "Custom")
        ccdPositions->updateValue(time);
}











/**
 * \brief Configure the Detector object using the ConfigurationParameters
 *
 * \param configParam: Configuration parameters
 */

 void Detector::configure(ConfigurationParameters &configParam)
 {
     // Configuration parameters for the subfield

    subFieldZeroPointRow    = configParam.getInteger("SubField/ZeroPointRow");
    subFieldZeroPointColumn = configParam.getInteger("SubField/ZeroPointColumn");
    numRowsPixelMap         = configParam.getInteger("SubField/NumRows");
    numColumnsPixelMap      = configParam.getInteger("SubField/NumColumns");
    numRowsBiasMap          = configParam.getInteger("SubField/NumBiasPrescanRows");
    numColumnsBiasMap       = configParam.getInteger("SubField/NumBiasPrescanColumns");
    numRowsSmearingMap      = configParam.getInteger("SubField/NumSmearingOverscanRows");

    Log.debug("Detector: Subfield zero point (row, col) = (" + to_string(subFieldZeroPointRow) + ", " + to_string(subFieldZeroPointColumn) + ")");
    Log.debug("Detector: Subfield center point (row, col) = (" + to_string(subFieldZeroPointRow + numRowsPixelMap/2)
                                                               + ", " + to_string(subFieldZeroPointColumn + numColumnsPixelMap/2) + ")");
    Log.debug("Detector: Subfield nr of rows = " + to_string(numRowsPixelMap));
    Log.debug("Detector: Subfield nr of columns = " + to_string(numColumnsPixelMap));

    // No parallel over-scan in case of partial readout

    if(readoutMode == "Partial")
    {
        Log.info("No smearing map for partial readout");
        numRowsSmearingMap = 0;
    }

    // Configuration parameters for the CCD detector
    ccdPosition                  = configParam.getString("CCD/Position");


    isFastCamera          = configParam.getString("Telescope/GroupID") == "Fast";
    includeShield         = configParam.getBoolean("CCDPositions/MetallicShield/IncludeMetallicShield");
    groupByExposure       = configParam.getBoolean("ControlHDF5Content/GroupByExposure");

    if (ccdPosition == "Custom")
    {
        customOriginOffsetX    = configParam.getDouble("CCD/OriginOffsetX");        // [mm]
        customOriginOffsetY    = configParam.getDouble("CCD/OriginOffsetY");        // [mm]
        customOrientationAngle = deg2rad(configParam.getDouble("CCD/Orientation")); // [rad]
        numRows                = configParam.getInteger("CCD/NumRows");             // [pixels]
        numColumns             = configParam.getInteger("CCD/NumColumns");          // [pixels]
        firstRowExposed        = configParam.getInteger("CCD/FirstRowExposed");     // [pixels]
        coveredLeft = 0, coveredRight = 0;                                          // [pixels]
        coveredTop = 0, coveredBottom = 0;                                          // [pixels]

        rotationAnglePsf = customOrientationAngle;  // Angle over which the PSF should be rotated

        if(isFastCamera && includeShield)                            // Include a metallic shield if we have a fast camera
            configureMetallicShield(configParam);
    }
    else
    {
        vector<double> orientation = configParam.getDoubleVector("CCDPositions/Orientation");

        if(configParam.getBoolean("CCDPositions/UsePositionsFromFile"))
        {
            string ccdPositionsInputFile = configParam.getAbsoluteFilename("CCDPositions/PositionsFileName");
            ccdPositions = new Parameter<double, 12>(ccdPositionsInputFile, 1);
        }

        else
        {
            vector<double> originOffsetX = configParam.getDoubleVector("CCDPositions/OriginOffsetX");
            vector<double> originOffsetY = configParam.getDoubleVector("CCDPositions/OriginOffsetY");

            array<double, 12> ccdPositionsArray;

            for (unsigned int ccd = 0; ccd < 4; ccd++)
            {
                ccdPositionsArray[ccd * 3] = originOffsetX[ccd];
                ccdPositionsArray[ccd * 3 + 1] = originOffsetY[ccd];
                ccdPositionsArray[ccd * 3 + 2] = orientation[ccd];
            }


            ccdPositions = new Parameter<double, 12>(ccdPositionsArray);
        }

        int idx = stoi(ccdPosition) - 1;  // Positions are named [1, 2, 3, 4] while the index into vector starts at 0

        numRows               = configParam.getIntegerAt("CCDPositions/NumRows", idx);
        numColumns           = configParam.getIntegerAt("CCDPositions/NumColumns", idx);

        rotationAnglePsf = deg2rad(orientation[idx]);  // Angle over which the PSF should be rotated

        coveredLeft = 0, coveredRight = 0;
        coveredTop = 0, coveredBottom = 0;

        if (isFastCamera)
        {
            firstRowExposed       = configParam.getIntegerAt("CCDPositions/FirstRowForFastCamera", idx);
            if (includeShield)
                configureMetallicShield(configParam);                // Configure metallic shield around CCD
        }
        else
        {
            firstRowExposed       = configParam.getIntegerAt("CCDPositions/FirstRowForNormalCamera", idx);
            includeShield         = false;
        }

    }

    Log.debug("Detector: selected ccdPosition = " + ccdPosition);
    // Log.debug("Detector: CCD originOffsetX, originOffsetY = " + to_string(originOffsetX) + ", " + to_string(originOffsetY) + " mm");
    // Log.debug("Detector: CCD orientationAngle = " + to_string(rad2deg(orientationAngle)) + " deg");
    Log.debug("Detector: CCD numRows, numColumns, firstRow = " + to_string(numRows) + ", " + to_string(numColumns) + ", " + to_string(firstRowExposed));

    pixelSize                           = configParam.getDouble("CCD/PixelSize");
//    quantumEfficiency                   = configParam.getDouble("CCD/QuantumEfficiency/Efficiency");                  // FIXME: No commented out lines of code. To be removed or not?
//    refAngleQE                          = configParam.getDouble("CCD/QuantumEfficiency/RefAngle");
//    relativeRefEfficiencyQE             = configParam.getDouble("CCD/QuantumEfficiency/RelativeRefEfficiency");
    meanQE                              = configParam.getDouble("CCD/QuantumEfficiency/MeanQuantumEfficiency");
    meanAngleDependencyQE               = configParam.getDouble("CCD/QuantumEfficiency/MeanAngleDependency");
//    expectedValueQuantumEfficiency      = configParam.getDouble("CCD/QuantumEfficiency/ExpectedValue");
    includeCosmicsInSubField            = configParam.getBoolean("Sky/IncludeCosmicsInSubField");
    includeCosmicsInSmearingMap         = configParam.getBoolean("Sky/IncludeCosmicsInSmearingMap");
    includeCosmicsInBiasMap             = configParam.getBoolean("Sky/IncludeCosmicsInBiasMap");
    cosmicHitRate                       = configParam.getDouble("Sky/Cosmics/CosmicHitRate");
    cosmicTrailLengthParams             = configParam.getDoubleVector("Sky/Cosmics/TrailLength");
    cosmicIntensityParams               = configParam.getDoubleVector("Sky/Cosmics/Intensity");

    if (cosmicIntensityParams.size() != 3)
    {
        Log.error("Detector::configure(): in input yaml file: Sky/Cosmics/Intensity array does not contain 3 numbers.");
        throw ConfigurationException("Detector: in input yaml file: Sky/Cosmics/Intensity array needs to have 3 numbers. Perhaps you used an old config file?");
    }

    darkCurrent                         = configParam.getDouble("CCD/DarkSignal/DarkCurrent");
    dsnu                                = configParam.getDouble("CCD/DarkSignal/DSNU");
    darkCurrentStability                = configParam.getDouble("CCD/DarkSignal/Stability");
    includeBFE                          = configParam.getBoolean("CCD/IncludeBFE");
    if(includeBFE)
        readBfeCoefficients(configParam.getAbsoluteFilename("CCD/BFE/CoefficientsFileName"));

    fullWellSaturationLimit             = configParam.getLong("CCD/FullWellSaturation");
    digitalSaturationLimit              = configParam.getLong("CCD/DigitalSaturation");
    readoutNoise                        = configParam.getDouble("CCD/ReadoutNoise");
    particulateContaminationEfficiency  = configParam.getDouble("CCD/Contamination/ParticulateContaminationEfficiency");
    molecularContaminationEfficiency    = configParam.getDouble("CCD/Contamination/MolecularContaminationEfficiency");

    refValueGainLeft          = configParam.getDouble("CCD/Gain/RefValueLeft");
    refValueGainRight         = configParam.getDouble("CCD/Gain/RefValueRight");
    gainStability             = configParam.getDouble("CCD/Gain/Stability");
    gainAllowedDifference     = configParam.getDouble("CCD/Gain/AllowedDifference");

    readoutMode               = configParam.getString("CCD/ReadoutMode/ReadoutMode");

    if(readoutMode == "Partial")
    {
        firstRowPartialReadout = configParam.getInteger("CCD/ReadoutMode/Partial/FirstRowReadout");
        numRowsPartialReadout = configParam.getInteger("CCD/ReadoutMode/Partial/NumRowsReadout");
    }
    else if (readoutMode != "Nominal")
    {
        Log.error("Detector::configure(): Unknown readout mode specification in configuration file: "  + readoutMode);
        throw ConfigurationException("Detector: Unknown readout mode specification in configuration file");
    }

    serialTransferTime = configParam.getDouble("CCD/SerialTransferTime") * 1E-9;              // [ns] -> [s]
    parallelTransferTime = configParam.getDouble("CCD/ParallelTransferTime") * 1E-6;          // [µs] -> [s]
    parallelTransferTimeFast = configParam.getDouble("CCD/ParallelTransferTimeFast") * 1E-6;  // [µs] -> [s]


    chargeInjectionLevel = configParam.getDouble("CCD/ChargeInjection/InjectionLevel");
    injectionRowInterval = configParam.getInteger("CCD/ChargeInjection/RowInterval");
    firstInjectedRow = configParam.getInteger("CCD/ChargeInjection/FirstRow");
    if (chargeInjectionLevel < 0 || chargeInjectionLevel > 100.0)
    {
        string message = "Detector:configure(): ChargeInjection/InjectionLevel not between 0 and 100";
        Log.error(message);
        throw ConfigurationException(message);
    }
    if (injectionRowInterval < 1 || injectionRowInterval > numRows)
    {
        string message = "Detector:configure(): ChargeInjection/rowInterval not between 1 and # of CCD rows";
        Log.error(message);
        throw ConfigurationException(message);
    }
    if (firstInjectedRow < 0 || firstInjectedRow > numRows)
    {
        string message = "Detector:configure(): ChargeInjection/FirstRow not between 0 and # of CCD rows";
        Log.error(message);
        throw ConfigurationException(message);
    }

//    polarizationEfficiency          = configParam.getDouble("CCD/Polarization/Efficiency");
//    refAnglePolarization            = configParam.getDouble("CCD/Polarization/RefAngle");
    expectedValuePolarization       = configParam.getDouble("CCD/Polarization/ExpectedValue");

    nominalOperatingTemperature     = configParam.getDouble("CCD/NominalOperatingTemperature");

    includeParticulateContamination = configParam.getBoolean("CCD/IncludeParticulateContamination");
    includeMolecularContamination   = configParam.getBoolean("CCD/IncludeMolecularContamination");
    includeDarkSignal               = configParam.getBoolean("CCD/IncludeDarkSignal");
    includePhotonNoise              = configParam.getBoolean("CCD/IncludePhotonNoise");
    includeReadoutNoise             = configParam.getBoolean("CCD/IncludeReadoutNoise");
    includeCTIeffects               = configParam.getBoolean("CCD/IncludeCTIeffects");
    includeChargeInjection          = configParam.getBoolean("CCD/IncludeChargeInjection");
    includeOpenShutterSmearing      = configParam.getBoolean("CCD/IncludeOpenShutterSmearing");
    includeQuantumEfficiency        = configParam.getBoolean("CCD/IncludeQuantumEfficiency");
    includeRelativeTransmissivity   = configParam.getBoolean("CCD/IncludeRelativeTransmissivity");
    includePolarization             = configParam.getBoolean("CCD/IncludePolarization");
    includeFullWellSaturation       = configParam.getBoolean("CCD/IncludeFullWellSaturation");
    includeDigitalSaturation        = configParam.getBoolean("CCD/IncludeDigitalSaturation");
    includeQuantisation             = configParam.getBoolean("CCD/IncludeQuantisation");
    includeFieldDistortion          = configParam.getBoolean("Camera/IncludeFieldDistortion");
    constantSkyBackground           = configParam.getBoolean("Sky/SkyBackground/UseConstantSkyBackground");
    includeGainNonlinearity         = configParam.getBoolean("CCD/IncludeGainNonlinearity");
    includeStraylight               = configParam.getBoolean("Sky/StrayLight/IncludeStrayLight");
     includeBadPixelMap             = configParam.getBoolean("CCD/BadPixelMap/includeBadPixelMap");

    if (includeRelativeTransmissivity)
    {
        // expectedValueNaturalVignetting      = configParam.getDouble("CCD/Vignetting/NaturalVignetting/ExpectedValue");    # FIXME remove?
        relTransmissivityCoefVector = configParam.getDoubleVector("CCD/RelativeTransmissivity/Coefficients");

        if (relTransmissivityCoefVector.size() != 3)
            {
                string msg = "Detector::configure(): number of coefficients for the relative transmissivity in input yaml file != 3.";
                throw ConfigurationException(msg);
            }

        radiusFOV                           = deg2rad(configParam.getDouble("CCD/RelativeTransmissivity/RadiusFOV"));
        expectedValueRelativeTransmissivity =  configParam.getDouble("CCD/RelativeTransmissivity/ExpectedValue");
    }

    // If a non-linear gain was requested, read the polynomial coefficients. 

    if (includeGainNonlinearity) 
    {
        gainNonlinearityCoefficients = configParam.getDoubleVector("CCD/Gain/Nonlinearity");     
        if (gainNonlinearityCoefficients.size() != 3)
            {
                string msg = "Detector::configure(): number of coefficients for the gain nonlinearity in input yaml file != 3.";
                throw ConfigurationException(msg);
            }
    }
    else 
    {
        gainNonlinearityCoefficients.resize(3);
        std::fill(gainNonlinearityCoefficients.begin(), gainNonlinearityCoefficients.end(), 0.0);
    }

    // The configuration for CTI

    missionDuration = configParam.getDouble("ObservingParameters/MissionDuration") * 31536000.0; // [s]

    CTImodel                   = configParam.getString("CCD/CTI/Model");
    if (CTImodel == "Simple")
    {
        meanCte                = configParam.getDouble("CCD/CTI/Simple/MeanCTE");
    }
    else if (CTImodel == "Short2013")
    {
        beta                    = configParam.getDouble("CCD/CTI/Short2013/Beta");
        temperature             = configParam.getDouble("CCD/CTI/Short2013/Temperature");
        numTrapSpecies          = configParam.getInteger("CCD/CTI/Short2013/NumTrapSpecies");
        trapCaptureCrossSection = configParam.getDoubleVector("CCD/CTI/Short2013/TrapCaptureCrossSection");
        releaseTime             = configParam.getDoubleVector("CCD/CTI/Short2013/ReleaseTime");
        meanTrapDensityBOL      = configParam.getDoubleVector("CCD/CTI/Short2013/TrapDensity/BOL");
        meanTrapDensityEOL      = configParam.getDoubleVector("CCD/CTI/Short2013/TrapDensity/EOL");
        radiationMap.resize(numRowsPixelMap, numColumnsPixelMap);
        radiationMap.fill(1.0);
        radiationSmearingMap.resize(numRowsSmearingMap, numColumnsPixelMap);
        radiationSmearingMap.fill(1.0);
        numberOfOccupiedTrapsPixelMap = arma::zeros<arma::Mat<float>>(numTrapSpecies, numColumnsPixelMap);
        numberOfOccupiedTrapsSmearingMap = arma::zeros<arma::Mat<float>>(numTrapSpecies, numColumnsPixelMap);

    }
    else if (CTImodel == "Short2013FromFile")
    {
        string ctiInputFile = configParam.getAbsoluteFilename("CCD/CTI/Short2013FromFile/CTIFileName");
        readCTIinputFile(ctiInputFile);
        radiationSmearingMap.resize(numRowsSmearingMap, numColumnsPixelMap);
        radiationSmearingMap.fill(1.0);
	    numberOfOccupiedTrapsPixelMap = arma::zeros<arma::Mat<float>>(numTrapSpecies, numColumnsPixelMap);
        numberOfOccupiedTrapsSmearingMap = arma::zeros<arma::Mat<float>>(numTrapSpecies, numColumnsPixelMap);
    }
    else
    {
        Log.error("Detector::configure(): Unkown CTI model specification in configuration file: "  + CTImodel);
        throw ConfigurationException("Detector: Unkown CTI model specification in configuration file");
    }

     // We only need to read in the bad pixel parameter if bad pixels should be included

     if (includeBadPixelMap)
     {
         badPixelParameter = configParam.getDouble("CCD/BadPixelMap/badPixelParameter");
     }

    // Configuration parameters for the HDF5 file output

    writePixelMaps      = configParam.getBoolean("ControlHDF5Content/WritePixelMaps");
    writeBiasMaps       = configParam.getBoolean("ControlHDF5Content/WriteBiasMaps");
    writeSmearingMaps   = configParam.getBoolean("ControlHDF5Content/WriteSmearingMaps");
    writeThroughputMaps = configParam.getBoolean("ControlHDF5Content/WriteThroughputMaps");
    writeCosmics        = configParam.getBoolean("ControlHDF5Content/WriteCosmics");
    writeCTI            = configParam.getBoolean("ControlHDF5Content/WriteCTI");
    writeBackgroundMap =
        configParam.getBoolean("ControlHDF5Content/WriteBackgroundMap") &&
        !constantSkyBackground;

     writeBadPixelMap  = includeBadPixelMap && configParam.getBoolean("ControlHDF5Content/WriteBadPixelMap");

    // Configuration parameters for the noise source random seeds

    readoutNoiseSeed    = configParam.getLong("RandomSeeds/ReadOutNoiseSeed");
    photonNoiseSeed     = configParam.getLong("RandomSeeds/PhotonNoiseSeed");
    badPixelSeed        = configParam.getLong("RandomSeeds/BadPixelSeed");
    darkSignalSeed      = configParam.getLong("RandomSeeds/DarkSignalSeed");

    // Get the sequential number of the very first exposure
    beginExposureNr     = configParam.getInteger("ObservingParameters/BeginExposureNr");
    finalExposureNr     = beginExposureNr + configParam.getInteger("ObservingParameters/NumExposures");

   numEdgePixels = 0;
 }








/**
 * /brief Configures the metallic shield around the CCD
 */
void Detector::configureMetallicShield(ConfigurationParameters &configParam)
{
    int bColumn = configParam.getIntegerAt(
        "CCDPositions/MetallicShield/ShieldColumnCoordinates", 0);
    int bRow    = configParam.getIntegerAt(
        "CCDPositions/MetallicShield/ShieldRowCoordinates", 0);
    int tColumn = configParam.getIntegerAt(
        "CCDPositions/MetallicShield/ShieldColumnCoordinates", 1);
    int tRow =   configParam.getIntegerAt(
        "CCDPositions/MetallicShield/ShieldRowCoordinates", 1);

    // What part of the pixelmap is covered by the metallic shield
    coveredLeft = std::min(int(numColumnsPixelMap),
                           std::max(int(bColumn - subFieldZeroPointColumn), 0));
    coveredRight = std::min(int(numColumnsPixelMap),
                            std::max(int(subFieldZeroPointColumn +
                                         numColumnsPixelMap - tColumn), 0));
    coveredBottom = std::min(int(numRowsPixelMap),
                             std::max(int(bRow - subFieldZeroPointRow), 0));
    coveredTop = std::min( int(numRowsPixelMap),
                           std::max(int(subFieldZeroPointRow + numRowsPixelMap - tRow), 0));
}


/**
 * \brief Read the CTI input HDF5 file. This allows for a spatially varying trap density, and thus CTI.
 */
void Detector::readCTIinputFile(string ctiInputFile)
{

    // Prepare the psfFile by performing some basic checks

    if (!FileUtilities::fileExists(ctiInputFile))
    {
        throw FileException("Detector: trying to load the CTI input HDF5 file (" + ctiInputFile + "), but file doesn't exist.");
    }

    try
    {
         CTIFile.open(ctiInputFile);
    }
    catch (H5::FileIException ex)
    {
        Log.error("H5::FileIException: " + string(ex.getCDetailMsg()));
        throw H5FileException("Detector: Could not open HDF5 file: " + ctiInputFile);
    }

    Log.info("Detector: Opened the CTI input HDF5 file " + ctiInputFile);

    // Read in the CTI info

    vector<double> temporary;
    CTIFile.readArray("/", "beta", temporary);
    beta = temporary[0];
    CTIFile.readArray("/", "temperature", temporary);
    temperature = temporary[0];

    CTIFile.readArray("/", "meanTrapDensityBOL", meanTrapDensityBOL);
    CTIFile.readArray("/", "meanTrapDensityEOL", meanTrapDensityEOL);
    CTIFile.readArray("/", "trapCaptureCrossSection", trapCaptureCrossSection);
    CTIFile.readArray("/", "releaseTime", releaseTime);

    numTrapSpecies = releaseTime.size();

    // Read in the radiation map [protons/sec]. This map has the same size as for the entire CCD.

    arma::Mat<float> map(numRows, numColumns);
    CTIFile.readArray("/", "radiationMap", map);

    // Rescale the radiation map so that it has mean = 1, and only keep the part relevant to subfield we're interested in.

    radiationMap.resize(numRowsPixelMap, numColumnsPixelMap);
    radiationMap = map.submat(subFieldZeroPointRow, subFieldZeroPointColumn, subFieldZeroPointRow+numRowsPixelMap-1, subFieldZeroPointColumn+numColumnsPixelMap-1);
    radiationMap /= arma::mean(arma::mean(map));

    // That's it!

    CTIFile.close();
}







/** \brief: Zeroes the pixel, bias register, and the smearing maps.
 * \pre pixel, bias register, and smearing maps filled with values from previous exposure.
 *
 * \post pixel, bias register, and smearing maps filled with zeroes.
 */

void Detector::reset()
{
    pixelMap.zeros();
    biasMapLeft.zeros();
    biasMapRight.zeros();
    smearingMap.zeros();

    cosmicEntryRowSubfield.clear();
    cosmicEntryColSubfield.clear();
    cosmicsTrailsSubfield.clear();
    cosmicsAnglesSubfield.clear();
    cosmicsIntensitiesSubfield.clear();

    cosmicEntryRowSmearingMap.clear();
    cosmicEntryColSmearingMap.clear();
    cosmicsTrailsSmearingMap.clear();
    cosmicsAnglesSmearingMap.clear();
    cosmicsIntensitiesSmearingMap.clear();

    cosmicEntryRowBiasMapLeft.clear();
    cosmicEntryColBiasMapLeft.clear();
    cosmicsTrailsBiasMapLeft.clear();
    cosmicsAnglesBiasMapLeft.clear();
    cosmicsIntensitiesBiasMapLeft.clear();

    cosmicEntryRowBiasMapRight.clear();
    cosmicEntryColBiasMapRight.clear();
    cosmicsTrailsBiasMapRight.clear();
    cosmicsAnglesBiasMapRight.clear();
    cosmicsIntensitiesBiasMapRight.clear();

    rowsOfCosmicsInSubField.clear();
    columnsOfCosmicsInSubField.clear();
    fluxOfCosmicsInSubField.clear();

    rowsOfCosmicsInSmearingMap.clear();
    columnsOfCosmicsInSmearingMap.clear();
    columnsOfCosmicsInSmearingMap.clear();

    rowsOfCosmicsInBiasMapLeft.clear();
    columnsOfCosmicsInBiasMapLeft.clear();
    fluxOfCosmicsInBiasMapLeft.clear();

    rowsOfCosmicsInBiasMapRight.clear();
    columnsOfCosmicsInBiasMapRight.clear();
    fluxOfCosmicsInBiasMapRight.clear();
}









/**
 * \brief: Take an exposure with the detector starting at the given time.
 *         The light is integrated during the given exposure time, during which
 *         the detector experiences the effects of jitter and thermo-elastic telescope
 *         drift. The background is assumed uniform for the whole subfield.
 *         Afterwards, the collected light is read out, convolving the image with the
 *         point spread function and adding various noise effects.
 *
 * \param exposureNr:   sequential number of the exposure
 * \param startTime:    Starting time of the exposure [s].
 * \param exposureTime: Duration of the exposure [s].
 *
 * \return endTime:     Time after the exposure (startTime + exposureTime + readoutTime)
 *
 * \pre Sub-pixel, pixel, bias register, and smearing map filled with values from previous exposure.
 *
 * \post Pixel unit in the pixel, bias register, and smearing maps: [ADU]
 */

double Detector::takeExposure(int exposureNr, double startTime, double exposureTime)
{
    // Advance the internal clock until the given start time
    internalTime = startTime;

    // Clear all arrays

    Log.debug("Detector: resetting subfield array for new exposure.");
    reset();

    // Integration of point sources and background, taking into account jitter + drift.
    if (!constantSkyBackground && (exposureNr == beginExposureNr))
    {
        fillBackgroundMap(camera, startTime, exposureTime);
    }

    Log.info("Detector: Integrating light for exposure " + to_string(exposureNr) + " with exposure time = " + to_string(exposureTime));

    integrateLight(exposureNr, startTime, exposureTime);


    // If this is the first exposure, we should initialize the number of occupied traps.
    // This can only be done after the detector has been exposed to the skybackground.
    // => Check if CTI is included && We use the Short2013 model
    //
    // JDR[08-01-2026]: I commented out the following if-statement because it causes the first exposure to show a different CTI trail 
    //                  compared to the other exposures. Cf Issue #989. Inside the applyShort2013CTI() function, I implemented the traps
    //                  to be all occupied at the beginning of the readout.

    // if (exposureNr == beginExposureNr) {
    //     if (includeCTIeffects && (CTImodel == "Short2013" || CTImodel == "Short2013FromFile")) {
    //         setInitialNumberOfOccupiedTraps(numberOfOccupiedTrapsPixelMap);
    //     }
    // }

    // Include noise effects like readout noise, photon noise, full well saturation, etc.
    // Note: readOut() needs the exposure time to compute the open shutter smearing.

    Log.info("Detector: adding noise effects to exposure " + to_string(exposureNr));

    readOut(exposureTime);

    // Write the CCD subfield, the bias map, and the smearing map to the HDF5 file

    Log.debug("Detector: writing PixelMap, smearing map, bias map and throughputMap #" + to_string(exposureNr) + " to HDF5 file.");


    writePixelMapsToHDF5(exposureNr);

    // Write the cosmic hits to the HDF5 file

    Log.debug("Detector: writing Cosmics of the PixelMap, smearing map, bias map #" + to_string(exposureNr) + " to HDF5 file.");

    if (writeCosmics)
    {
      writeCosmicHitsToHDF5(exposureNr);
    }

    // Advance the internal clock

    internalTime += exposureTime + readoutTimeBeforeNextExposure;

    return internalTime;
}











/**
 *\brief Generate throughput map, containing for each sub-field pixel the combined throughput efficiency
 *       of vignetting, polarisation, particulate & molecular contamination, and quantum efficiency.  Each
 *       array value is a value between 0 and 1.
 *
 * \details Because of vignetting, the stars at the edge of the FOV look dimmer than the stars close
 *          to the optical axis. If the incoming flux before vignetting at pixel (i,j) is F(i,j),
 *          then the flux after vignetting taken into account is F(i,j) * vignettingMap(i,j).
 *          Because of contamination (both particulate and molecular) the throughput efficiency
 *          decreases over the entire FOV by the same factor.
 *
 * \note    The throughput map is written to the HDF5 map.
 */

void Detector::generateThroughputMap()
{
    Log.info("Detector: generating throughput map.");

    throughputMap.fill(1.0);

    if(includeRelativeTransmissivity  && includeOpenShutterSmearing)
        mechanicalVignettingMask.fill(1);

    double xFPmmDistorted, yFPmmDistorted;             // Distorted focal plan coordinates   [mm]
    double xFPmmUndistorted, yFPmmUndistorted;         // Undistorted focal plan coordinates [mm]
    double angle;                                      // Gnomonic radial distance from the optical axis [rad]
    double relativeTransmissivityVariation;


//    const double refAnglePolarizationRadians = deg2rad(refAnglePolarization);       // Reference angle for the polarisation efficiency [radians]
//    const double acosPolarizationEfficiency = acos(polarizationEfficiency);

//    const double refAngleQuantumEfficiencyRadians = deg2rad(refAngleQE);     // Reference angle for the quantum efficiency [radians]
//    const double acosQuantumEfficiency = acos(relativeRefEfficiencyQE);        // Relative efficiency due to the angle dependency of the QE at the reference angle

    if (includeRelativeTransmissivity || includePolarization || includeQuantumEfficiency)
    {
        // Loop over all pixels in the pixel map

        for (unsigned int row = 0; row < numRowsPixelMap; row++)
        {
            for (unsigned int column = 0; column < numColumnsPixelMap; column++)
            {
                // Distorted pixel coordinates (in the detector) -> distorted focal-plane coordinates

                tie(xFPmmDistorted, yFPmmDistorted) = pixelToFocalPlaneCoordinates(row + subFieldZeroPointRow, column + subFieldZeroPointColumn);

                // Convert from distorted to undistorted focal plane coordinates (Cf GitHub issue #716)
                if (includeFieldDistortion)
                {
                    tie(xFPmmUndistorted, yFPmmUndistorted) =  camera.distortedToUndistortedFocalPlaneCoordinates(xFPmmDistorted, yFPmmDistorted);
                }
                else
                {
                    xFPmmUndistorted = xFPmmDistorted;
                    yFPmmUndistorted = yFPmmDistorted;
                }

                // Angular distance [radians] of the pixel from the optical axis

                angle = camera.getGnomonicRadialDistanceFromOpticalAxis(xFPmmUndistorted, yFPmmUndistorted);  // [radians]

                if (includeRelativeTransmissivity)
                {
                    if (angle >= radiusFOV)
                    {
                        throughputMap(row, column) = 0.0;

                        if (includeOpenShutterSmearing)
                            mechanicalVignettingMask(row, column) = 0;
                    }

                    else
                    {
                        angle = rad2deg(angle); // [degrees]
                        relativeTransmissivityVariation = (  relTransmissivityCoefVector[0] * pow(angle, 2)
                                                           + relTransmissivityCoefVector[1] * pow(angle, 4)
                                                           + relTransmissivityCoefVector[2] * pow(angle, 6)) / 100.;

                        throughputMap(row, column) *= (1 - relativeTransmissivityVariation);
                    }
                }

                // Polarisation (Eq. 4-11 in PLATO-DLR-PL-RP-001)

                // NOTE: the polarization is angle dependent, but since no info on this dependency is currently available,
                //       we assume fow now it is fixed over the entire FOV.

                if (includePolarization)
                    throughputMap(row, column) *= expectedValuePolarization; //cos(angle / refAnglePolarizationRadians * acosPolarizationEfficiency);

                // Quantum efficiency (Eq. 4-12 in PLATO-DLR-PL-RP-001)
                // Pixel units before: [photons]
                // Pixel units after: [electrons]

                // NOTE: the QE is angle dependent, but since no info on this dependency is currently available,
                //       we assume for now it is fixed over the entire FOV.

                if (includeQuantumEfficiency)
                    throughputMap(row, column) *= meanQE * meanAngleDependencyQE; //(meanQE * cos(angle / refAngleQuantumEfficiencyRadians * acosQuantumEfficiency));
            }
        }
    }

    // Particulate contamination (Sect. 4.2.4.3 in PLATO-DLR-PL-RP-001)

    if (includeParticulateContamination)
    {
        throughputMap *= particulateContaminationEfficiency;
    }

    // Molecular contamination (Sect. 4.2.4.4 in PLATO-DLR-PL-RP-001)

    if (includeMolecularContamination)
    {
        throughputMap *= molecularContaminationEfficiency;
    }
}












/**
 * Checks whether the gain values for left- and right-hand side of the detector are not
 * too far apart, according to the specified allowed difference.  In case the gain values
 * are too far apart, a warning message is shown to the user.
 */
void Detector::checkGain()
{
    double allowedDifference = min(refValueGainLeft, refValueGainRight) * gainAllowedDifference / 100.0;

    if(abs(refValueGainLeft - refValueGainRight) > allowedDifference)
    {
        Log.warning("Detector: Difference in gain between the left- and right-hand side of the detector too large.");
    }
}










/**
 *\brief Reads the coefficients a^X_ij for the brighter-fatter effect,
 *       from the file with the given name..
 *
 *       These parameters will be the same for all pixels (0, 0) and will only
 *       be available for the pixels (i, j) that are within a window centred
 *       at pixel (0, 0).  This can be done because the influence of pixels (i, j)
 *       rapidly decreases with distance from pixel (0, 0).
 *
 * \param filename: Name of the HDF5 file from which to read the BFE coefficients.
 */
void Detector::readBfeCoefficients(string filename)
{
    Log.info("Detector: reading BFE coefficients.");

    // Basic checks on the file with BFE coefficients

    HDF5File coefficientsFile;

    if(!FileUtilities::fileExists(filename))

        throw FileException("Detector: trying to load the BFE coefficients HDF5 file (" + filename + "), but file doesn't exist.");

    try
    {
         coefficientsFile.open(filename);
    }
    catch(H5::FileIException ex)
    {
        Log.error("H5::FileIException: " + string(ex.getCDetailMsg()));
        throw H5FileException("Detector: Could not open HDF5 file: " + filename);
    }

    // Prepare to read X and a_ij^X from the file

    unsigned constexpr int numNeighbors = 4;    // Number of neighbours X
    bfeRange = coefficientsFile.readIntegerGroupAttribute("Neighbors", "range"); // Range over which to consider source charges Q_ij
    int windowDim = 2 * bfeRange + 1;   // Window in which to consider source charges Q_ij

    string neighborNames[numNeighbors] = {"East", "North", "West", "South"};

    bfeCoefficients.zeros(windowDim, windowDim, numNeighbors);  // (row, column, X)

    string neighborName;
    arma::fmat slice = arma::fmat(windowDim, windowDim, arma::fill::zeros); // Temporary storage of the BFE coefficients for given neighbour

    // Loop over all neighbours (groups)

    for(int index = 0; index < numNeighbors; index++)
    {
        neighborName = neighborNames[index];

        if(!coefficientsFile.hasGroup("Neighbors/" + neighborName))
            throw H5FileException("Detector: The HDF5 file (" + filename + ") doesn't contain the expected group \"" + neighborName + "\".");

        if(!coefficientsFile.hasDataset("Neighbors/" + neighborName, "aX"))
            throw H5FileException("Detector: The HDF5 file (" + filename + ") doesn't contain the expected dataset aX in group \"" + neighborName + "\".");

        // Read the BFE coefficients for the current neighbour

        coefficientsFile.readArray("/Neighbors/" + neighborName, "aX", slice);
        bfeCoefficients.slice(index) = slice;

        bfeNeighbors[index][0] = coefficientsFile.readIntegerGroupAttribute("Neighbors/" + neighborName, "row");
        bfeNeighbors[index][1] = coefficientsFile.readIntegerGroupAttribute("Neighbors/" + neighborName, "column");
    }

    Log.debug("Detector: closing HDF5 file " + filename);
    coefficientsFile.close();
}













/**
 * \brief Verify if a point with given planar focal plane coordinates is in the subfield
 *
 * \param xFP    Planar focal plane x-coordinate in the FP reference frame [mm]
 * \param yFP    Planar focal plane y-coordinate in the FP reference frame [mm]
 *
 * \return true if the point is in the subfield on the CCD, false otherwise.
 */

bool Detector::isInSubfield(double xFP, double yFP)
{
    double row, column;
    tie(row, column) = focalPlaneToPixelCoordinates(xFP, yFP);

    // Check wether these pixel coordinates falls on the subfield

    return    (column >= subFieldZeroPointColumn) && (column < subFieldZeroPointColumn + numColumnsPixelMap)
           && (row    >= subFieldZeroPointRow)    && (row < subFieldZeroPointRow + numRowsPixelMap);
}

















 /**
 * \brief   Check whether the given (row, column) indices are within the array range of the exposed part of the pixel map.
 *
 * \details  The input parameters row & column come from a coordinate transformation
 *           in the focal plane, and as a result are not necessarily integers. For this
 *           function it's not necessary to round them to the nearest integer.
 *
 * \param  row:    Row index. NOT a coordinate in the CCD frame, but in the subfield frame.    [pixel].
 * \param  column: Column index. NOT a coordinate in the CCD frame, but in the subfield frame. [pixel].
 *
 * \return  True if the given (row, column) coordinates are in the pixel map; false otherwise.
 */

bool Detector::isInPixelMap(double row, double column)
{
    return (column >= coveredLeft) && (row >= coveredBottom) && (column < numColumnsPixelMap - coveredRight) && (row < numRowsPixelMap - coveredTop);
}











/*
 * \brief Apply throughput efficiency. This is the combined effect of:
 *          - vignetting (brightness attenuation towards the edges of the FOV);
 *          - particulate contamination;
 *          - molecular contamination;
 *          - quantum efficiency.
 */


void Detector::applyThroughputEfficiency()
{

    // Generate the throughput map which includes vignetting, polarisation,
    // particulate & molecular contamination, and quantum efficiency. The
    // throughput may be time dependent and therefore needs to regenerated
    // every exposure. We assume that during the jittering in Camera::exposeDetector()
    // the time dependent parameters have been updated so that their current
    // value corresponds to the very last jitter step, which are the same values
    // that we need here.
    // We also assume that the throughput does not change significantly _within_
    // each exposure, so that we don't need to include the throughput generation
    // in the jitter loop.

    generateThroughputMap();

    // Element-wise multiplication with the throughput map
    // Beware of Armadillo's quirky notation...

    pixelMap = pixelMap % throughputMap;

}





/**
 *\brief Adds the dark signal to the pixel map.  This follows a normal distribution,
 *       centered around the dark current and with the DSNU (percentage of the dark signal)
 *       as standard deviation.  The noise of the simulated dark signal is of the order
 *       of the sqrt of the dark signal (simulated with a normal distribution).
 *
 * \param exposureTime: Exposure time [s].
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Bias register and smearing maps filled with zeroes.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map.
 */
void Detector::addDarkSignal(float exposureTime)
{
    double darkSignal;

    // Add dark signal to the pixel map

    // When is dark current accumulated for the pixel map?
    //     -  exposure + readout

    const double darkCurrentOverDeltaTemp = darkCurrentStability * (getTemperature() - nominalOperatingTemperature);
    double darkSignalRef = (darkCurrent + darkCurrentOverDeltaTemp) * (exposureTime + readoutTimeBeforeNextExposure + readoutTimeDuringNextExposure);
    darkSignalDistribution = normal_distribution<double>(darkSignalRef, darkSignalRef * dsnu / 100.0);


    for(unsigned int row = 0; row < numRowsPixelMap; row++)
    {
        for(unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            // Take DSNU into account

            darkSignal = darkSignalDistribution(darkSignalGenerator);

            // Add the noise

            darkNoiseDistribution = normal_distribution<double>(darkSignal, sqrt(darkSignal));
            darkSignal = darkNoiseDistribution(darkNoiseGenerator);


            pixelMap(row, column) += darkSignal;
        }
    }

    // Add dark signal to the smearing map

    // When is dark current accumulated for the smearing map?
    //     - normal camera, nominal mode: whole readout
    //  - fast camera, nominal mode: readout during the next exposure
    //  - partial readout: no smearing map

    darkSignalRef = darkCurrent
            * (isFastCamera ? readoutTimeDuringNextExposure : readoutTimeBeforeNextExposure);
    darkSignalDistribution = normal_distribution<double>(darkSignalRef, darkSignalRef * dsnu / 100.0);

    for(unsigned int row = 0; row < numRowsSmearingMap; row++)
    {
        for(unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            // Take DSNU into account

            darkSignal = darkSignalDistribution(darkSignalGenerator);

            // Add the noise

            darkNoiseDistribution = normal_distribution<double>(darkSignal, sqrt(darkSignal));
            darkSignal = darkNoiseDistribution(darkNoiseGenerator);

            smearingMap(row, column) += darkSignal;
        }
    }
}







/**
 *
 * \brief: Generate the bad pixel map for the entire CCD
 *
 * \note: The bad pixels follow a Bernoulli distribution.
 *  
 */ 
void Detector::generateBadPixelMap()
{
    for (unsigned int row=0; row < numRows; row++)
    {
        for (unsigned int col = 0; col < numColumns; col++)
        {
            badPixelMap(row, col) = badPixelDistribution(badPixelGenerator);
        }
    }
}







/**
 *
 * /brief: Include the bad pixel in the subfield
 *
 * /note: We select the bad pixel from the subfield from the bad pixel ccd map.
 *
 */ 
void Detector::applyBadPixelMap()
{
    int subFieldEndPointRow    = subFieldZeroPointRow + numRowsPixelMap - 1;
    int subFieldEndPointColumn = subFieldZeroPointColumn+numColumnsPixelMap-1;

    pixelMap = pixelMap % badPixelMap.submat(subFieldZeroPointRow,
                                             subFieldZeroPointColumn,
                                             subFieldEndPointRow, subFieldEndPointColumn);
}




/**
 * \brief: Reads out the detector and apply the following effects:
 *          - Cosmics
 *          - BFE
 *          - CTE
 *          - full-well saturation (i.e. blooming)
 *          - readout noise
 *          - F-FEE over-/under shoot
 *          - quantisation:
 *              - gain
 *              - electronic offset (i.e. bias)
 *              - digital saturation
 *
 * \param exposureTime: Exposure time [s].
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre Bias register map filled with zeroes.
 *
 * \post Pixel unit in the pixel, bias register, and smearing maps: [ADU].
 */
void Detector::readOut(float exposureTime)
{

    // Add cosmic hits [electrons -> electrons]

    if(includeCosmicsInSubField | includeCosmicsInBiasMap | includeCosmicsInSmearingMap)
    {
        Log.debug("Detector: including cosmic hits.");
        addCosmics(exposureTime);
    }
    else
    {
        Log.debug("Detector: no cosmic hits included.");
    }
        
    // Apply full-well saturation (blooming) [electrons -> electrons]

    if (includeFullWellSaturation)
    {
        Log.debug("Detector: applying full well saturation.");
        applyFullWellSaturation();
    }
    else
    {
        Log.debug("Detector: no full well saturation applied.");
    }

        // Brighter-Fatter effect [electrons -> electrons]

    if (includeBFE)
    {
        Log.debug("Detector: adding Brighter-Fatter effect");

        applyBFE();
    }
    else
    {
        Log.debug("Detector: no Brighter-Fatter effect added");
    }

    // Apply Charge Transfer Inefficiency (CTI) [electrons -> electrons]

    if (includeCTIeffects)
    {
        Log.debug("Detector: applying charge transfer inefficiency.");
        applyCTI();
    }
    else
    {
        Log.debug("Detector: no charge transfer inefficiency applied.");
    }

    // Add the readout noise [electrons -> electrons]

    if (includeReadoutNoise)
    {
        Log.debug("Detector: adding readout noise of CCD and FEE.");
        addReadoutNoise();
    }
    else
    {
        Log.debug("Detector: no readout noise added.");
    }

    // Apply the F-FEE over-/undershoot [electrons -> electrons] 

    if(isFastCamera && frontEndElectronics->getIncludeOverAndUnderShoot())
    {
        Log.debug("Detector: adding (F-)FEE over-/undershoot");
        applyOverAndUnderShoot();
    }
    else{
        Log.debug("Detector: (F-)FEE over-/undershoot not applied: " + to_string(isFastCamera) + " " + to_string(frontEndElectronics->getIncludeOverAndUnderShoot()));
    }
    
    // Apply quantisation [electrons -> ADU]
    // This consists of:
    // - applying CCD non-linearity
    // - applying FEE and CCD gain (converting from electrons to ADU)
    // - adding the electronic offset
    // - applying digital saturation

    if(includeQuantisation)
    {
        Log.debug("Detector: applying quantisation.");
        applyQuantisation();
    }
    else
    {
        Log.debug("Detector: no quantisation applied.");
    }

    // Apply the bad pixels

    if (includeBadPixelMap)
    {
        Log.debug("Detector: applying bad pixel map.");
        applyBadPixelMap();
    }
    else
    {
        Log.debug("Detector: no bad pixel map applied.");
    }

}









/**
 * \brief Adds the Brighter-Fatter Effect (BFE) to the pixel map, following the method
 *        proposed by Guyonnet et al. 2015 (https://arxiv.org/abs/1501.01577).  This 
 *        implementation is valid only for shallow PSFs.
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre No bias register or smearing maps.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post No bias register or smearing maps.
 */
void Detector::applyBFE()
{

    // For each pixel (0, 0), we only account for the influence of pixels (i, j)
    // that are within the given range (to evaluate Eq. (11) in Guyonnet et al.)

    arma::fmat deltaQ = arma::fmat(numRowsPixelMap, numColumnsPixelMap, arma::fill::zeros); // δQ_00 for whole sub-field

    double charge00, chargeX;           // Q_00, Q_X
    int neighborRow, neighborColumn;    // X

    // Loop over all pixels (stay away from the edges)
    // -> current pixel correponds to (0, 0) in Guyonnet et al. 2015

    for(int row = bfeRange; row < numRowsPixelMap - bfeRange; row++)
    {
        for(int column = bfeRange; column < numColumnsPixelMap - bfeRange; column++)
        {
            charge00 = pixelMap(row, column); // Charge in the current pixel -> Q_00

            // Loop over all neighbours X

            for(int neighbor = 0; neighbor < 4; neighbor++)
            {
                neighborRow = bfeNeighbors[neighbor][0];        // Row offset w.r.t. current pixel [pixels]
                neighborColumn = bfeNeighbors[neighbor][1];     // Column offset w.r.t. current pixel [pixels]

                chargeX = pixelMap(row + neighborRow, column + neighborColumn);     // Charge includeMechanicalVignetting neighbor X -> Q_X

                 // Eq. (11) in Guyonnet et al. 2015

                deltaQ(row, column) += arma::accu(bfeCoefficients.slice(neighbor) 
                                                  % pixelMap(arma::span(row - bfeRange, row + bfeRange), 
                                                             arma::span(column - bfeRange, column + bfeRange))) 
                                       * 0.25 * (charge00 + chargeX);
            }
        }
    }

    pixelMap += deltaQ;
}










/**
 * \brief: Add photon noise (i.e. shot noise) to the pixel and smearing maps.
 *         It follows a Poisson distribution and each pixel is treated
 *         independently of the other pixels.
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre No bias register or smearing maps.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map.
 */
void Detector::addPhotonNoise()
{
    // Add photon noise to the pixel map

    Log.debug("Detector: adding photon noise to pixel map");

    for (unsigned int row = 0; row < numRowsPixelMap; row++)
    {
        for (unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            photonNoiseDistribution = poisson_distribution<long>(pixelMap(row, column));
            pixelMap(row, column) = photonNoiseDistribution(photonNoiseGenerator);
        }
    }

    // Add photon noise to the smearing map

    Log.debug("Detector: adding photon noise to smearing map");

    for (unsigned int row = 0; row < numRowsSmearingMap; row++)
    {
        for (unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            photonNoiseDistribution = poisson_distribution<long>(smearingMap(row, column));
            smearingMap(row, column) = photonNoiseDistribution(photonNoiseGenerator);
        }
    }
}








/**
 * \brief: Add cosmic hits to the pixel, bias register, and smearing map.
 *         - The number of cosmic hits is determined by a random sample from a Poisson
 *           distribution with the configured mean cosmic hit rate, exposure time,
 *           size of the map (number of rows and columns [pixels]), and pixel size.
 *         - The entry points are uniformly distributed over the maps.
 *         - The entry angles are uniformly distributed over the [0, 2π] interval.
 *         - The length of the trails is uniformly distributed over the given interval.
 *         - The total number of electrons in the trail is uniformly distributed over
 *           the given interval.
 *
 * This function is a wrapper function for Detector::addCosmics(exposureTime, map, numRows, numColumns)
 * that does the actual work but which is not called directly.
 *
 * \param exposureTime: Exposure time [s].
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post Pixel unit in the bias register map: [electrons].
 */
void Detector::addCosmics(float exposureTime)
{
    // Initialize the (class-variable) distributions of the cosmic intensity, trail length, etc.
    // For the Intensity distribution, the parameters are the location, scale>0, and shape.

    cosmicHitRateDistribution     = poisson_distribution<long>(cosmicHitRate);                                                   // [hits/cm^2/s]
    cosmicEntryAngleDistribution  = uniform_real_distribution<double>(0, 2 * PI);                                                // [radians]
    cosmicTrailLengthDistribution = uniform_real_distribution<double>(cosmicTrailLengthParams[0], cosmicTrailLengthParams[1]);   // [pixels]
    cosmicIntensityDistribution   = skew_normal_distribution(cosmicIntensityParams[0], cosmicIntensityParams[1], cosmicIntensityParams[2]); // [e-/hit]

    // Cosmics in the subfield

    if (includeCosmicsInSubField)
    {
        // auto& exposedPixelMap = pixelMap.submat(coveredBottom, coveredLeft, numRowsPixelMap - coveredTop - 1, numColumnsPixelMap - coveredRight - 1);
        Log.debug("Detector: adding cosmic hits to the sub-field");
        addCosmics(exposureTime + readoutTimeBeforeNextExposure + readoutTimeDuringNextExposure, pixelMap, rowsOfCosmicsInSubField,
                   columnsOfCosmicsInSubField, fluxOfCosmicsInSubField, numRowsPixelMap, numColumnsPixelMap, "image area");
    }

    // Cosmics in the over-scan
    cosmicEntryColumnDistribution = uniform_real_distribution<double>(0, numColumnsPixelMap - 1);                                // [pixels]
    if (includeCosmicsInSmearingMap)
    {
        Log.debug("Detector: adding cosmic hits to smearing map");

        if(isFastCamera)
        {
            addCosmics(readoutTimeDuringNextExposure, smearingMap, rowsOfCosmicsInSmearingMap, columnsOfCosmicsInSmearingMap,
                       fluxOfCosmicsInSmearingMap,numRowsSmearingMap, numColumnsPixelMap, "smearing map");
        }
        else
        {
            addCosmics(readoutTimeBeforeNextExposure, smearingMap, rowsOfCosmicsInSmearingMap, columnsOfCosmicsInSmearingMap,
                       fluxOfCosmicsInSmearingMap, numRowsSmearingMap, numColumnsPixelMap, "smearing map");
        }
    }

    // Cosmics in the pre-scan
    // This is a special case because the rows of the prescan are all virtual, so only hot pixels and no trails.
    // The following is only approximative.

    if (includeCosmicsInBiasMap)
    {
        Log.debug("Detector: adding cosmic hits to bias map");
        cosmicTrailLengthDistribution = uniform_real_distribution<double>(0.0, 1.e-6);    // Only hot pixels, no trails
        const double biasMapRowLifeTime = (numColumns / 2 + numColumnsBiasMap) * serialTransferTime + parallelTransferTime;
        addCosmics(biasMapRowLifeTime, biasMapLeft, rowsOfCosmicsInBiasMapLeft, columnsOfCosmicsInBiasMapLeft, fluxOfCosmicsInBiasMapLeft,
                   numRowsBiasMap, numColumnsBiasMap, "bias map (left half)");
        addCosmics(biasMapRowLifeTime, biasMapRight, rowsOfCosmicsInBiasMapRight, columnsOfCosmicsInBiasMapRight,
                   fluxOfCosmicsInBiasMapRight, numRowsBiasMap, numColumnsBiasMap, "bias map (right half)");
    }




}











/**
 * \brief: Add cosmic hits to the given map.
 *         - The number of cosmic hits is determined by a random sample from a Poisson
 *           distribution with the configured mean cosmic hit rate, exposure time,
 *           size of the map (number of rows and columns [pixels]), and pixel size.
 *         - The entry points are uniformly distributed over the maps.
 *         - The entry angles are uniformly distributed over the [0, 2π] interval.
 *         - The length of the trails is uniformly distributed over the given interval.
 *         - The total number of electrons in the trail is uniformly distributed over
 *           the given interval.
 *
 * This function is not called directly in Detector, but only through the method
 *      Detector::addCosmics(exposureTime)
 *
 * \param exposureTime:        amount of time exposed to cosmic particle influx [s].
 * \param map:                 map affected by cosmics [e-].  Either the pixel, bias register, or smearing map.
 * \param rowsOfCosmicsMap:    a vector that stores the rows where the cosmics hit.
 * \param columnsOfCosmicsMap: a vector that stores the column where the cosmics hit.
 * \param fluxOfCosmicsMap:    a vector that stores the flux of the cosmics.
 * \param numRows:             number of rows in the map [pixels].
 * \param numColumns:          number of columns in the map [pixels].
 * \param area:                name of the area to which the cosmics are added ("image area", "smearing map", "bias map").
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post Pixel unit in the bias register map: [electrons].
 */
void Detector::addCosmics(float exposureTime, arma::Mat<float> &map, vector<unsigned int> &rowsOfCosmicsMap, vector<unsigned int> &columnsOfCosmicsMap, vector<double> &fluxOfCosmicsMap, int numRows, int numColumns, string area)
{


    // Characteristics of an individual trail

    double entryRow, entryColumn, entryAngle, trailLength, intensity, sigma;

    // Each trail consists of 200 "trail points"
    /// (equally spaced points over the trail length)

    int numTrailPoints = 200;
    arma::vec trailRows, trailColumns, trailWeights;    // All trail points: row, column, and weight
    int trailRow, trailColumn;                          // Individual trail point: row and column



    int minRow = 0;
    int minCol = 0;

    int maxRow = numRows;
    int maxCol = numColumns;

    // If we we have a "image area" we need to make sure the entry Row and Columns only fall on the exposed part of the subfield

    if (area == "image area")
    {
        minRow = coveredBottom;
        minCol = coveredLeft;
        maxCol = numColumns - coveredRight;
        maxRow = numRows - coveredTop;

        // If subfield is completly covered, we return the function without adding cosmics to the subfield
        if (minRow == maxCol || minCol == maxCol)
            return;

        cosmicEntryColumnDistribution = uniform_real_distribution<double>(minCol, maxCol - 1);      // [pixels]
    }

    cosmicEntryRowDistribution    = uniform_real_distribution<double>(minRow, maxRow-1);   // different for each subfield vs smearing map vs bias map

    // Number of cosmic hits
    // - cosmic hit rate [events / cm^2 / s]
    // - exposure time [s]
    // - dimensions [pixels] -> [micron] -> [cm]
    //
    // To make sure the number of cosmics is as expected when considering a large number
    // of exposures, we have to account for the decimal part of the number of cosmic hits
    // per exposure too, instead of rounding down this value. E.g. if you have 1.5 cosmics
    // per exposure, you want 1500 cosmics for 1000 exposures, instead of 1000 cosmics.
    // The solution is to add as many cosmics as denoted by the integer part, and add one
    // extra cosmic, with a chance equal to the decimal part.  This is handled by the
    // uniform random distribution in [0,1]:
    //     - random number < decimal part => add one extra cosmic
    //  - random number >= decimal part => don't add an extra cosmic

    double numCosmicHitsAsDouble = cosmicHitRateDistribution(
            cosmicHitRateGenerator) * exposureTime
            * (numRows * pixelSize / 10000.0)
            * (numColumns * pixelSize / 10000.0);
    double decimalPartNumCosmicHits = numCosmicHitsAsDouble - (int) numCosmicHitsAsDouble;

    // Round down the number of cosmic hits to an integer, possibly zero.

    int numCosmicHits = (int) numCosmicHitsAsDouble;

      // Add 1 cosmic with a chance equal to the decimal part.

    if (decimalNumCosmicHitsDistribution(decimalNumCosmicHitsGenerator) < decimalPartNumCosmicHits)
    {
        numCosmicHits += 1;
    }

    Log.debug("Detector: number of cosmic hits for the " + area + ": "  + to_string(numCosmicHits));
    if (numCosmicHits == 0) return;

    double meanEntryAngle = 0.0;
    double meanTrailLength = 0.0;
    double meanIntensity = 0.0;



    for (unsigned int cosmicHit = 0; cosmicHit < numCosmicHits; cosmicHit++)
    {
        entryRow    = int(floor(cosmicEntryRowDistribution(cosmicEntryRowGenerator)));        // Entry row [pixels] (uniform distribution over the rows of the sub-fields)
        entryColumn = int(floor(cosmicEntryColumnDistribution(cosmicEntryColumnGenerator)));  // Entry column [pixels] (uniform distribution over the columns of the sub-field)
        entryAngle  = cosmicEntryAngleDistribution(cosmicEntryAngleGenerator);      // Entry angle [radians] (uniform distribution between 0 and 2π)
        trailLength = cosmicTrailLengthDistribution(cosmicTrailLengthGenerator);    // Trail length [pixels] (uniform distribution over interval)
        intensity   = cosmicIntensityDistribution(cosmicIntensityGenerator);        // Number of e- in cosmic hit [e-] (skew-normal distribution over interval)

        // Save the cosmics generated numbers above, so that we can persist them in the HDF5 file

        if (area == "image area")
        {
            cosmicEntryRowSubfield.push_back(entryRow);
            cosmicEntryColSubfield.push_back(entryColumn);
            cosmicsTrailsSubfield.push_back(trailLength);
            cosmicsAnglesSubfield.push_back(entryAngle);
            cosmicsIntensitiesSubfield.push_back(intensity);
        }
        else if (area == "smearing map")
        {
            cosmicEntryRowSmearingMap.push_back(entryRow);
            cosmicEntryColSmearingMap.push_back(entryColumn);
            cosmicsTrailsSmearingMap.push_back(trailLength);
            cosmicsAnglesSmearingMap.push_back(entryAngle);
            cosmicsIntensitiesSmearingMap.push_back(intensity);
        }
        else if (area == "bias map (left half)")
        {
            cosmicEntryRowBiasMapLeft.push_back(entryRow);
            cosmicEntryColBiasMapLeft.push_back(entryColumn);
            cosmicsTrailsBiasMapLeft.push_back(trailLength);
            cosmicsAnglesBiasMapLeft.push_back(entryAngle);
            cosmicsIntensitiesBiasMapLeft.push_back(intensity);
        }
        else if (area == "bias map (right half)")
        {
            cosmicEntryRowBiasMapRight.push_back(entryRow);
            cosmicEntryColBiasMapRight.push_back(entryColumn);
            cosmicsTrailsBiasMapRight.push_back(trailLength);
            cosmicsAnglesBiasMapRight.push_back(entryAngle);
            cosmicsIntensitiesBiasMapRight.push_back(intensity);
        }
        else
        {
            Log.warning("Detector: addCosmics(): unknow `area`");
        }


        // Compute the mean of the sample distribution to save in the log file

        meanEntryAngle += entryAngle;
        meanTrailLength += trailLength;
        meanIntensity += intensity;

        double trailStep = trailLength / numTrailPoints;                            // Distance between two "trail points" (for the current trail)

        trailWeights = arma::linspace(0, trailLength, numTrailPoints);
        trailRows = entryRow + trailWeights * sin(entryAngle);
        trailColumns = entryColumn + trailWeights* cos(entryAngle);

        // Apply the decay function

        sigma = arma::max(trailWeights) / 3.0;
        trailWeights = arma::exp(-arma::pow(trailWeights, 2) / (2 * pow(sigma, 2)));
        trailWeights /= arma::sum(trailWeights);

        // Add the flux coming from the cosmic hit to all its trail points in the pixel map

        for (unsigned int index = 0; index < numTrailPoints; index++)
        {
            trailRow = int(floor(trailRows(index)));
            trailColumn = int(floor(trailColumns(index)));

            if ((trailRow >= 0) && (trailRow < numRows) && (trailColumn >= 0) && (trailColumn < numColumns))
            {
                map(trailRow, trailColumn) += (trailWeights(index) * intensity);

                // Store the column, row and flux value in their respective vectors. If a pixel in the map has had a been hit, the extra flux gets simply added.

                if (rowsOfCosmicsMap.empty() && columnsOfCosmicsMap.empty() && fluxOfCosmicsMap.empty())
                {
                    rowsOfCosmicsMap.push_back(trailRow);
                    columnsOfCosmicsMap.push_back(trailColumn);
                    fluxOfCosmicsMap.push_back(trailWeights(index) * intensity);
                }

                if (trailRow != rowsOfCosmicsMap.back() || trailColumn != columnsOfCosmicsMap.back())
                {
                    rowsOfCosmicsMap.push_back(trailRow);
                    columnsOfCosmicsMap.push_back(trailColumn);
                    fluxOfCosmicsMap.push_back(trailWeights(index) * intensity);
                }
                else
                {
                    fluxOfCosmicsMap.back() += (trailWeights(index) * intensity);
                }

            }
        }

    }


    meanEntryAngle /= numCosmicHits;
    meanTrailLength /= numCosmicHits;
    meanIntensity /= numCosmicHits;

    Log.info("Detector: mean cosmic entry angle (" + area + "): " + to_string(meanEntryAngle) + " rad");
    Log.info("Detector: mean cosmic trail length (" + area + "): " + to_string(meanTrailLength) + " pix");
    Log.info("Detector: mean cosmic hit intensity (" + area + "): " + to_string(meanIntensity) + " e-");
}













/**
 * \brief: Apply the effect of full-well saturation (i.e. blooming) to the pixel map.
 *         If a pixel receives more electrons than the full-well saturation limit 
 *         (expressed in [electrons / pixel]), the additional electrons flow evenly
 *         distributed in positive and negative charge-transfer direction. Electrons
 *         reaching the edge of the CCD will not be detected.
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 * \pre Full-well saturation limit expressed in [electrons].
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Effect of full-well saturation (i.e. blooming) applied to the pixel map.
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map, unless cosmics have been added to it.
 */

void Detector::applyFullWellSaturation()
{
    double pixelValue, numExcessElectrons;

    int jmod;// Row coordinate where excess electrons are transferred from and to

    for (unsigned int row = 0; row < numRowsPixelMap; row++)
    {
        for (unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            pixelValue = pixelMap(row, column);

            // If the full-well saturation limit has been exceeded, distribute
            // the electrons evenly in the wells above and below until the
            // saturation has disappeared (stay in the same column!)

            if (pixelValue > fullWellSaturationLimit)
            {
                // Transfer excess electrons up

                jmod = row;
                numExcessElectrons = (pixelValue - fullWellSaturationLimit) / 2.0;   // Move half of the excess electrons down...

                // When we move half of the excess electrons down, the pixel below may also become saturated
                // (or was already saturated). When processing this pixel below, we need to avoid that it also
                // sends its excess electrons up and down, and therefore sends part of its excess electrons
                // back up where they actually came from. The variable 'transfer2Saturated' is to keep track
                // whether the destination pixel is also saturated, so that it does not send back electrons
                // where they came from.

                bool transfer2Saturated = false;

                while (numExcessElectrons > 0 && jmod < numRowsPixelMap)
                {
                    if(!transfer2Saturated)
                    {
                        pixelMap(jmod, column) -= numExcessElectrons;
                    }

                    jmod++;

                    // Electrons reaching the edge of the CCD will not be detected

                    if (jmod < numRowsPixelMap)
                    {
                        if(pixelMap(jmod, column) >= fullWellSaturationLimit)
                        {
                            transfer2Saturated= true;
                        }

                        else{

                            transfer2Saturated = false;

                            pixelMap(jmod, column) += numExcessElectrons;

                            // Make sure the pixel you move the excess electrons to
                            // does not get saturated too

                            if (pixelMap(jmod, column) > fullWellSaturationLimit)
                            {
                                numExcessElectrons = pixelMap(jmod, column) - fullWellSaturationLimit;
                            }

                            else
                            {
                                numExcessElectrons = 0;
                            }
                        }
                    }
                }

                // Transfer excess electrons down

                jmod = row;
                numExcessElectrons = (pixelValue - fullWellSaturationLimit) / 2.0;    // ...and the rest of the excess electrons up

                while (numExcessElectrons > 0 && jmod >= 0)
                {
                    pixelMap(jmod, column) -= numExcessElectrons;
                    jmod--;

                    // Electrons reaching the edge of the CCD will not be detected

                    if (jmod >= 0)
                    {
                        pixelMap(jmod, column) += numExcessElectrons;

                        // Make sure the pixel you move the excess electrons to does not get saturated too

                        if (pixelMap(jmod, column) > fullWellSaturationLimit)
                        {
                            numExcessElectrons = pixelMap(jmod, column) - fullWellSaturationLimit;
                        }

                        else
                        {
                            numExcessElectrons = 0;
                        }
                    }
                }
            }
        }
    }
}











/**
 * \brief Apply the effect of the charge-transfer inefficiency to the pixel map. 
 *        A CCD is read out, row after row, a part of the charge is always left 
 *        behind which then dribbles into the trailing pixels. This causes each
 *        star to have a small "tail". Only visible when the CTI = 1 - CTE is poor.
 *        The exact model used depends on the configuration in the input file. 
 *
 *  \note The pixel map should be expressed in [e-] and not [ADU]
 *
 * \pre Pixel unit in the pixel map: [electrons]
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map, unless cosmics have been added to it.
 */

void Detector::applyCTI()
{
    if (CTImodel == "Simple")
    {
        Log.info("Detector: applying simple CTI model");
        applySimpleCTImodel();
    }
    else if (CTImodel == "Short2013")
    {
        Log.info("Detector: applying Short et al. (2013) CTI model with a constant trap density");
        applyShort2013CTImodel("pixelMap");
        applyShort2013CTImodel("smearingMap");
    }
    else if (CTImodel == "Short2013FromFile")
    {
        Log.info("Detector: applying Short et al. (2013) CTI model with CTI info from file");
        applyShort2013CTImodel("pixelMap");
        applyShort2013CTImodel("smearingMap");

    }
}












void Detector::setInitialNumberOfOccupiedTraps(arma::Mat<float> &numberOfOccupiedTraps)
{
    // Set general parameters

    const double maxVolumePerPixel = pixelSize * pixelSize * 1.e-18  / 2.0;                                  // Vg [m^3]
    const double effectiveElectronMass = 0.5 * Constants::FREEELECTRONMASS;                                  // me [kg]
    const double thermalVelocity = sqrt(3.0 * Constants::KBOLTZMANN * temperature / effectiveElectronMass);  // vt [m/s]
    const double dwellTime = ((numColumns / 2) + numColumnsBiasMap) * serialTransferTime;                    // t  [s]

    // Initialise the trapspecies

    arma::Row<float> alpha(numTrapSpecies, arma::fill::zeros);
    for (int k = 0; k < numTrapSpecies; k++)
    {
        alpha(k) = dwellTime * trapCaptureCrossSection[k] * thermalVelocity * pow(fullWellSaturationLimit, beta) / (2.0 * maxVolumePerPixel);
    }

    arma::Row<float> skyBackground;
    skyBackground = pixelMap.row(0);

    // Set initial number of occupied traps for pixelMap

    for (int k = 0; k < numTrapSpecies; k++)
    {
        arma::Mat<float> currentTrapDensityMap =
	    (meanTrapDensityBOL[k] + (meanTrapDensityEOL[k] - meanTrapDensityBOL[k]) * beginExposureNr / finalExposureNr) * radiationMap;

        arma::Row<float> gamma(numColumnsPixelMap, arma::fill::zeros);
        gamma = 2 * currentTrapDensityMap.row(0) *
               (subFieldZeroPointRow + 1) / pow(fullWellSaturationLimit, beta) / (1 + beta); // +1 as row=0 also has to be transferred once
        arma::Row<float> A = (1 - arma::exp(-alpha(k) * arma::pow(skyBackground / 2, 1 - beta))) /
                             (gamma % arma::pow(skyBackground, beta - 1) + 1);
        arma::Row<float> B = gamma % arma::pow(skyBackground, beta);
        double C = (1 - exp(-dwellTime / releaseTime[k]));
        numberOfOccupiedTraps.row(k) = (A % B) / (A + C);
    }
}







/**
 * \brief Apply charge injection to mitigate the CTI effect. Only the PixelMap is affected.
 *
 * \pre Pixel unit in the pixel map: [electrons]
 * \post Pixel unit in the pixel map: [electrons].
 */

void Detector::applyChargeInjection()
{
    const double chargeToInject = chargeInjectionLevel / 100.0 * fullWellSaturationLimit;   // [e-]

    for (int rowNumber = 0; rowNumber < numRowsPixelMap; rowNumber++)
    {
        // rowNumber is the row number of the subfield. Add the zeropoint to get the CCD row number.

        if ((rowNumber + subFieldZeroPointRow - firstInjectedRow) % injectionRowInterval == 0)
        {
            pixelMap.row(rowNumber) += chargeToInject;
        }
    }
}









/**
 * \brief: Apply the effect of the charge-transfer inefficiency to the
 *         pixel map, using a simple CTI model. The CTI of this simple model
 *         has no dependence on the flux level, nor on the distance of the
 *         readout register.
 *
 * \note The serial register is assumed to have a CTE of 1.
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map, unless cosmics have been added to it.
 */

void Detector::applySimpleCTImodel()
{
    float cti = 1.0 - meanCte;

    // Computing the effects of CTE requires the use of a binomial distribution.
    // To speed up things, we first pre-compute some parts of this distribution.

    // Pre-compute the (natural) logarithms of the first N natural numbers

    vector<double> logs(numRowsPixelMap + subFieldZeroPointRow);
    iota(logs.begin(), logs.end(), 1.0);
    transform(logs.begin(), logs.end(), logs.begin(), ptr_fun<double, double>(log));

    // Compute the partial sums of these logarithms
    // sumOfLogsUpTo[i] contains log((i+1)!) = log(1) + ... + log(i+1)

    vector<double> sumOfLogsUpTo(numRowsPixelMap + subFieldZeroPointRow);
    partial_sum(logs.begin(), logs.end(), sumOfLogsUpTo.begin());

    arma::Row<float> readout;   // Readout strip

    // Loop over all rows in the pixel map (starting at the row farthest away from
    // the readout register)

    for (int row = numRowsPixelMap - 1; row >= 0; row--)
    {
        // Reset the readout register

        readout.zeros(numColumnsPixelMap);

        // Each row picks up flux that is left behind when transferring the rows
        // that are closer to the readout register, row-by-row to the readout
        // register (these rows are looped over via the "index" variable - note
        // that the detector zeropoint is added to it!).

        if (row + subFieldZeroPointRow == 0)
        {
            const double factor1 = meanCte;
            readout += pixelMap(0, arma::span::all) * factor1;
        }
        else
        {
            for (unsigned int index = subFieldZeroPointRow; index <= row + subFieldZeroPointRow; index++)
            {
                const double cteFactor = pow(meanCte, index + 1) * pow(cti, row + subFieldZeroPointRow - index);

                if ((index == 0) || (row - (index - subFieldZeroPointRow) == 0))
                {
                    readout += pixelMap(index - subFieldZeroPointRow, arma::span::all) * cteFactor;
                }
                else
                {
                    const double binomialFactor = exp(sumOfLogsUpTo[row + subFieldZeroPointRow - 1]
                                                      - sumOfLogsUpTo[row - (index - subFieldZeroPointRow) - 1]
                                                      - sumOfLogsUpTo[index - 1]);

                    readout += pixelMap(index - subFieldZeroPointRow, arma::span::all) * cteFactor;
                }
            }
        }

        pixelMap(row, arma::span::all) = readout(0, arma::span::all);
    }
}








/**
 * \brief: Apply the effect of the charge-transfer inefficiency to the map,
 *         using the model described in Short et al., MNRAS 430, 3078-3085 (2013).
 *         Only parallel readout is taken into account here.
 *
 * \note The readout register is assumed to be right next to row [0] of the pixel map.
 *       The map needs to be in [e-], not in [ADU]
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map, unless cosmics have been added to it.
 */

void Detector::applyShort2013CTImodel(string map)
{
    int numColumnsCTI;
    int numRowsCTI;
    int zeroPointRow;

    arma::Mat<float> *matMap    = nullptr;
    arma::Mat<float> *radiation = nullptr;
    arma::Mat<float> *numberOfOccupiedTraps = nullptr;

    // Configure for the different maps

    if (map == "pixelMap")
    {
        numRowsCTI = numRowsPixelMap;
        numColumnsCTI = numColumnsPixelMap;
        zeroPointRow = subFieldZeroPointRow;
        matMap = &pixelMap;
        radiation = &radiationMap;
        numberOfOccupiedTraps  = &numberOfOccupiedTrapsPixelMap;
    }
    else if (map == "smearingMap")
    {
        numRowsCTI = numRowsSmearingMap;
        numColumnsCTI = numColumnsPixelMap;
        zeroPointRow = 4510;
        matMap = &smearingMap;
        radiation = &radiationSmearingMap;
        numberOfOccupiedTraps = &numberOfOccupiedTrapsPixelMap;
        *numberOfOccupiedTraps = (*numberOfOccupiedTraps) / ( ((float)subFieldZeroPointRow + (float)numRowsPixelMap) / 4511.);
    } else
    {
      return;
    }



    // Compute the maximum geometrical volume that electrons can occupy within a pixel.
    // I.e. the volume of the electron cloud when the capacity of the full well is maxed out.
    // E.g if the pixel size is 18 micron, then the volume is 18e-6 * 18e-6 * 1.e-6 / 2.0

    const double maxVolumePerPixel = pixelSize * pixelSize * 1.e-18  / 2.0;                                  // Vg [m^3]

    // Time it takes to transfer 1 row during readout

    const double numberColumnsReadOut = (numColumns / 2) + numColumnsBiasMap;
    const double dwellTime = numberColumnsReadOut * serialTransferTime;                                      // [s]

    // Compute the thermal velocity of the electrons in the silicon

    const double effectiveElectronMass = 0.5 * Constants::FREEELECTRONMASS;                                  // me [kg]
    const double thermalVelocity = sqrt(3.0 * Constants::KBOLTZMANN * temperature / effectiveElectronMass);  // vt [m/s]

    // Arrays to keep track of the trapdensity and the number of occupied traps in a row

    arma::Row<float> trapDensity(numColumnsCTI);

    // Arrays to keep track of the captured and released electrons, for each column in a particular row.

    arma::Row<float> numberOfCapturedElectrons(numColumnsCTI);                                             // Nc
    arma::Row<float> numberOfReleasedElectrons(numColumnsCTI);                                             // Nr

    arma::Row<float> alpha(numTrapSpecies, arma::fill::zeros);
    arma::Row<float> gamma(numColumnsCTI, arma::fill::zeros);

    // Eq. (23) of Short et al. 2013

    for (int ispecies = 0; ispecies < numTrapSpecies; ispecies++)
    {
        alpha(ispecies) = dwellTime * trapCaptureCrossSection[ispecies] * thermalVelocity * pow(fullWellSaturationLimit, beta) 
                            / (2.0 * maxVolumePerPixel);
    }

    // Loop over all rows of the pixel/smearing Map.
    // For each row, the computations are done for all columns simultaneously.

    for (int irow= 0; irow < numRowsCTI; irow++)
    {
        // Loop over all trap species
        for (int ispecies = 0; ispecies < numTrapSpecies; ispecies++)
        {
            // Interpolate between the BOL and EOL to get the trap density for species k corresponding to the current `internalTime`

            arma::Mat<float> currentTrapDensityMap = (meanTrapDensityBOL[ispecies] 
                                                       + (meanTrapDensityEOL[ispecies] - meanTrapDensityBOL[ispecies]) * internalTime / missionDuration) 
                                                     * (*radiation);

            // At the end of every exposure, right before the readout, assume that all traps of every species are occupied. 
            // Given the fast capture times, this seems a reasonable assumption.

            if (irow == 0) {
                (*numberOfOccupiedTraps).row(ispecies) = currentTrapDensityMap.row(irow);
            }
            
            // Compute the accumulated number of traps that the charges will cross during the transfer. 
            // We loop over all the rows that will be crossed as a double to increase accuracy.

            arma::Row<double> totalTrapsAsDouble(numColumnsPixelMap, arma::fill::zeros);

            for (int row = 0; row < irow + 1; row++)
            {
                totalTrapsAsDouble = totalTrapsAsDouble + currentTrapDensityMap.row(irow);
            }

            double valueRow = (meanTrapDensityBOL[ispecies] + (meanTrapDensityEOL[ispecies] - meanTrapDensityBOL[ispecies]) * internalTime / missionDuration);
            arma::Row<double> uniformRow(numColumnsPixelMap, arma::fill::ones);
            totalTrapsAsDouble += zeroPointRow * uniformRow * valueRow;

            // We convert the Row<double> into Row<float>

            arma::Row<float> totalTraps = arma::conv_to<arma::Row<float>>::from(totalTrapsAsDouble);


            // Compute the number of electrons captured in a trap, according to Eq. (22)-(23) of Short et al. (2013).
            // Note that Armadillo uses % for elementwise multiplication.
            // numberOfOccupiedTraps is a NumTrapSpecies x NumColumns matrix.

            gamma = 2 * totalTraps / pow(fullWellSaturationLimit, beta) / (1 + beta); // +1 as row = 0 also has to be transferred once
            
            numberOfCapturedElectrons =
                (gamma % arma::pow((*matMap).row(irow), beta) - (*numberOfOccupiedTraps).row(ispecies)) 
                / (gamma % arma::pow((*matMap).row(irow), beta - 1) + 1) 
                % (1 - arma::exp(-alpha(ispecies) * arma::pow((*matMap).row(irow), 1 - beta)));

            // Captured electron numbers can't be negative, so clip negative value to zero.

            arma::Col<arma::uword> isNegative = arma::find(numberOfCapturedElectrons < 0.0);
            numberOfCapturedElectrons(isNegative).zeros();
            numberOfCapturedElectrons.elem(find_nonfinite(numberOfCapturedElectrons)).zeros();

            // Update the number of occupied traps with the estimated number of captured electrons

            (*numberOfOccupiedTraps).row(ispecies) += numberOfCapturedElectrons;

            // Correct the number of occupied traps with the electrons that were released again during the charge transfer time.

            numberOfReleasedElectrons = (*numberOfOccupiedTraps).row(ispecies) * (1-exp(-dwellTime/releaseTime[ispecies]));
            (*numberOfOccupiedTraps).row(ispecies) -= numberOfReleasedElectrons;

            // Add the electron excess to the current pixel value

            (*matMap).row(irow) += numberOfReleasedElectrons - numberOfCapturedElectrons;
        }
    }

    if (map == "smearingMap")
    {
        *numberOfOccupiedTraps = (*numberOfOccupiedTraps) / ( (4510. + (float)numRowsSmearingMap) / ((float)subFieldZeroPointRow + 1.));
    }
}













/**
 * \brief: Apply the effect of readout smearing to the pixel and the smearing map.
 *         This effect is due to the absence of a shutter (common in space-based
 *         instruments) - the CCD still receives light during frame transfer.
 *         The flux of each pixel is affected by the flux of the pixels
 *         in the same column.  Because the CCD is exposed during the whole
 *         readout and multiple exposures are created, also the pixels further
 *         away from the readout register are affected.
 *
 * NOTES: A smearing map is created and will be used in photometry to remove
 *        the smearing effect from the pixel map.
 *
 * \param exposureTime: Exposure time [s].
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post No bias register map, unless cosmics have been added to it.
 */
void Detector::applyOpenShutterSmearing(float exposureTime)
{
    if (includeRelativeTransmissivity)
    {
        // The mask indicating which pixels in the sub-field are within the FOV and which
        // ones not, has already been created upon construction of the throughput map.

        double rowFOV, xFPmmFirstExposed, yFPmmFirstExposed, xFPmmTop, yFPmmTop, angleFirstExposed, angleTop;

        // Loop over all columns in the sub-field

        for(unsigned int column = 0; column < numColumnsPixelMap; column++) //[pixels in sub-field]
        {
            // If the subfield is close to the edge of the CCD, part of the column may not be exposed due
            // to mechanical vignetting. Stars with pixel coordinates inside a vignetted part of the CCD 
            // should therefore be ignored to determine the open shutter smearing. 
            // First, find the intersection of the current column of the detector with the circle representing the FOV:
            // (rowFOV, column).  If no intersection can be found, this is NaN.
            rowFOV = getRowEdgeFOV(column);

            // The std:: construct is to avoid a gcc & libstdc++ bug with older gcc versions.
            // Cf: Github issue #492
            //     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48891
            //     https://stackoverflow.com/questions/33770374/why-is-isnan-ambiguous-and-how-to-avoid-it

            if(std::isnan(rowFOV))
            {
                // All exposed rows are within the FOV
                numExposedRowsInFOV(column) = numRows - firstRowExposed;
            }
            else
            {
                tie(xFPmmFirstExposed, yFPmmFirstExposed) = pixelToFocalPlaneCoordinates(firstRowExposed, column + subFieldZeroPointColumn);    // detector [pixels] -> focal plane [mm]
                angleFirstExposed = camera.getGnomonicRadialDistanceFromOpticalAxis(xFPmmFirstExposed, yFPmmFirstExposed);

                tie(xFPmmTop, yFPmmTop) = pixelToFocalPlaneCoordinates(numRows - 1, column + subFieldZeroPointColumn);
                angleTop = camera.getGnomonicRadialDistanceFromOpticalAxis(xFPmmTop, yFPmmTop);

                if (angleFirstExposed > radiusFOV)
                {
                    if (angleTop > radiusFOV)
                        numExposedRowsInFOV(column) = 0;

                    else
                        numExposedRowsInFOV(column) = numRows - (int) rowFOV;
                }

                else
                {
                    if (angleTop > radiusFOV)
                        numExposedRowsInFOV(column) = (int) rowFOV - firstRowExposed + 1;
                    else
                        numExposedRowsInFOV(column) = numRows - firstRowExposed;
                }
            }
        }
    }



    // Average out the fluxes in the pixel map per column (of the whole CCD) and make sure it is
    // scaled with the readout time (during which the detector is susceptible to open-shutter smearing)
    // instead of with the exposure time:
    // - rows in the sub-field (in the exposed part of the detector and in the FOV): use actual fluxes (accumulated during exposure)
    // - rows outside the sub-field (in the exposed part of the detector and in the FOV): use total sky background
    // - rows outside the FOV and/or not exposed are not considered (as these rows are shielded off against incoming radiation)

    arma::Row<float> openShutterSmearing = arma::sum(pixelMap % mechanicalVignettingMask, 0);   // Flux in the exposed part of the sub-field that is inside the FOV (per column)
    arma::Row<int> numExposedSubFieldRowsInFOV = arma::sum(mechanicalVignettingMask, 0);        // Number of pixels in the exposed part of the sub-field that are inside the FOV (per column)

    arma::Row<int> numExposedNonSubFieldRowShifts = numExposedRowsInFOV - numExposedSubFieldRowsInFOV + numRowsSmearingMap; // Number of pixels in the exposed part of the detector that do not reside in the sub-field + parallel over-scan
    arma::Row<float> openShutterSmearingOutsideSubField = arma::conv_to<arma::Row<float>>::from(numExposedNonSubFieldRowShifts) * camera.getTotalSkyBackground();

   // Apply all throughput efficiencies (these have not been applied to the total sky background yet)
   // Note that mechanical vignetting has already been taken into account (if applicable)

    if(includeRelativeTransmissivity)
        openShutterSmearingOutsideSubField *= expectedValueRelativeTransmissivity;

    if(includePolarization)
        openShutterSmearingOutsideSubField *= expectedValuePolarization;

    if(includeParticulateContamination)
        openShutterSmearingOutsideSubField *= particulateContaminationEfficiency;

    if(includeMolecularContamination)
        openShutterSmearingOutsideSubField *= molecularContaminationEfficiency;

    // Sect. 4.2.4.5 of PLATO-DLR-PL-RP-001:
    // The expected value E_ang is then the mean over all pixels and results in a value of 0.993

    if(includeQuantumEfficiency)
        openShutterSmearingOutsideSubField *= meanQE;

    openShutterSmearing += openShutterSmearingOutsideSubField;

    // Fast camera: lower half of the CCD is shielded off -> no contribution to open-shutter smearing here

    arma::Row<float> openShutterSmearingTime = arma::conv_to<arma::Row<float>>::from(numExposedRowsInFOV + numRowsSmearingMap) /
        (numRows - firstRowExposed + numRowsSmearingMap) * readoutTimeBeforeNextExposure;

    arma::Row<float> factor = openShutterSmearingTime % arma::pow(arma::conv_to<arma::Row<float>>::from(numExposedRowsInFOV), -1) / exposureTime;

    openShutterSmearing %= factor;

    // Add the effect of the open-shutter smearing to the pixel map

    for (unsigned int row = 0; row < numRowsPixelMap; row++)
        pixelMap(row, arma::span::all) += openShutterSmearing;

    // Add the effect of the open-shutter smearing to the smearing map

    for (unsigned int row = 0; row < numRowsSmearingMap; row++)
        smearingMap(row, arma::span::all) += openShutterSmearing;
}










/**
 * \brief Find the intersection of the circle representing the FOV and the given column of the
 *        detector.  In case this circle does intersect with the detector, the row coordinate
 *        is returned; if not, NaN is returned.
 *
 * \param column: Column (coordinate) [pixels], for which to find the intersection with the
 *                circle representing the FOV.
 *
 * \return In case this circle does intersect with the detector, the row coordinate
 *         is returned; if not, NaN is returned.
 **/
double Detector::getRowEdgeFOV(int column)
{
    double offsetCol;
    double offsetRow;
    double pixelSizeMm = pixelSize / 1000.0;    // Pixel size [µm] -> [mm]

    if (ccdPosition == "Custom")
    {
        offsetCol = customOriginOffsetX;
        offsetRow = customOriginOffsetY;
    }
    else
    {
        int ccd = stoi(ccdPosition) - 1;
        array<double, 12> currentCcdPositions = (*ccdPositions)();
        offsetCol = currentCcdPositions[ccd * 3];
        offsetRow = currentCcdPositions[ccd * 3 + 1];
    }

    // Quadratic equation: a * x**2 + b * x + c  = 0
    // Find intersection between circle representing the FOV and the given column

    double a = pow(pixelSizeMm, 2);
    double b = 2 * pixelSizeMm * ( pixelSizeMm * subFieldZeroPointRow - offsetRow);
    double c = pow(pixelSizeMm * subFieldZeroPointRow - offsetRow, 2)
             + pow((column + subFieldZeroPointColumn) * pixelSizeMm - offsetCol , 2)
             - pow(camera.getFocalLength() * tan(radiusFOV), 2);

    // Discriminant (should be positive)

    double discriminant = pow(b, 2) - 4 * a * c;

    if (discriminant < 0)
        return nan("");

    double discriminantSqrt = sqrt(discriminant);

    // First solution of the equation
    // Check whether it intersects with the detector (if not, try the other solution)

    double solutionRow = (-b + discriminantSqrt) / (2 * a);

    if ((solutionRow >= 0) && (solutionRow < numRows))
        return solutionRow;

    // Second solution of the equation
    // Check whther it intersects with the detector (if not, the whole row is within the FOV)

    solutionRow = (-b - discriminantSqrt) / (2 * a);

    if ((solutionRow >= 0) && (solutionRow < numRows))
        return solutionRow;

    // No proper solution found (i.e. circle of FOV does not intersect
    // with the detector in the given column)

    return nan("");
}













/**
 * \brief Apply the readout noise to the pixel map, bias map, and smearing map.  
 *        The readout noise is contributed to by the detector and by the FEE.
 *
 * \details Readout noise occurs due to the imperfect nature of the CCD amplifiers.
 *          When the electrons are transferred to the amplifier, the induced voltage
 *          is measured. However, this measurement is not perfect, but gives a value
 *          which is on average correct, with the readout noise as variance.
 *
 * \pre Pixel unit in the pixel map: [electrons].
 * \pre Pixel unit in the smearing map: [electrons].
 * \pre No bias register map, unless cosmics have been added to it.
 * \pre Readout noise expressed in [electrons].
 *
 * \post Pixel unit in the pixel map: [electrons].
 * \post Pixel unit in the smearing map: [electrons].
 * \post Pixel unit in the bias register map: [electrons].
 * \post Added readout noise to the bias and smearing maps.
 */

void Detector::addReadoutNoise()
{
    // Adding the readout noise in quadrature (CCD & FEE)

    const double totalReadoutNoise = sqrt(pow(readoutNoise, 2) + pow(frontEndElectronics->getReadoutNoise(), 2));

    readoutNoiseDistribution = normal_distribution<double>(0.0, totalReadoutNoise);

    // Add readout noise to the pixel map

    for (unsigned int row = 0; row < numRowsPixelMap; row++)
    {
        for (unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            pixelMap(row, column) += readoutNoiseDistribution(readoutNoiseGenerator);
        }
    }

    // Add readout noise to the bias prescan map

    for(unsigned int row = 0; row < numRowsBiasMap; row++)
    {
        for(unsigned int column = 0; column < numColumnsBiasMap; column++)
        {
            biasMapLeft(row, column) += readoutNoiseDistribution(readoutNoiseGenerator);
            biasMapRight(row, column) += readoutNoiseDistribution(readoutNoiseGenerator);
        }
    }

    // Add readout noise to the smearing overscan map

    for (unsigned int row = 0; row < numRowsSmearingMap; row++)
    {
        for (unsigned int column = 0; column < numColumnsPixelMap; column++)
        {
            smearingMap(row, column) += readoutNoiseDistribution(readoutNoiseGenerator);
        }
    }
}









/**
 * \brief: Apply quantisation to the bias register, smearing, and smearing map.  This consists of
 *         the following steps:
 *         - applying FEE and CCD gain (converting from electrons to ADU)
 *         - adding the electronic offset
 *         - forcing the ADUs to be integers
 *         - applying digital saturation
 *
 * \pre Pixel unit in the pixel, smearing, and bias register maps: [electrons].
 *
 * \post Pixel unit in the pixel, smearing, and bias register maps: [ADU].
 */
void Detector::applyQuantisation()
{
    // Apply the gain, to increase the dynamic range of the detector.
    // Pixel units before: [electrons]
    // Pixel units after: [ADU]

    applyGain();

    // Take into account the bias level (i.e. add the constant "zero" level
    // introduced by the amplifier).
    // Pixel units before: [ADU]
    // Pixel units after: [ADU]

    addElectronicOffset();


    // At this point, because pixel values are floats, the previous steps may have
    // resulted in fractional ADUs. Take care that pixel maps, bias maps, and smearing
    // maps do not have fractional values.

    pixelMap = arma::floor(pixelMap);
    biasMapLeft = arma::floor(biasMapLeft);
    biasMapRight = arma::floor(biasMapRight);
    smearingMap = arma::floor(smearingMap);


    // Take into account digital saturation. If even after dividing by the gain
    // the number of ADUs in a pixel is still higher than the analogue-digital
    // converter (ADC) can represent with its fixed amount of bits, clip all
    // values that are too high to the saturation level of the ADC.
    // Pixel units before: [ADU]
    // Pixel units after: [ADU]

    if (includeDigitalSaturation)
    {
        Log.debug("Detector: applying digital saturation to pixelMap, biasMap and smearingMap (digitalSaturationLimit=" + to_string(digitalSaturationLimit) + ")");
        applyDigitalSaturation();
    }
    else
    {
        Log.debug("Detector: no digital saturation applied.");
    }
}










/**
 * \brief: Divide the bias register, smearing, and pixel map by the detector gain.
 *         This converts these three maps from electrons to ADU. The gain is applied
 *         _before_ adding the bias.
 *
 * \pre Pixel unit in the pixel, smearing, and bias register maps: [electrons].
 *
 * \post Pixel unit in the pixel, smearing, and bias register maps: [ADU].
 */
void Detector::applyGain()
{
    Log.debug("Detector: applying gain to pixelMap, biasMap and smearingMap");

    // Index of the last column of the left detector half...

    const int lastIndexCcdLeft = numColumns / 2 - 1;                                // in the CCD reference frame [pixels]
    const int lastIndexSubFieldLeft = lastIndexCcdLeft - subFieldZeroPointColumn;   // in the sub-field reference frame [pixels]

    // Detector gain (left & right) [µV / e-]

    const double ccdGainOverDeltaTemp = gainStability * (getTemperature() - nominalOperatingTemperature);

    const double ccdGainLeft = refValueGainLeft + ccdGainOverDeltaTemp;
    const double ccdGainRight = refValueGainRight + ccdGainOverDeltaTemp;

    // Combined gain (FEE & CCD): [ADU / e-]
    // FEE gain (left & right):   [ADU / µV]

    combinedGainLeft = frontEndElectronics->getGainLeftAdc(internalTime) * ccdGainLeft;
    combinedGainRight = frontEndElectronics->getGainRightAdc(internalTime) * ccdGainRight;

    // In what follows we have to take into account whether the non-linearity of the gain needs to be taken into account.
    // If not, than the pixel level simply needs to be multiplied by the gain to go from [e-] to [ADU]. The caveat is that we need to 
    //         multiply with the right gain, depending on which half of the CCD the subfield is. 
    // If yes, we use the following formula:
    //         I_out  = B + I_in + a0 + a1 * I_in + a2 * I_in^2
    // where I_out is the pixel signal in ADU with non-linearity taken into account, I_in is the pixel signal in ADU without non-linearity
    // taken into account. B is the bias in ADU which we ignore here, because the bias will be applied after the gain in PlatoSim.
    // If 
    //         I_in = g * I_e
    // with g=gain in [ADU/e-], and I_e is the signal in [e-], we can rewrite the formula above as:
    //         I_out = a0 + g * I_e * (1 + a1 + a2 * g * I_e)
    // which is what is implemented, again with the caveat that the right gain must be chosen.
    //
    // Note: in Armadillo, algebraic multiplication is done with '*', elementwise multiplication with '%'.

    const double a0 = gainNonlinearityCoefficients[0];                              // [ADU]
    const double a1 = gainNonlinearityCoefficients[1];                              // [-]
    const double a2 = gainNonlinearityCoefficients[2];                              // [-]

    if(lastIndexSubFieldLeft >= ((int) numColumnsPixelMap - 1))                     // Left ADC only
    {
        if (includeGainNonlinearity) {
            pixelMap    = a0 + combinedGainLeft * pixelMap % (1.0 + a1 + a2 * combinedGainLeft * pixelMap);
            smearingMap = a0 + combinedGainLeft * smearingMap % (1.0 + a1 + a2 * combinedGainLeft * smearingMap);
        } else {
            pixelMap    *= combinedGainLeft;
            smearingMap *= combinedGainLeft;
        }
    }
    else if(lastIndexSubFieldLeft < 0)                                              // Right ADC only
    {
        if (includeGainNonlinearity) {
            pixelMap    = a0 + combinedGainRight * pixelMap % (1.0 + a1 + a2 * combinedGainRight * pixelMap);
            smearingMap = a0 + combinedGainRight * smearingMap % (1.0 + a1 + a2 * combinedGainRight * smearingMap);
        } else {
            pixelMap    *= combinedGainRight;
            smearingMap *= combinedGainRight;
        }
    }
    else
    {
        // 0 -> lastIndexSubFieldLeft (incl.): left ADC

        if (includeGainNonlinearity) {
            // I abbreviate/alias the submatrices into myMap and myMap2 to make the equations more readable.

            const auto &myMap = pixelMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft));
            pixelMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft)) =
                a0 + combinedGainLeft * myMap % (1.0 + a1 + a2 * combinedGainLeft * myMap);

            const auto &myMap2 = smearingMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft)); 
            smearingMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft)) = 
                a0 + combinedGainLeft * myMap2 % (1.0 + a1 + a2 * combinedGainLeft * myMap2);
        } else {
            pixelMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft)) *= combinedGainLeft;
            smearingMap.submat(arma::span::all, arma::span(0, lastIndexSubFieldLeft)) *= combinedGainLeft;
        }

        // lastIndexSubFieldLeft + 1 -> numColumnsSubPixelMap - 1 (incl.): right ADC

        if (includeGainNonlinearity) {
            const auto &myMap = pixelMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1));
            pixelMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1)) = 
                a0 + combinedGainRight * myMap % (1.0 + a1 + a2 * combinedGainRight * myMap);

            const auto &myMap2 = smearingMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1));
            smearingMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1)) = 
                a0 + combinedGainRight * myMap2 % (1.0 + a1 + a2 * combinedGainRight * myMap2);
        } else {
            pixelMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1)) *= combinedGainRight;
            smearingMap.submat(arma::span::all, arma::span(lastIndexSubFieldLeft + 1, numColumnsPixelMap - 1)) *= combinedGainRight;
        }
    }

    if (includeGainNonlinearity) {
        biasMapLeft  = a0 + combinedGainLeft * biasMapLeft % (1.0 + a1 + a2 * combinedGainLeft * biasMapLeft);
        biasMapRight = a0 + combinedGainRight * biasMapRight % (1.0 + a1 + a2 * combinedGainRight * biasMapRight);
    } else {
        biasMapLeft  *= combinedGainLeft;
        biasMapRight *= combinedGainRight;
    }

    Log.info("Detector: gain of left part of CCD: " + to_string(combinedGainLeft));
    Log.info("Detector: gain of right part of CCD: " + to_string(combinedGainRight));

    if (includeGainNonlinearity) {
        Log.info("Detector: including gain non-linearity");
    }

}











/**
 * \brief: Add the electronic offset (i.e. bias level) of the FEE to the pixel map,
 *         smearing map, and bias map.
 *
 * \pre Pixel unit in the pixel, smearing, and bias maps: [ADU].
 * \pre Electronic offset (i.e. bias level) expressed in [ADU].
 *
 * \post Pixel unit in the pixel, smearing, and bias register maps: [ADU].
 * \post Electronic offset added to the pixel, smearing, and bias register maps.
 */

void Detector::addElectronicOffset()
{
    const double offset = frontEndElectronics->getElectronicOffset(internalTime);

    Log.debug("Detector: adding a bias to pixelMap, biasMap and smearingMap (electronicOffset=" + to_string(offset)+ ")");

    // Add the electronic offset to the pixel, bias register, and smearing maps

    pixelMap += offset;
    biasMapLeft += offset;
    biasMapRight += offset;
    smearingMap += offset;
}












/**
 * \brief Apply the effect of digital saturation to the pixel map,
 *        smearing map, and bias register map. This means that the pixel values in
 *        these maps (expressed in [ADU / pixel]) are Lastped off to the digital saturation
 *        limit of the detector (also expressed in [ADU / pixel]).
 *
 * \pre Pixel unit in the pixel, smearing, and bias maps: [ADU].
 * \pre Digital saturation limit expressed in [ADU / pixel].
 *
 * \post Pixel unit in the pixel, smearing, and bias register maps: [ADU].
 */
void Detector::applyDigitalSaturation()
{
    // Last off the values in the pixel map

    pixelMap(arma::find(pixelMap > digitalSaturationLimit)).fill(digitalSaturationLimit);

    // Last off the values in the bias register map

    biasMapLeft(arma::find(biasMapLeft > digitalSaturationLimit)).fill(digitalSaturationLimit);
    biasMapRight(arma::find(biasMapRight > digitalSaturationLimit)).fill(digitalSaturationLimit);

    // Last off the values in the smearing map

    smearingMap(arma::find(smearingMap > digitalSaturationLimit)).fill(digitalSaturationLimit);
}












/**
 * \brief Applies the F-FEE over-/undershoot to the pixel map.
 *
 * \pre Pixel unit in the pixel, smearing, and bias maps: [ADU].
 *
 * \post Pixel unit in the pixel, smearing, and bias register maps: [ADU].
 */
void Detector::applyOverAndUnderShoot()
{
    const unsigned int halfDectectorWidth = numColumns / 2;

    // NOTES
    // - We apply this effect row per row, because otherwise, we will have to keep too much
    //   data in memory
    // - Both detector halves must be treated independently
    // - The pixels between the bias map and the pixel map are assumed to contain the sky background

    arma::frowvec readoutRegister;   // Placeholder
    arma::frowvec difference;        // Placeholder
    arma::frowvec totalContribution; // Has to be filled with zeroes for every row and every detector half
    int lengthReadoutRegister;

    double skyBackground = camera.getTotalSkyBackground();

    if (includeRelativeTransmissivity)
        skyBackground *= expectedValueRelativeTransmissivity;
    if (includePolarization)
        skyBackground *= expectedValuePolarization;
    if (includeParticulateContamination)
        skyBackground *= particulateContaminationEfficiency;
    if (includeMolecularContamination)
        skyBackground *= molecularContaminationEfficiency;
    skyBackground *= meanQE * meanAngleDependencyQE;

    arma::frowvec skyBackgroundExtraPixels(frontEndElectronics->getOverAndUnderShootRange());
    skyBackgroundExtraPixels.fill(skyBackground);

    // At least partially on the left detector half

    if (subFieldZeroPointColumn < halfDectectorWidth)
    {
        const int firstIndexLeftHalf = subFieldZeroPointColumn;
        const int lastIndexLeftHalf = min(halfDectectorWidth - 1, subFieldZeroPointColumn + numColumnsPixelMap - 1);
        const int numCcdPixelsLeftHalf = lastIndexLeftHalf - firstIndexLeftHalf + 1;

        lengthReadoutRegister = numCcdPixelsLeftHalf + frontEndElectronics->getOverAndUnderShootRange();    // Pixels in sub-field on left CCD half + some extra pixels closest to the readout electronics
        readoutRegister.zeros(lengthReadoutRegister);
        readoutRegister.head(frontEndElectronics->getOverAndUnderShootRange()) = skyBackgroundExtraPixels;

        for (unsigned int row = 0; row < numRowsPixelMap; row++)
        {
            readoutRegister(arma::span(frontEndElectronics->getOverAndUnderShootRange(), lengthReadoutRegister - 1)) = pixelMap.row(row).head(numCcdPixelsLeftHalf);
            totalContribution.zeros(lengthReadoutRegister);

            difference = readoutRegister.tail(lengthReadoutRegister - 1) - readoutRegister.head(lengthReadoutRegister - 1);

            for (int deltaX = 0; deltaX < frontEndElectronics->getOverAndUnderShootRange(); deltaX++)
            {
                // Ditch last deltaX and tail(numCcdPixelsLeftHalf)

                totalContribution.tail(lengthReadoutRegister - frontEndElectronics->getOverAndUnderShootRange()) += frontEndElectronics->getOverAndUnderShootStrength() * difference.head(lengthReadoutRegister - 1 - deltaX).tail(numCcdPixelsLeftHalf) * exp(-frontEndElectronics->getOverAndUnderShootDecayRate() * pow(deltaX, frontEndElectronics->getOverAndUnderShootDecaySpeed()));

                // totalContribution.tail(lengthReadoutRegister - deltaX) += frontEndElectronics->getOverAndUnderShootStrength() * difference * exp(-frontEndElectronics->getOverAndUnderShootDecayRate() * pow(deltaX, frontEndElectronics->getOverAndUnderShootDecaySpeed()));
            }

            pixelMap.row(row).head(numCcdPixelsLeftHalf) += totalContribution.tail(lengthReadoutRegister - frontEndElectronics->getOverAndUnderShootRange());
        }
    }

    // At least partially on the right detector half

    if (subFieldZeroPointColumn + numColumnsPixelMap - 1 >= halfDectectorWidth)
    {
        const int firstIndexRightHalf = max(halfDectectorWidth, subFieldZeroPointColumn);
        const int lastIndexRightHalf = subFieldZeroPointColumn + numColumnsPixelMap - 1;
        const int numCcdPixelsRightHalf = lastIndexRightHalf - firstIndexRightHalf + 1;

        lengthReadoutRegister = numCcdPixelsRightHalf + frontEndElectronics->getOverAndUnderShootRange();      // Pixels in sub-field on right CCD half + some extra pixels closest to the readout electronics
        readoutRegister.zeros(lengthReadoutRegister);
        readoutRegister.tail(frontEndElectronics->getOverAndUnderShootRange()) = skyBackgroundExtraPixels;

        for (unsigned int row = 0; row < numRowsPixelMap; row++)
        {
            readoutRegister(arma::span(0, lengthReadoutRegister - frontEndElectronics->getOverAndUnderShootRange() - 1)) = pixelMap.row(row).tail(numCcdPixelsRightHalf);
            totalContribution.zeros(lengthReadoutRegister);

            difference = readoutRegister.head(lengthReadoutRegister - 1) - readoutRegister.tail(lengthReadoutRegister - 1);

            for (int deltaX = 0; deltaX < frontEndElectronics->getOverAndUnderShootRange(); deltaX++)
            {
                // Ditch first deltaX and head(numCcdPixelsRightHalf)

                totalContribution.head(lengthReadoutRegister - frontEndElectronics->getOverAndUnderShootRange()) += frontEndElectronics->getOverAndUnderShootStrength() * difference.tail(lengthReadoutRegister - 1 - deltaX).head(numCcdPixelsRightHalf) * exp(-frontEndElectronics->getOverAndUnderShootDecayRate() * pow(deltaX, frontEndElectronics->getOverAndUnderShootDecaySpeed()));
            }

            pixelMap.row(row).tail(numCcdPixelsRightHalf) += totalContribution.head(lengthReadoutRegister - frontEndElectronics->getOverAndUnderShootRange());
        }
    }
}














/**
 * \brief Compute the (x,y) coordinates [mm] in the FP reference system
 *        given the (real-valued) pixel row and column numbers on the CCD.
 *
 * \note  The rows correspond to the y-direction, and the columns to the x-direction.
 *        Pixel (row, col) = (0,0) starts at (yFP, xFP) = (0, 0).
 *
 * \param row     CCD row coordinate, real-valued (e.g. 3.5)    [pix]
 * \param column  CCD column coordinate, real-valued (e.g. 8.3) [pix]
 *
 * \return (xFP, yFP)  A pair of (x,y) coordinates in the FP reference system [mm]
 */

pair<double, double> Detector::pixelToFocalPlaneCoordinates(double row, double column)
{

    // Convert the pixel coordinates into [mm] coordinates

    const double xCCDmm = column * pixelSize / 1000.0;
    const double yCCDmm = row * pixelSize / 1000.0;

    // Convert the CCD coordinates into FP coordinates [mm]

    double xFP, yFP;

    if (ccdPosition == "Custom")
    {
        xFP = (xCCDmm - customOriginOffsetX) * cos(customOrientationAngle) - (yCCDmm - customOriginOffsetY) * sin(customOrientationAngle);
        yFP = (xCCDmm - customOriginOffsetX) * sin(customOrientationAngle) + (yCCDmm - customOriginOffsetY) * cos(customOrientationAngle);
    }

    else
    {
        int ccd = stoi(ccdPosition) - 1;

        array<double, 12> currentCcdPositions = (*ccdPositions)();

        double originOffsetX = currentCcdPositions[ccd * 3];
        double originOffsetY = currentCcdPositions[ccd * 3 + 1];
        double orientationAngle = deg2rad(currentCcdPositions[ccd * 3 + 2]);

        xFP = (xCCDmm - originOffsetX) * cos(orientationAngle) - (yCCDmm - originOffsetY) * sin(orientationAngle);
        yFP = (xCCDmm - originOffsetX) * sin(orientationAngle) + (yCCDmm - originOffsetY) * cos(orientationAngle);
    }

    // That's it

    return make_pair(xFP, yFP);
}










/**
 * \brief Compute the (real-valued) pixel coordinates of the star on the CCD, given the
 *        (x,y) coordinates [mm] in the FP reference system
 *
 * \note  - The rows correspond to the y-direction, and the columns to the x-direction.
 *        - Pixel (row, col) = (0,0) starts at (yFP, xFP) = (0, 0).
 *
 * \param xFP  x-coordinate of the point in the FP reference system  [mm]
 * \param yFP  y-coordinate of the point in the FP reference system  [mm]
 *
 * \return (row, column)  row and column pixel coordinates of the point (real-valued) [pix]
 */

pair<double, double> Detector::focalPlaneToPixelCoordinates(double xFP, double yFP)
{
    double xCCDmm, yCCDmm;

    // Convert the FP coordinates into CCD coordinates [mm]

    if (ccdPosition == "Custom")
    {
        xCCDmm = customOriginOffsetX + xFP * cos(customOrientationAngle) + yFP * sin(customOrientationAngle);
        yCCDmm = customOriginOffsetY - xFP * sin(customOrientationAngle) + yFP * cos(customOrientationAngle);
    }

    else
    {
        int ccd = stoi(ccdPosition) - 1;

        array<double, 12> currentCcdPositions = (*ccdPositions)();

        double originOffsetX = currentCcdPositions[ccd * 3];
        double originOffsetY = currentCcdPositions[ccd * 3 + 1];
        double orientationAngle = deg2rad(currentCcdPositions[ccd * 3 + 2]);

        xCCDmm = originOffsetX + xFP * cos(orientationAngle) + yFP * sin(orientationAngle);
        yCCDmm = originOffsetY - xFP * sin(orientationAngle) + yFP * cos(orientationAngle);
    }

    // Convert the [mm] coordinates into pixel coordinates

    const double column = xCCDmm / pixelSize * 1000.0;
    const double row = yCCDmm / pixelSize * 1000.0;

    // That's it

    return make_pair(row, column);
}












/**
 * \brief  Return the focal plane coordinates of the center pixel of the subfield
 *
 * \return (xFP, yFP)   focal plane coordinates in the FP'reference system [mm]
 */

pair<double, double> Detector::getFocalPlaneCoordinatesOfSubfieldCenter()
{
    double centerRow = subFieldZeroPointRow + numRowsPixelMap / 2.0;
    double centerCol = subFieldZeroPointColumn + numColumnsPixelMap / 2.0;

    // The columns correspond to the x-coordinate, the rows to the y-coordinate

    double xFP, yFP;
    tie(xFP, yFP) = pixelToFocalPlaneCoordinates(centerRow, centerCol);

    return make_pair(xFP, yFP);
}










/**
 * \brief Return the (X,Y) coordinates in the FP' reference frame in [mm] of the 4 corners
 *        of the subfield
 *
 * \return (X00, Y00, X01, Y01, X11, Y11, X10, Y10)  [mm]
 *         where: (X00, Y00) are the FP coordinates of the lower left corner of the subfield
 *                (X01, Y01) are the FP coordinates of the lower right corner of the subfield
 *                (X11, Y11) are the FP coordinates of the upper right corner of the subfield
 *                (X10, Y10) are the FP coordinates of the upper left corner of the subfield
 */

tuple<double, double, double, double, double, double, double, double> Detector::getFocalPlaneCoordinatesOfSubfieldCorners()
{
    double corner00Xmm, corner00Ymm, corner01Xmm, corner01Ymm, corner11Xmm, corner11Ymm, corner10Xmm, corner10Ymm;
    double row, col;

    // Lower left corner

    row = subFieldZeroPointRow;
    col = subFieldZeroPointColumn;
    tie(corner00Xmm, corner00Ymm) = pixelToFocalPlaneCoordinates(row, col);

    // Lower right corner

    row = subFieldZeroPointRow;
    col = subFieldZeroPointColumn + numColumnsPixelMap;
    tie(corner01Xmm, corner01Ymm) = pixelToFocalPlaneCoordinates(row, col);

    // Upper right corner

    row = subFieldZeroPointRow + numRowsPixelMap;
    col = subFieldZeroPointColumn + numColumnsPixelMap;
    tie(corner11Xmm, corner11Ymm) = pixelToFocalPlaneCoordinates(row, col);

    // Upper left corner

    row = subFieldZeroPointRow + numRowsPixelMap;
    col = subFieldZeroPointColumn;
    tie(corner10Xmm, corner10Ymm) = pixelToFocalPlaneCoordinates(row, col);

    return make_tuple(corner00Xmm, corner00Ymm, corner01Xmm, corner01Ymm, corner11Xmm, corner11Ymm, corner10Xmm, corner10Ymm);
}












/**
 * \brief Return the solid angle of 1 single pixel on the sky. [sr]
 *
 * \param plateScale  The platescale of the camera [arcsec/micron]
 *
 * \return            Solid angle in [s]
 */

double Detector::getSolidAngleOfOnePixel(double plateScale)
{
    return sqDeg2sr(pow(pixelSize * plateScale / 3600.0, 2));
}












// /**
//  * \brief      Return the orientation of the CCD with respect to the orientation
//  *             of the focal plane. The rotations of the CCD are counter
//  *             clockwise.
//  *
//  * \return     the orientation of the CCD [radians]
//  */
// double Detector::getOrientationAnglePsf()
// {
//     return orientationAngle;
// }












/**
 * \brief     Set the subfield with a given array.
 *
 * \details   This function is primarily used for testing the code. One should not first get the pixelMap
 *            perform an operation, and then setSubfield() again. Instead, let Detector do the operation.
 *
 * \param subfield
 */

void Detector::setSubfield(const arma::Mat<float> &subfield)
{
    // Check if the given matrix has the proper dimensions. If not complain, and exit.

    if ((subfield.n_rows != pixelMap.n_rows) || (subfield.n_cols != pixelMap.n_cols))
    {
        Log.error("Detector: setSubfield with incompatible array shape: ("
                  + to_string(subfield.n_rows) + ", " + to_string(subfield.n_cols) + ") != ("
                  + to_string(pixelMap.n_rows) + ", " + to_string(pixelMap.n_cols) + ")");
        exit(1);
    }

    // Copy the contents of the subfield array into our pixelMap

    pixelMap = subfield;
}












/**
 * \brief Return a copy of the pixelMap matrix
 *
 * \details   This function is primarily used for testing the code. One should not first get the pixelMap
 *            perform an operation, and then setSubfield() again. Instead, let Detector do the operation.
 *
 * \return pixelMap
 */

 arma::Mat<float> Detector::getSubfield()
 {
    return pixelMap;
 }











/**
 * \brief: Creates the group(s) in the HDF5 file where the detector specific
 *         information will be stored.  These groups have to be created once,
 *         at the very beginning.
 */
void Detector::initHDF5Groups()
{
    Log.debug("Detector: initialising HDF5 groups");

    hsize_t exposures = static_cast<hsize_t>(finalExposureNr - beginExposureNr);
    H5::PredType type = includeQuantisation ? H5::PredType::NATIVE_UINT16
                                            : H5::PredType::NATIVE_FLOAT;
    if (writePixelMaps)
    {
	hsize_t dim[3] = {exposures,
			  static_cast<hsize_t>(numRowsPixelMap),
			  static_cast<hsize_t>(numColumnsPixelMap)};

        hdf5File.createGroup("/Images", "subfield", dim, type);
      }

    if (writeBiasMaps)
    {
        hsize_t dim[3] = {
	  exposures,
	  static_cast<hsize_t>(numRowsBiasMap),
	  static_cast<hsize_t>(numColumnsBiasMap)};

        hdf5File.createGroup("/BiasMapsLeft", "bias", dim, type);
        hdf5File.createGroup("/BiasMapsRight", "bias", dim, type);

    }

    if (writeSmearingMaps)
    {
      hsize_t dim[3] = {exposures,
			static_cast<hsize_t>(numRowsSmearingMap),
			static_cast<hsize_t>(numColumnsPixelMap)};

      hdf5File.createGroup("/SmearingMaps", "smearing", dim, type);
    }

    if (writeThroughputMaps)
    {
        hsize_t dim[3] = {exposures,
			  static_cast<hsize_t>(numRowsPixelMap),
			  static_cast<hsize_t>(numColumnsPixelMap)};

	hdf5File.createGroup("/ThroughputMaps", "throughput", dim, H5::PredType::NATIVE_FLOAT);
    }
   
    if (writeBackgroundMap || constantSkyBackground)
    {
        hdf5File.createGroup("/BackgroundMap");
    }    
    
    if (writeCTI && (CTImodel == "Short2013"))
    {
        hdf5File.createGroup("/CTI");
    }

    if (writeCosmics)
    {
          hdf5File.createGroup("/Cosmics");
          subFieldCosmics = hdf5File.createGroupForCosmics(
              "/SubField", exposures);
          smearingCosmics = hdf5File.createGroupForCosmics(
              "/SmearingMap", exposures);
          leftBiasCosmics = hdf5File.createGroupForCosmics(
              "/BiasMapLeft", exposures);
          rightBiasCosmics = hdf5File.createGroupForCosmics(
	      "/BiasMapRight", exposures);
      }

    if (includeStraylight)
      {
        hdf5File.createGroup("/Straylight");
      }

    if (writeBadPixelMap)
    {
        hdf5File.createGroup("/BadPixelMap");
    }
}









/**
 * \brief: Write the bad pixel map for subfield and entire CCD to the HDF5 file.
 */ 
void Detector::writeBadPixelMapToHDF5()
{
    arma::Mat<float> badPixelCCDMap =
        arma::conv_to<arma::Mat<float>>::from(badPixelMap);
    badPixelCCDMap.transform([](double val) {return (val > 0.5) ? 0. : 1.; });
    hdf5File.writeArray("/BadPixelMap", "CCD", badPixelCCDMap);

    unsigned int subFieldLastPointRow = subFieldZeroPointRow + numRowsPixelMap;
    unsigned int subFieldLastPointColumn = subFieldZeroPointColumn + numColumnsPixelMap;


    arma::Mat<float> badPixelSubfieldMap =
        arma::conv_to<arma::Mat<float>>::from(
                badPixelCCDMap.submat(subFieldZeroPointRow, subFieldZeroPointColumn, subFieldLastPointRow - 1, subFieldLastPointColumn - 1));

    hdf5File.writeArray("/BadPixelMap", "Subfield", badPixelSubfieldMap);
}


/**
 * \brief: Write the background map to the HDF5 file.
 */
void Detector::writeBackgroundMapToHDF5()
{
    if (writeBackgroundMap)
    {
        string imageName = "skyBackground";
        arma::Mat<float> backgroundMapBOS = backgroundMap*transmissionEfficiencyBOS;
        hdf5File.writeArray("/BackgroundMap", imageName, backgroundMapBOS);
    }
}









/**
 * \brief: Writes the pixel map, smearing map, bias maps and throughput maps into the HDF5 file.
 *
 * \param exposureNr:   Sequential number of the exposure
 */
void Detector::writePixelMapsToHDF5(int exposureNr)
{

    // Write the float arrays to HDF5
    if (!includeQuantisation) {

        // Add the image to the "Images" group
        if (writePixelMaps) {
	    hdf5File.writeArray("/Images", "subfield", exposureNr-beginExposureNr, pixelMap);
	}
	// Add the smearing map
	if (writeSmearingMaps) {
	    hdf5File.writeArray("/SmearingMaps", "smearing", exposureNr-beginExposureNr, smearingMap);
	}
	// Add the bias maps
	if (writeBiasMaps) {
	    hdf5File.writeArray("/BiasMapsLeft", "bias", exposureNr-beginExposureNr, biasMapLeft);
	    hdf5File.writeArray("/BiasMapsRight", "bias", exposureNr-beginExposureNr, biasMapRight);
	}
    }
    else
    {
        // Add the image to the "Images" group
        if (writePixelMaps) {
            hdf5File.writeArray("/Images", "subfield", exposureNr-beginExposureNr, floatToUint(pixelMap, "pixelMap"));
	}
	// Add the smearing map
	if (writeSmearingMaps) {
	   hdf5File.writeArray("/SmearingMaps", "smearing", exposureNr-beginExposureNr, floatToUint(smearingMap, "smearingMap"));
	}
	// Add the bias maps
	if (writeBiasMaps) {
	    hdf5File.writeArray("/BiasMapsLeft", "bias", exposureNr-beginExposureNr, floatToUint(biasMapLeft, "biasMapLeft"));
	    hdf5File.writeArray("/BiasMapsRight", "bias", exposureNr-beginExposureNr, floatToUint(biasMapRight, "biasMapRight"));
	}
    }

    if (writeThroughputMaps)
    {
      hdf5File.writeArray("/ThroughputMaps", "throughput", exposureNr-beginExposureNr, throughputMap);
    }
}











/**
 * \brief Write the EOL and BOL trap density maps of each species to the HDF5 file.
 *
 */

void Detector::writeCTIToHDF5()
{
    stringstream myStream;

    if (writeCTI && (CTImodel == "Short2013"))
    {

        // FIXME: This informational log statement is not visible in the log file (low priority)

        Log.info("Detector: writing BOL and EOL trap density maps to HDF5 file");

        for (int k = 0; k < numTrapSpecies; k++)
        {
            // First the map for BOL

            myStream << "trapDensityMapForSpecies" << k << "BOL";
            string mapName = myStream.str();

            arma::Mat<float> trapDensityMap = meanTrapDensityBOL[k] * radiationMap;
            hdf5File.writeArray("/CTI", mapName, trapDensityMap);

            // Clear the string stream and compose the EOL trap density map name

            myStream.str(string());      // insert empty string
            myStream.clear();            // clear eof bit
            myStream << "trapDensityMapForSpecies" << k << "EOL";
            mapName = myStream.str();

            // Save the EOL trap density map

            trapDensityMap = meanTrapDensityEOL[k] * radiationMap;
            hdf5File.writeArray("/CTI", mapName, trapDensityMap);

            // Once more clear the stream to that it's ready to be be used again

            myStream.str(string());      // insert empty string
            myStream.clear();            // clear eof bit
        }
    }
}










/**
 * Creates the subgroup for cosmics in the HDF5 file. This is used to manage the
 * amount of subgroups in the HDF5 file.
 */
void Detector::makeSubGroupForCosmics(string field, int exposureNr)
{
    // Create sub group that we need to create
    stringstream subgroupStream;
    subgroupStream << "/exposure" << setfill('0') << setw(3) << exposureNr / 1000;
    string subgroupName = "/Cosmics/" + field + subgroupStream.str();
    hdf5File.createGroup(subgroupName);
}









/**
 * Writes the colum, row and flux values of cosmics to the HDF5 file. This function
 * calls HDF5File::writeCosmicsWhithoutGroupByExposure, if cosmics is included
 * in the repective Field.
 *
 * /params exposureNr:   Sequential number of the exposure
 * /note: This function gets called when groupByExposure is true.
 */
void Detector::writeCosmicHitsToHDF5(int exposureNr)
{

   if (includeCosmicsInSubField && writeCosmics)
   {
       hdf5File.writeCosmics(subFieldCosmics,
			     exposureNr,
			     cosmicEntryRowSubfield,
			     cosmicEntryColSubfield,
			     cosmicsTrailsSubfield,
			     cosmicsAnglesSubfield,
			     cosmicsIntensitiesSubfield,
			     rowsOfCosmicsInSubField,
			     columnsOfCosmicsInSubField,
			     fluxOfCosmicsInSubField);
   }
}













/**
 * \brief Returns the current temperature of the detector.
 */
double Detector::getTemperature()
{
    return temperatureGenerator.getNextTemperature(internalTime);
}












/**
 * \brief Returns the duration of the readout before the next exposure can start [s].
 */
double Detector::getReadoutTimeBeforeNextExposure()
{
    return readoutTimeBeforeNextExposure;
}





/**
 *
 * \brief: Initializes the background map. This is done only once.
 *
 * \note:  The flux stored in the array does not take transmission efficiency into account.
 *         To obtain the flux from the stellar background this map should still be
 *         multiplied by the transmission efficiency. (this is exposure dependent)
 *
 */
void Detector::fillBackgroundMap(Camera &camera, double startTime, double exposureTime)
{

    // For each pixel in the background map (same dimensions as as subfield)
    for (int row=0; row<numRowsPixelMap; row++)
    {
        for (int col=0; col<numColumnsPixelMap; col++)
        {
            // Convert the pixel coordinates to focal plane coordinates
            double xFPd , yFPd ;
            double xFPmm, yFPmm;
            tie(xFPd, yFPd) = pixelToFocalPlaneCoordinates(row+subFieldZeroPointRow, col+subFieldZeroPointColumn);

            // Apply the inverse distortion
            if (includeFieldDistortion)
            {
                tie(xFPmm, yFPmm) = camera.distortedToUndistortedFocalPlaneCoordinates(xFPd, yFPd);
            }
            else
            {
                xFPmm = xFPd;
                yFPmm = yFPd;
            }

            transmissionEfficiencyBOS = camera.getTransmissionEfficiency(startTime);
            double flux = camera.getBackgroundFlux(xFPmm, yFPmm, *this, startTime, exposureTime, readoutTimeBeforeNextExposure)/transmissionEfficiencyBOS;

            backgroundMap(row, col) = flux;

        }
    }

}





/**
 *
 * \brief: Adds the straylight in electrons to the pixelmap.  
 * 
 */
void Detector::addStraylightToPixelMap(double time)
{
    double light = (*straylight).getStrayLightMoon(time);

    pixelMap += light;  // [electrons]
    straylightValues.push_back(light);


};





/**
 *
 * \brief: Converts the input map to a 2-byte unsinged short
           Includes a safety check, first check that the extrema
	   of the map are indeed within the boundaries of such a data type.
 * 
 */
arma::Mat<uint16_t> floatToUint(arma::Mat<float> &A, string mapName)
{
    // Write the input map as 2-byte (16 bit) unsigned short integers.
    // As a safety check, first check that the extrema of the map are indeed
    // within the boundaries of such a data type.

    if((A.min() < 0) || (A.max() >= (1 << 16)))
    {
        throw ConfigurationException("Detector: quantisation was applied but " + mapName + " values are not in [0, 2^16[");
    }

    // Convert the float matrix to an unsigned uint16_t matrix

    arma::Mat<uint16_t> uintMap = arma::conv_to<arma::Mat<uint16_t>>::from(A);
    return uintMap;
}

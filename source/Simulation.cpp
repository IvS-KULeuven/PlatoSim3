/**
 * \class Simulation
 *
 * \brief The starting point for any simulation.
 *
 */

#include "Simulation.h"


/**
 * \brief      Constructor
 *
 * \details
 *
 * The constructor reads the YAML input file, and creates the HDF5 output file.
 * Based on the user input a Jitter generator is created and all spacecraft
 * components are initialized.
 *
 * \param[in]  inputFilename   the YAML input file
 * \param[in]  outputFilename  the HDF5 output file
 */

Simulation::Simulation(string inputFilename, string outputFilename)
{
    // Parse the configuration parameters file
    Log.info("Simulation: reading the input parameters file");

    ConfigurationParameters configParams(inputFilename);


    // Set the random seeds of the simulation. Seeds are set in the input yaml file using long integers.
    // If they are set to -1, the following functions resets them using the system clock. This is useful
    // when the simulated time series is partitioned in segments so that each segment has a different
    // seed. The seeds that are actually used are written to the HDF5 file.

    setRandomSeeds(configParams);

    // Configure the Simulation object using the configuration parameters file

    configure(configParams);

    // Check if the output HDF5 filename already exists. If so, complain.

    if (fileExists(outputFilename))
    {
        Log.error("Simulation: Output file " + outputFilename + " already exists. Aborting.");
        exit(1);
    }

    // Depending on whether or not PlatoSim is used in a network environment create a new abstract detector factory

    if (sendImagettesToClient || getWindowPositionFromServer)
    {
        detectorFactory = new ClosedLoopDetectorFactory;

        Log.info("Simulation: create a connected detector factory instance");

        // In the closed-loop case, data is sent over the network and normally not stored locally
        if (configParams.nodeExists("ControlHDF5Content/GenerateRealHdf5") && 
                configParams.getBoolean("ControlHDF5Content/GenerateRealHdf5"))
        {
            // create a regular HDF5 file only if configured to do so
            hdf5File = new HDF5File();
        } 
        else
        {
            // create a specific empty hdf5 output file otherwise
            hdf5File = new ClosedLoopHDF5File();
        }

    }
    else
    {
        detectorFactory = new DetectorFactory;

        hdf5File = new HDF5File();
    }

    // Open the HDF5 output file where the images will be written

    hdf5File->open(outputFilename);

    // Write the version info to the output HDF5 file

    hdf5File->writeVersionInformation();

    // Acquisition times
    
    double readoutTimeDuringNextExposure;
    tie(readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure) = configureReadoutTime(configParams);
    exposureTime = cycleTime - readoutTimeBeforeNextExposure;
    if (cycleTime < readoutTimeBeforeNextExposure)
      {
	Log.warning("Simulation: exposure time is negative value: " + to_string(exposureTime));
      }

    Log.debug("Simulation: Cycle time: " + to_string(cycleTime));
    Log.debug("Simulation: Exposure time: " + to_string(exposureTime));
    Log.debug("Simulation: Readout time before next exposure: " + to_string(readoutTimeBeforeNextExposure));

    // Depending on what the user requested, define the proper platform jitter generator

    if (!useJitter)
    {
        jitterGenerator = new NoJitter();
    }
    else
    {
        if (jitterSource == "FromFile")
        {
            jitterGenerator = new JitterFromFile(configParams);
        }
        else if (jitterSource == "FromRedNoise")
        {
            jitterGenerator = new JitterFromRedNoise(configParams);
        }
        else if (jitterSource == "FromNetwork")
        {
            jitterGenerator = new JitterFromNetwork(configParams);
        }
        else
        {
            string errorMessage = "Simulation: Jitter Source '" + jitterSource + "' is not supported.";

            Log.error(errorMessage);

            throw IllegalArgumentException(errorMessage);
        }

    }

    // Depending on what the user requested, define the proper telescope thermo-elastic drift generator

    if (!useDrift)
    {
        driftGenerator = new NoDrift();
    }
    else
    {
        if (driftSource == "FromFile")
        {
            driftGenerator = new ThermoElasticDriftFromFile(configParams);
        }
        else if (driftSource == "FromRedNoise")
        {
            driftGenerator = new ThermoElasticDriftFromRedNoise(configParams);
        }
	else
        {
            string errorMessage = "Simulation: Drift Source '" + driftSource + "' is not supported.";
            Log.error(errorMessage);
            throw IllegalArgumentException(errorMessage);
        }
    }


    if(useFeeTemperatureFromFile)
    {
    	feeTemperatureGenerator = new TemperatureFromFile(configParams, "FEE");
    }
    else if(useFeeNominalTemperature)
    {
    	feeTemperatureGenerator = new NominalTemperature(configParams, "FEE");
    }

    if(useDetectorTemperatureFromFile)
    {
    	detectorTemperatureGenerator = new TemperatureFromFile(configParams, "CCD");
    }
    else if(useDetectorNominalTemperature)
    {
    	detectorTemperatureGenerator = new NominalTemperature(configParams, "CCD");
    }

    // Initialise the spacecraft components

    platform   = new Platform(configParams, *hdf5File, *jitterGenerator);
    telescope  = new Telescope(configParams, *hdf5File, *platform, *driftGenerator);
    sky        = new Sky(configParams);
    camera     = new Camera(configParams, *hdf5File, *platform, *telescope, *sky);


    // Depending on how the PSF is computed (analytically or pre-mapped) the Detector object is different.

    if (psfModel == "MappedFromFile")
    {
        detector = detectorFactory->createDetectorWithAsymmetricalMappedPsfInstance(configParams, *hdf5File, *camera, *feeTemperatureGenerator, *detectorTemperatureGenerator, readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure);
    }
    else if (psfModel == "AnalyticGaussian")
    {
        detector = detectorFactory->createDetectorWithAnalyticGaussianPsfInstance(configParams, *hdf5File, *camera, *feeTemperatureGenerator, *detectorTemperatureGenerator, readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure);
    }
    else if (psfModel == "AnalyticNonGaussian")
    {
        detector = detectorFactory->createDetectorWithAnalyticNonGaussianPsfInstance(configParams, *hdf5File, *camera, *feeTemperatureGenerator, *detectorTemperatureGenerator, readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure);
    }
    else
    {
        string errorMessage = "Simulation: PSF Model '" + psfModel + "' is not supported.";
        Log.error(errorMessage);
        throw IllegalArgumentException(errorMessage);
    }

    // Write the input parameters to the output HDF5 file

    writeInputParametersToHDF5(configParams);

}







/**
 * \brief      Destructor, release memory of all spacecraft components
 */
Simulation::~Simulation()
{
    // Delete order is the inverse of the order in which they were created

    delete detector;
    delete camera;
    delete telescope;
    delete sky;
    delete platform;
    delete jitterGenerator;
    delete driftGenerator;

    // Close the output hdf5 file

    hdf5File->close();

    delete hdf5File;
}








/**
 * \brief Configure the Simulation object using the input parameter file
 *
 * \param configParams  Contains all configuration parameters from the input file
 */

void Simulation::configure(ConfigurationParameters &configParams)
{
    cycleTime                       = configParams.getDouble("ObservingParameters/CycleTime");
    beginExposureNr                 = configParams.getInteger("ObservingParameters/BeginExposureNr");
    numExposures                    = configParams.getInteger("ObservingParameters/NumExposures");
    useJitter                       = configParams.getBoolean("Platform/UseJitter");
    jitterSource                    = configParams.getString("Platform/JitterSource");
    includeFieldDistortion          = configParams.getBoolean("Camera/IncludeFieldDistortion");     // TODO do we want to do this or should this be asked to Camera?
    useDrift                        = configParams.getBoolean("Telescope/UseDrift");
    driftSource                     = configParams.getString("Telescope/DriftSource");
    psfModel                        = configParams.getString("PSF/Model");
    useFeeTemperatureFromFile       = configParams.getString("FEE/Temperature") == "FromFile";
    useFeeNominalTemperature        = configParams.getString("FEE/Temperature") == "Nominal";
    useDetectorTemperatureFromFile  = configParams.getString("CCD/Temperature") == "FromFile";
    useDetectorNominalTemperature   = configParams.getString("CCD/Temperature") == "Nominal";
    sendImagettesToClient           = configParams.getBoolean("ControlTcpConnection/SendImagettesToClients");
    getWindowPositionFromServer     = configParams.getBoolean("ControlTcpConnection/GetWindowPositionsFromServer");
    writeStarCatalog                = configParams.getBoolean("ControlHDF5Content/WriteStarCatalog");

    // The readout of different CCDs are shifted in time because of the power budget.
    // Find out the right time shift.

    string ccdPosition              = configParams.getString("CCD/Position");
    bool isFastCamera               = configParams.getString("Telescope/GroupID") == "Fast";
    if (ccdPosition == "Custom")
    {
        timeShift = configParams.getDouble("CCD/TimeShift");
    }
    else if (!isFastCamera)
    {
        int index = stoi(ccdPosition) - 1;   // Position are named  [1, 2, 3, 4] while the index into vector starts at 0
        timeShift = configParams.getDoubleAt("CCDPositions/TimeShift", index);
    }
    else
    {
        timeShift = 0.0;
    }

    Log.debug("Simulation: configure(): time shift for current CCD configuration: " + to_string(timeShift));
}





/**
 * \brief Determines the duration of
 *        - the readout that takes place before the next exposure starts,
 *        - and the readout that takes place during the next exposure,
 *        depending on the camera type (normal / fast) and the readout mode
 *        (nominal / partial readout).
 *
 * For the normal cameras the entire CCD is read out (with open shutter) after
 * the exposure during a time interval called 'readoutTimeBeforeNextExposure'.
 * Only after this readout, a new exposure is started.
 * For the fast camera, half of the CCD is first quickly frame-transferred,
 * after which it is read out slowly. In this case a new exposure is already
 * started after the quick frame-transfer, and starts thus during the slow readout
 * of the previous exposure.
 * Hence the need for two parameters 'readoutTimeBeforeNextExposure' and
 * 'readoutTimeDuringNextExposure'.
 *
 * \param configParams Contains all configuration parameters from the input file
 *
 * \return (readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure)
 *
 */

pair<double, double> Simulation::configureReadoutTime(ConfigurationParameters &configParams)
{

        int numRows, numColumns, firstRowExposed;
        bool isFastCamera = configParams.getString("Telescope/GroupID") == "Fast";
        string ccdPosition = configParams.getString("CCD/Position");

        if (ccdPosition == "Custom")
        {
                numRows = configParams.getInteger("CCD/NumRows");                   // [pixels]
                numColumns = configParams.getInteger("CCD/NumColumns");             // [pixels]
                firstRowExposed = configParams.getInteger("CCD/FirstRowExposed");   // [pixels]
        }

        else
        {
                int idx = stoi(ccdPosition) - 1; // Positions are named [1, 2, 3, 4] while the index into vector starts at 0

                numRows = configParams.getIntegerAt("CCDPositions/NumRows", idx);           // [pixels]
                numColumns = configParams.getIntegerAt("CCDPositions/NumColumns", idx);     // [pixels]

                isFastCamera = configParams.getString("Telescope/GroupID") == "Fast";

                if (isFastCamera)
                        firstRowExposed = configParams.getIntegerAt("CCDPositions/FirstRowForFastCamera", idx);     // [pixels]

                else
                        firstRowExposed = configParams.getIntegerAt("CCDPositions/FirstRowForNormalCamera", idx);   // [pixels]
        }

        string readoutMode = configParams.getString("CCD/ReadoutMode/ReadoutMode");

        if((readoutMode != "Nominal") && (readoutMode != "Partial"))
        {
                Log.error("Simulation::configureReadoutTime(): Unknown readout mode specification in configuration file: "  + readoutMode);
                throw ConfigurationException("Simulation: Unknown readout mode specification in configuration file");
        }

        double serialTransferTime = configParams.getDouble("CCD/SerialTransferTime") * 1E-9;		  // [ns] -> [s]
        double parallelTransferTime = configParams.getDouble("CCD/ParallelTransferTime") * 1E-6;	  // [µs] -> [s]
        double parallelTransferTimeFast = configParams.getDouble("CCD/ParallelTransferTimeFast") * 1E-6;  // [µs] -> [s]


	int numColumnsBiasMap =  configParams.getInteger("SubField/NumBiasPrescanColumns");     // [pixels]
	int numBiasPrescanRows = configParams.getInteger("SubField/NumBiasPrescanRows");        // [pixels]
	int numBiasPrescanColumns = configParams.getInteger("SubField/NumBiasPrescanColumns");  // [pixels]
	int numRowsSmearingMap = configParams.getInteger("SubField/NumSmearingOverscanRows");   // [pixels]


        double readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure;

        // Both detector halves are read out simultaneously
        // -> columns read out by the FEE:
        //              - half of the CCD
        //              - serial pre-scan
        //              - Parallel pre-scan

        int numColumnsReadout = numColumns / 2 + numColumnsBiasMap + numBiasPrescanColumns;

        // How many rows will be actually read out by the FEE?
        //      - nominal mode: image area + parallel over-scan
        //      normal camera: image area = whole CCD
        //      fast camera: image area = lower half of the CCD
        //      - partial readout: configurable
        // The rest of the image area will be dumped

        int numRowsReadout=0;
        int numRowsDump=0;



        // -----------
        // Fast camera
        // -----------

        if (isFastCamera)
    {
                // Move the upper half of the CCD down to the lower half, row-by-row

                int numRowsFrameTransfer = numRows - firstRowExposed;

                readoutTimeBeforeNextExposure = numRowsFrameTransfer * parallelTransferTimeFast;

                // The actual readout of the lower half of the CCD (after frame transfer) is done
                // while the next exposure has already started

                // Nominal mode

                if (readoutMode == "Nominal")
                {
                        numRowsReadout = firstRowExposed + numRowsSmearingMap;
                        numRowsDump = 0;

                }

                // Rows read out by the FEE: rows in the block (other rows in image area are dumped)
                // Note: no parallel over-scan

                else if (readoutMode == "Partial")
                {
                numRowsReadout = configParams.getInteger("CCD/ReadoutMode/Partial/NumRowsReadout");
                        numRowsDump = firstRowExposed - numRowsReadout;
                }

                readoutTimeDuringNextExposure = numRowsDump * parallelTransferTimeFast
                                + numRowsReadout * (parallelTransferTime + numColumnsReadout * serialTransferTime);
        }



        // -------------
        // Normal camera
        // -------------

        else
        {

                // Nominal mode (full-frame readout)

                if (readoutMode == "Nominal")
                {

                        // Rows read out by the FEE:
                        //              - rows of image area
                        //              - parallel over-scan

                        numRowsReadout = numRows + numRowsSmearingMap;

                        // No rows dumped

                        numRowsDump = 0;
                }

                // Partial readout

                else if (readoutMode == "Partial")
                {
                        // Rows read out by the FEE: rows in the block (other rows in image area are dumped)
                        // Note: no parallel over-scan
                        numRowsReadout = configParams.getInteger("CCD/ReadoutMode/Partial/NumRowsReadout");
                        numRowsDump = numRows - numRowsReadout;
                }

                readoutTimeBeforeNextExposure = numRowsDump * parallelTransferTimeFast
                                + numRowsReadout * (numColumnsReadout * serialTransferTime + parallelTransferTime);

                readoutTimeDuringNextExposure = 0;
        }

        return make_pair(readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure);
}







/**
 * \brief      Loop over all exposures
 *
 * \param[in]  startTime  begin time of the very first exposure. Time is expressed in seconds in the rest of the code.
 */

void Simulation::run()
{
    // Update the internal clock

    currentTime = beginExposureNr * (exposureTime + readoutTimeBeforeNextExposure) + timeShift;

    Log.info("Simulation: running exposures " + to_string(beginExposureNr) + " to " + to_string(beginExposureNr+numExposures-1));

    // Declare the imagetteNumber and set the endOfSimulation variable to false

    int n = beginExposureNr;

    bool endOfSimulation = false;

    // Continue the simulation until no more jittersteps are send from a tcp connection server

    while (!endOfSimulation)
    {
        // if no jitter from network is used, end the simulation, when the max number of exposures from the yaml file is reached
        if ((!useJitter || jitterSource != "FromNetwork") && n >= beginExposureNr + numExposures)
        {
            Log.info("Simulation: end of simulation reached");

            endOfSimulation = true;
        }
        else
        {
            Log.info("Simulation: Starting exposure " + to_string(n) + " at time " + to_string(currentTime));

            currentTime = detector->takeExposure(n, currentTime, exposureTime);

            endOfSimulation = jitterGenerator->getSimulationState();

            n++;
        }
    }

    if (writeStarCatalog)
    {
        writeStarCatalogToHDF5();
    }
}


















/**
 * \brief      Write information about the stars that were detected in the subField
 *             to the HDF5 output file.
 *
 * \details    The Camera collects all the stars that fall within the boundaries of the subField.
 *
 *             This function should only be called after all exposures have been taken in order
 *             to have the complete collections of stars that have been detected in the subField.
 *
 */
void Simulation::writeStarCatalogToHDF5()
{
    Log.info("Simulation: writing info on detected stars to HDF5 in /StarCatalog");

    set<unsigned int> allStarIDs = camera->getAllStarIDs();

    // For all detected stars, copy the equatorial sky coordinates and the magnitude
    // from the user-given star catalog to the output HDF5 file in a custom group.

    hdf5File->createGroup("/StarCatalog");
    const int Nstars = allStarIDs.size();
    vector<unsigned int> starIDs(Nstars);    // set<> is not contiguous, vector<> is. Needed for HDF5.
    vector<double> RA(Nstars);
    vector<double> dec(Nstars);
    vector<double> Vmag(Nstars);
    vector<double> rowPix(Nstars);
    vector<double> colPix(Nstars);
    vector<double> xFPmm(Nstars);
    vector<double> yFPmm(Nstars);

    double xFPrad, yFPrad;

    if (!allStarIDs.empty())
    {
        int k = 0;
        for (auto starID: allStarIDs)
        {
            starIDs[k] = starID;
            tie(RA[k], dec[k], Vmag[k]) = sky->getInfoOfStarWithID(starID);  // RA & dec returned in radians!
            const bool useInitialOrientation = true;
            tie(xFPmm[k], yFPmm[k]) = camera->skyToFocalPlaneCoordinates(RA[k], dec[k], useInitialOrientation);

            RA[k]  *= Angle::degrees;    // [rad] -> [deg]
            dec[k] *= Angle::degrees;    // [rad] -> [deg]

            if (includeFieldDistortion)
            {
              if(psfModel == "MappedFromFile")
              {
                detector->applyDistortion(xFPmm[k], yFPmm[k]);
              }
              else
              {
                tie(xFPmm[k], yFPmm[k]) = camera->undistortedToDistortedFocalPlaneCoordinates(xFPmm[k], yFPmm[k]);
              }
            }

            tie(rowPix[k], colPix[k]) = detector->focalPlaneToPixelCoordinates(xFPmm[k], yFPmm[k]);
            k++;
        }
        hdf5File->writeArray("StarCatalog/", "starIDs", starIDs.data(), starIDs.size());
        hdf5File->writeArray("StarCatalog/", "RA",      RA.data(),      RA.size());
        hdf5File->writeArray("StarCatalog/", "Dec",     dec.data(),     dec.size());
        hdf5File->writeArray("StarCatalog/", "Vmag",    Vmag.data(),    Vmag.size());
        hdf5File->writeArray("StarCatalog/", "xFPmm",   xFPmm.data(),   xFPmm.size());
        hdf5File->writeArray("StarCatalog/", "yFPmm",   yFPmm.data(),   yFPmm.size());
        hdf5File->writeArray("StarCatalog/", "colPix",  colPix.data(),  colPix.size());
        hdf5File->writeArray("StarCatalog/", "rowPix",  rowPix.data(),  rowPix.size());
    }
    else
    {
        Log.warning("Simulation: no information about detected stars to write to HDF5");
    }
}











/**
 * \brief [brief description]
 *
 */

void Simulation::writeInputParametersToHDF5(ConfigurationParameters &configParams)
{
    Log.info("Simulation: writing input parameters to HDF5");

    // Make the parent group

    string parentGroup = "/InputParameters";

    hdf5File->createGroup(parentGroup);

    string subGroup;

    // Define some Lambda functions that will make it much easier to add the input parameters

    auto addDouble = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getDouble(subGroup + "/" + attributeName));
    };

    auto addInteger = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getInteger(subGroup + "/" + attributeName));
    };

    auto addLong = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getLong(subGroup + "/" + attributeName));
    };

    auto addString = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getString(subGroup + "/" + attributeName));
    };

    auto addBoolean = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getBoolean(subGroup + "/" + attributeName));
    };

    auto addDoubleVector = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getDoubleVector(subGroup + "/" + attributeName));
    };

    auto addIntegerVector = [&] (string attributeName)
    {
        hdf5File->writeAttribute(parentGroup + "/" + subGroup, attributeName, configParams.getIntegerVector(subGroup + "/" + attributeName));
    };


    // Copy the input parameters to the output HDF5 file

    subGroup = "ObservingParameters";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("MissionDuration");
    addInteger("NumExposures");
    addInteger("BeginExposureNr");
    addDouble("CycleTime");
    addDouble("Fluxm0");
    addString("StarCatalogFile");

    subGroup = "Sky";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("IncludeVariableSources");
    addString("VariableSourceList");
    addBoolean("IncludeCosmicsInSubField");
    addBoolean("IncludeCosmicsInSmearingMap");
    addBoolean("IncludeCosmicsInBiasMap");
    subGroup = "Sky/SkyBackground";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("UseConstantSkyBackground");
    addDouble("BackgroundValue");
    subGroup = "Sky/Cosmics";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("CosmicHitRate");
    addDoubleVector("TrailLength");
    addDoubleVector("Intensity");

    subGroup = "Platform";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("UseJitter");
    addString("JitterSource");
    addDouble("JitterYawRms");
    addDouble("JitterPitchRms");
    addDouble("JitterRollRms");
    addDouble("JitterTimeScale");
    addString("JitterFileName");
    subGroup = "Platform/Orientation";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    subGroup = "Platform/Orientation/Angles";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("RAPointing");
    addDouble("DecPointing");
    addDouble("SolarPanelOrientation");
    subGroup = "Platform/Orientation/Quaternion";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDoubleVector("Components");


    subGroup = "Telescope";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("GroupID");
    addDouble("AzimuthAngle");
    addDouble("TiltAngle");
    addDouble("LightCollectingArea");
    addBoolean("UseDrift");
    addString("DriftSource");
    addDouble("DriftYawRms");
    addDouble("DriftPitchRms");
    addDouble("DriftRollRms");
    addDouble("DriftTimeScale");
    addString("DriftFileName");
    subGroup = "Telescope/TransmissionEfficiency";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("BOL");
    addDouble("EOL");

    subGroup = "Camera";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("PlateScale");
    addDouble("ThroughputBandwidth");
    addDouble("ThroughputLambdaC");
    addBoolean("IncludeAberrationCorrection");
    addBoolean("IncludeFieldDistortion");
    addBoolean("IncludePointLikeGhosts");
    addBoolean("IncludeExtendedGhosts");
    subGroup = "Camera/FocalPlaneOrientation";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    addDouble("ConstantValue");
    addString("FromFile");
    subGroup = "Camera/FocalLength";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    addDouble("ConstantValue");
    addString("FromFile");
    subGroup = "Camera/AberrationCorrection";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Type");
    addString("OrbitFile");
    addDouble("StartTime");
    subGroup = "Camera/FieldDistortion";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    addDoubleVector("ConstantCoefficients");
    addDoubleVector("ConstantInverseCoefficients");
    addString("CoefficientsFromFile");
    addString("InverseCoefficientsFromFile");
    subGroup = "Camera/Ghosts";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    subGroup = "Camera/Ghosts/PointLike";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("FluxRatio");
    addDouble("DistanceCutOff");
    subGroup = "Camera/Ghosts/Extended";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("FluxRatio");
    addDouble("DistanceRatio");
    addDoubleVector("RadiusCoefficients");

    subGroup = "PSF";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Model");
    subGroup = "PSF/MappedFromFile";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Filename");
    addInteger("NumberOfPixels");
    addDouble("ChargeDiffusionStrength");
    addBoolean("IncludeChargeDiffusion");
    addBoolean("IncludeJitterSmoothing");
    subGroup = "PSF/AnalyticGaussian";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("Sigma00");
    addDouble("SigmaX18");
    addDouble("SigmaY18");
    subGroup = "PSF/AnalyticNonGaussian";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("ParameterFileName");
    addDouble("ChargeDiffusionStrength");
    addBoolean("IncludeChargeDiffusion");
    subGroup = "PSF/AnalyticNonGaussian/Sigma";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    addDouble("ConstantValue");
    addString("FromFile");

    subGroup = "FEE";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("NominalOperatingTemperature");
    addString("Temperature");
    addString("TemperatureFileName");
    addDouble("ReadoutNoise");
    addBoolean("IncludeOverAndUnderShoot");
    subGroup = "FEE/Gain";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("RefValueLeft");
    addDouble("RefValueRight");
    addDouble("Stability");
    addDouble("AllowedDifference");
    subGroup = "FEE/ElectronicOffset";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addInteger("RefValue");
    addDouble("Stability");
    subGroup = "FEE/OverAndUnderShoot";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("Strength");
    addDouble("DecaySpeed");
    addDouble("DecayRate");
    addInteger("Range");

    subGroup = "CCD";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Position");
    addDouble("OriginOffsetX");
    addDouble("OriginOffsetY");
    addDouble("Orientation");
    addInteger("NumColumns");
    addInteger("NumRows");
    addInteger("FirstRowExposed");
    addDouble("TimeShift");
    addDouble("PixelSize");
    addLong("FullWellSaturation");
    addInteger("DigitalSaturation");
    addDouble("ReadoutNoise");
    addDouble("SerialTransferTime");
    addDouble("ParallelTransferTime");
    addDouble("ParallelTransferTimeFast");
    addDouble("NominalOperatingTemperature");
    addString("Temperature");
    addString("TemperatureFileName");
    addBoolean("IncludeFlatfield");
    addBoolean("IncludeDarkSignal");
    addBoolean("IncludeBFE");
    addBoolean("IncludePhotonNoise");
    addBoolean("IncludeReadoutNoise");
    addBoolean("IncludeCTIeffects");
    addBoolean("IncludeChargeInjection");
    addBoolean("IncludeOpenShutterSmearing");
    addBoolean("IncludeQuantumEfficiency");
    addBoolean("IncludeRelativeTransmissivity");
    addBoolean("IncludePolarization");
    addBoolean("IncludeParticulateContamination");
    addBoolean("IncludeMolecularContamination");
    addBoolean("IncludeConvolution");
    addBoolean("IncludeFullWellSaturation");
    addBoolean("IncludeDigitalSaturation");
    addBoolean("IncludeQuantisation");
    addBoolean("IncludeGainNonlinearity");
    subGroup = "CCD/Flatfield";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Source");
    subGroup = "CCD/Flatfield/FromFile";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("FilePath");
    subGroup = "CCD/Flatfield/FromRedNoise";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("FlatfieldNoiseRMS");
    subGroup = "CCD/BFE";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("CoefficientsFileName");
    subGroup = "CCD/Gain";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("RefValueLeft");
    addDouble("RefValueRight");
    addDouble("Stability");
    addDouble("AllowedDifference");
    addDoubleVector("Nonlinearity");
    subGroup = "CCD/QuantumEfficiency";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("MeanQuantumEfficiency");
    addDouble("MeanAngleDependency");
    subGroup = "CCD/ReadoutMode";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("ReadoutMode");
    subGroup = "CCD/ReadoutMode/Partial";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addInteger("FirstRowReadout");
    addInteger("NumRowsReadout");
    subGroup = "CCD/RelativeTransmissivity";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDoubleVector("Coefficients");
    addDouble("RadiusFOV");
    addDouble("ExpectedValue");
    subGroup = "CCD/Polarization";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("ExpectedValue");
    subGroup = "CCD/Contamination";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("ParticulateContaminationEfficiency");
    addDouble("MolecularContaminationEfficiency");
    subGroup = "CCD/DarkSignal";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("DarkCurrent");
    addDouble("DSNU");
    addDouble("Stability");
    subGroup = "CCD/CTI";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addString("Model");
    subGroup = "CCD/CTI/Simple";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("MeanCTE");
    subGroup = "CCD/CTI/Short2013";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("Beta");
    addDouble("Temperature");
    addInteger("NumTrapSpecies");
    addDoubleVector("TrapCaptureCrossSection");
    addDoubleVector("ReleaseTime");
    subGroup = "CCD/CTI/Short2013/TrapDensity";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDoubleVector("BOL");
    addDoubleVector("EOL");
    subGroup = "CCD/ChargeInjection";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDouble("InjectionLevel");
    addInteger("RowInterval");
    addInteger("FirstRow");

    subGroup = "SubField";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addInteger("ZeroPointRow");
    addInteger("ZeroPointColumn");
    addInteger("NumColumns");
    addInteger("NumRows");
    addInteger("NumBiasPrescanRows");
    addInteger("NumBiasPrescanColumns");
    addInteger("NumSmearingOverscanRows");
    addInteger("SubPixels");

    subGroup = "Photometry";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("IncludePhotometry");
    addInteger("ContaminationRadius");
    addDouble("MaskUpdateInterval");
    addString("TargetFileName");

    subGroup = "RandomSeeds";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addLong("ReadOutNoiseSeed");
    addLong("PhotonNoiseSeed");
    addLong("JitterSeed");
    addLong("FlatFieldSeed");
    addLong("DriftSeed");
    addLong("CosmicSeed");
    addLong("DarkSignalSeed");

    subGroup = "ControlHDF5Content";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("WritePixelMaps");
    addBoolean("WriteBiasMaps");
    addBoolean("WriteSmearingMaps");
    addBoolean("WriteThroughputMaps");
    addBoolean("WriteFlatfieldMap");
    addBoolean("WriteSubPixelImages");
    addBoolean("WriteStarPositions");
    addBoolean("GroupByExposure");
    addBoolean("WriteGhostPositions");
    addBoolean("WriteACS");
    addBoolean("WriteCosmics");
    addBoolean("WriteDiffusedPSF");
    addBoolean("WriteHighResolutionPSF");
    addBoolean("WriteStarCatalog");
    addBoolean("WriteTelescopeACS");
    addBoolean("WriteTransmissionEfficiency");

    subGroup = "ControlTcpConnection";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("SendImagettesToClients");
    addBoolean("GetWindowPositionsFromServer");
    addString("WindowPositionServerAddress");
    addString("JitterServerAddress");
    addString("ImagetteClientAddress");
    addInteger("WindowPositionSocketTimeout");
    addInteger("JitterSocketTimeout");

    subGroup = "CameraGroups";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addDoubleVector("AzimuthAngle");
    addDoubleVector("TiltAngle");

    subGroup = "CCDPositions";
    hdf5File->createGroup(parentGroup + "/" + subGroup);
    addBoolean("UsePositionsFromFile");
    addString("PositionsFileName");
    addDoubleVector("OriginOffsetX");
    addDoubleVector("OriginOffsetY");
    addDoubleVector("Orientation");
    addIntegerVector("NumColumns");
    addIntegerVector("NumRows");
    addIntegerVector("FirstRowForNormalCamera");
    addIntegerVector("FirstRowForFastCamera");
    addDoubleVector("TimeShift");
}






/**
 * @brief Set the random seeds of the simulation.
 *
 * If the user set a random seed to -1, use the system's clock to set it. This is useful
 * to simulate time series that were partitioned in several segments, so that each
 * segment has a different random seed.
 *
 * @param configParams
 *
 * @return configParams will have adapted seeds if they were set to -1.
 */

void Simulation::setRandomSeeds(ConfigurationParameters &configParams)
{
    vector<string> seedNames = configParams.getKeys("RandomSeeds");

    for (unsigned int n = 0; n < seedNames.size(); n++)
    {
        const string seedPath = "RandomSeeds/" + seedNames[n];
        long originalSeed = configParams.getLong(seedPath);
        if (originalSeed == -1)
        {
            long newSeed = long(std::time(nullptr) + n * 11111);
            configParams.setParameter(seedPath, to_string(newSeed));
        }
    }
}


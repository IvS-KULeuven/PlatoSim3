/**
 * \class PointSpreadFunction
 *
 * \brief Base class for the PSF, both symmetrical and asymmetrical.
 */

#include "PointSpreadFunction.h"





/**
 * \brief Constructor.
 *
 * \details Initialises the groups in the HDF5 file where the different maps (i.e. pixel map,
 *          bias register map, smearing map, etc.) will be saved.
 *
 * The following maps are initialized to zero (partly through the base class Detector):
 *      - pixelMap
 *      - subPixelMap
 *      - biasMap
 *      - smearingMap
 *      - flatfieldMap
 *      - throughputMap
 *      - cteMap
 *
 * The flatfieldMap is filled at sub-pixel level, the throughputMap and cteMap are filled at pixel level.
 *
 * \param configParam: Configuration parameters for the detector.
 * \param hdf5file:HFD5 file to write the detector images to.
 * \param camera: Camera to which to attach the detector.
 * \param readoutTimeBeforeNextExposure Duration of the readout that takes place before the next exposure can start [s].
 */
PointSpreadFunction::PointSpreadFunction(ConfigurationParameters &configParam, HDF5File &hdf5file) : HDF5Writer(hdf5file)
{
    // Parse the parameters from the configuration file.

    configure(configParam);

    isSelected = false;
    isRotated  = false;

    // Create the groups in the HDF5 file where the different PSFs and their description will be saved.

    initHDF5Groups();
    
    // Prepare the psfFile by performing some basic checks

    if (!FileUtilities::fileExists(absolutePath))
    {
        throw FileException("PointSpreadFunction: trying to load the PSF HDF5 file (" + absolutePath + "), but file doesn't exist.");
    }

    try
    {
        psfFile.open(absolutePath);
    }
    catch (H5::FileIException ex)
    {
        Log.error("H5::FileIException: " + string(ex.getCDetailMsg()));
        throw H5FileException("PointSpreadFunction: Could not open HDF5 file: " + absolutePath);
    }

    Log.info("PointSpreadFunction: Opened the HDF5 file " + absolutePath + " containing the PSFs");
}





/**
 * \brief Destructor.
 *
 * \details Closes the HDF5 file and releases the memory.
 */
PointSpreadFunction::~PointSpreadFunction()
{
    flushOutput();
    psfFile.close();
}





/**
 * \brief Configures the PointSpreadFunction object using the given
 *        configuration parameters
 *
 * \param[in] configParam: Configuration parameters.
 */
void PointSpreadFunction::configure(ConfigurationParameters &configParam)
{
    string model = configParam.getString("PSF/Model");

    if (model == "MappedFromFile")
    {
        // The user specified to use the pre-calculated PSFs from file
        // The number of sub-pixels per pixel is derived from the number of pixels specified
        // and the size of the array in the file. (The number of sub-pixels should be in the file).

        absolutePath           = configParam.getAbsoluteFilename("PSF/MappedFromFile/Filename");
        numberOfPixels         = configParam.getInteger("PSF/MappedFromFile/NumberOfPixels");
	includeChargeDiffusion = configParam.getBoolean("PSF/MappedFromFile/IncludeChargeDiffusion");
	writeHighResolutionPSF = configParam.getBoolean("ControlHDF5Content/WriteHighResolutionPSF");
	writeDiffusedPSF       = configParam.getBoolean("ControlHDF5Content/WriteDiffusedPSF");
    }
    else
    {
        string errorMessage = "PointSpreadFunction: Model '" + model + "' is not supported.";
        Log.error(errorMessage);
        throw IllegalArgumentException(errorMessage);
	writeHighResolutionPSF = false;
    }

    string focalLengthSource = configParam.getString("Camera/FocalLength/Source");
    if (focalLengthSource == "ConstantValue")
    {
        focalLengthValue = configParam.getDouble("Camera/FocalLength/ConstantValue") * 1000;     // [m] -> [mm]
    }
    else if (focalLengthSource == "FromFile")
    {
        string focalLengthInputFile = configParam.getAbsoluteFilename("Camera/FocalLength/FromFile");
        Parameter<double> *focalLength = new Parameter<double>(focalLengthInputFile, 1000);             // [m] -> [mm]
        focalLengthValue = (*focalLength)();
    }
    Log.info("PointSpreadFunction: Using a focal length: " + to_string(focalLengthValue) + " mm");
}



/**'
 * \brief Creates the group(s) in the HDF5 file where the PSF information will be stored.
 *        These group(s) has (have) to be created once, at the very beginning.
 */
void PointSpreadFunction::initHDF5Groups()
{
  if (writeHighResolutionPSF || (writeDiffusedPSF && includeChargeDiffusion))
    {
      Log.debug("PointSpreadFunction: initialising HDF5 groups");
      hdf5File.createGroup("/PSF");
    }  
}





/**
 * \brief Writes all recorded and remaining information to the HDF5 output file.
 */
void PointSpreadFunction::flushOutput()
{
    Log.info("PointSpreadFunction: Flushing output to HDf5 file.");

    if (!isSelected)
        return;

    // Save the PSF subpixel map when it is rebinned to pixel level.

    if (writeHighResolutionPSF || (writeDiffusedPSF && includeChargeDiffusion))
      {
	rebinToPixels();
      }
}




/**
 * \brief Selects the proper PSF matching the given focal-plane coordinates closest.  The appropriate PSF will be
 *        selected, i.e. the PSF for which the focal-plane position matches best.
 *
 * \param[in] xFP: Focal-plane x-coordinate [mm].
 *
 * \param[in] yFP: Focal-plane y-coordinate [mm].
 */
void PointSpreadFunction::select(double xFP, double yFP)
{
    using StringUtilities::dtos;

    // TODO: Should we take any action if different PSFs are selected for this object?

    if (isSelected)
    {
        Log.warning("PointSpreadFunction: Another PSF was previously selected.");
    }

    unsigned int datasetIndex = 1;
    string datasetName;
    double xPsf, yPsf, distanceSquared;
    double minDistanceSquared = std::numeric_limits<double>::max();
    string selectedDatasetName;

    while(true)
    {
        datasetName = to_string(datasetIndex);

        if(psfFile.hasDataset("/", datasetName))
        {
            // The following two attributes are stored as strings in the HDF5 file rather than doubles,
            // so we need to convert them.

            xPsf = stod(psfFile.readStringDatasetAttribute("/", datasetName, "centerCoordinates1"));
            yPsf = stod(psfFile.readStringDatasetAttribute("/", datasetName, "centerCoordinates2"));

            distanceSquared = pow(xPsf - xFP, 2) + pow(yPsf - yFP, 2);

            if(distanceSquared < minDistanceSquared)
            {
                minDistanceSquared = distanceSquared;
                selectedDatasetName = datasetName;
            }
        }
        else
        {
            break;
        }

        datasetIndex++;
    }

    psfFile.readArray("/", selectedDatasetName, psfMap);

    rotationAngle = 0.0;

    // We should be able to read the number of sub-pixels per pixel that was used to generate the PSFs
    // from an attribute in the HDF5 file. Unfortunately, this is not available and we therefore derive
    // the number from the array size of the psfMap and the number of pixels, currently specified
    // in the input file.

    numberOfSubPixelsPerPixel = psfMap.n_rows / numberOfPixels;

    if (writeHighResolutionPSF || (writeDiffusedPSF && includeChargeDiffusion))
      {
	hdf5File.writeAttribute("/PSF", "selectedPSF", "Realistic PSF selected from dataset " + datasetName + ".");
      }
    
    isSelected = true;
    initializeDistortionMap();
}






/**
 * \brief Rotates the PSF with the given angle.  Beware that the PSF that has been provided is already
 *        rotated, this will be taken into account.
 *
 * \param[in] angle: Angle by which the PSF should be rotated [radians].
 */
void PointSpreadFunction::rotate(double angle)
{
    if (isRotated)
    {
        Log.warning("PointSpreadFunction: Ignoring rotation: PSF was already rotated before.");
    }
    else
    {
        // This part of the code currently assumes that the PSF has not been rotated yet.
        // The rotationAngle of the generated PSFs is 45 degrees, so the actual rotation should be
        // the requested angle minus the rotationAngle of the PSF.

        psfMap = ArrayOperations::rotateArray(psfMap, angle);

        rotationAngle = angle;
        isRotated = true;

        psfMap /= arma::accu(psfMap);

        Log.debug("PointSpreadFunction: rotated current PSF over angle " + to_string(rad2deg(angle)) + " deg");

        // Write the psfMap of the rotated PSF to the HDF5 output file
	
	if (writeHighResolutionPSF)
	{
	  hdf5File.writeArray("/PSF", "highResPSF", psfMap);
	  hdf5File.writeAttribute("/PSF", "rotationAngle", rotationAngle);
	}
    }
}


/** \brief Rebins the PSF sub-pixel map to pixel level.
 *        This method does not change the psfMap of the PointSpreadFunction class.
 *
 * \param[in] targetPixels: Target number of pixels.
 *
 * \return Rebinned PSF map.
 *
 */
arma::fmat PointSpreadFunction::rebinToPixels()
{
    unsigned int binSize = psfMap.n_rows / numberOfSubPixelsPerPixel;
    arma::fmat rebinnedMap = ArrayOperations::rebin(psfMap, binSize, binSize);
    isRebinned = true;
    rebinnedMap /= arma::accu(rebinnedMap);

    // Write the rebinned PSF to the output HDF5 file

    if (writeHighResolutionPSF || (writeDiffusedPSF && includeChargeDiffusion))
      {
	hdf5File.writeArray("/PSF", "rebinnedPSFpixel", rebinnedMap);
      }

    return rebinnedMap;
}





/**
 * \brief Rebins the PSF map to the target number of sub-pixels. The number of sub-pixels used
 *        to generate the PSF is not necessarily the same as the number of sub-pixels per pixel
 *        for the detector. So the PSF needs to be rebinned to the number of sub-pixels per pixel
 *        for the detector, which is the specified target number of sub-pixels.
 *        This method does not change the psfMap of the PointSpreadFunction class.
 *
 * \param[in] targetSubPixels: Target number of sub-pixels (i.e. after rebinning).
 *
 * \return Rebinned PSF map (with the given number of sub-pixels).
 */
arma::fmat PointSpreadFunction::rebinToSubPixels(unsigned int targetSubPixels)
{
    if (targetSubPixels == numberOfSubPixelsPerPixel)
        return psfMap;

    unsigned int binSize = psfMap.n_rows / numberOfSubPixelsPerPixel * targetSubPixels;

    arma::fmat rebinnedMap = ArrayOperations::rebin(psfMap, binSize, binSize);

    isRebinned = true;

    rebinnedMap /= arma::accu(rebinnedMap);

    // Write the rebinned PSF to the output HDF5 file

    if (writeHighResolutionPSF || (writeDiffusedPSF && includeChargeDiffusion))
      {
	hdf5File.writeArray("/PSF", "rebinnedPSFsubPixel", rebinnedMap);
      }

    return rebinnedMap;
}







/*
 * This function gets the selected psf
 */

arma::fmat PointSpreadFunction::getOriginalPSF()
{
  psfMap = ArrayOperations::rotateArray(psfMap, -rotationAngle);
  psfMap /= arma::accu(psfMap);
  return psfMap;
}


/*
 * This function gets the distortion map
 */

vector<std::array<double, 4>> PointSpreadFunction::getDistortionMap()
{
  return distortionMap;
}









/*
 * Initialize the distortion map. This can be done either by copying it from
 * the HDF5 file or generated if the starPointing angles are given.
 */

void PointSpreadFunction::initializeDistortionMap()
{

    // First we check if a Coordinates Map is defined in the hdf5 file

    if (psfFile.hasGroup("/Coordinates map"))
        readDistortionmapFromFile();
    else
    {
        // If no map is defined we will generate such a map.
        generateDistortionMap();
    }
}






/*
 * Reads the distortion map from the HDf5 and copies its content in distortionMap.
 */
void PointSpreadFunction::readDistortionmapFromFile()
{
    // We read in the table that converts the undistorted coordinates to distorted coordinates from the psf HDF5 file.
    vector<double> xUndistorted;
    vector<double> yUndistorted;

    vector<double> xDistorted;
    vector<double> yDistorted;

    psfFile.readArray("/Coordinates map/Undistorted", "x", xUndistorted);
    psfFile.readArray("/Coordinates map/Undistorted", "y", yUndistorted);
    psfFile.readArray("/Coordinates map/Distorted", "x", xDistorted);
    psfFile.readArray("/Coordinates map/Distorted", "y", yDistorted);



    for (unsigned int i=0; i < xDistorted.size(); i++)
    {
        const std::array<double, 4> coordinates = { xUndistorted.at(i), yUndistorted.at(i), xDistorted.at(i), yDistorted.at(i) };
        distortionMap.push_back(coordinates);
    }
}






/*
 * Generates a distortion map if no map is given in the HDf5 file.
 */
void PointSpreadFunction::generateDistortionMap()
{

    unsigned int datasetIndex = 1;
    string datasetName;
    string selectedDatasetName;

    while(true)
    {
        datasetName = to_string(datasetIndex);

        if(psfFile.hasDataset("/", datasetName))
        {
            // The distorted FP coordinats are stored as strings in the HDF5 file rather than doubles,
            // so we need to convert them.

            double xPsf = stod(psfFile.readStringDatasetAttribute("/", datasetName, "centerCoordinates1"));
            double yPsf = stod(psfFile.readStringDatasetAttribute("/", datasetName, "centerCoordinates2"));

            // We then read in the star pointing angles and from those we obtain the undistorted FP coordiantes.

            double starPointing1[1];
            double starPointing2[1];
            psfFile.readArrayDatasetAttribute("/", datasetName, "starPointing1", starPointing1);
            psfFile.readArrayDatasetAttribute("/", datasetName, "starPointing2", starPointing2);

            double xPsfUndistorted = focalLengthValue*tan(deg2rad(starPointing1[0]));
            double yPsfUndistorted = focalLengthValue*tan(deg2rad(starPointing2[0]));

            // Once we have these values we add them to the distortionMap
            const std::array<double, 4> coordinates = { xPsfUndistorted, yPsfUndistorted, xPsf, yPsf };
            distortionMap.push_back(coordinates);
        }
        else
        {
            break;
        }

        datasetIndex++;
    }
}

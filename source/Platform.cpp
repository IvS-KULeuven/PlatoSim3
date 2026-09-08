
#include "Platform.h"
#include "Units.h"



/**
 * \brief Constructor
 * 
 * \param configParams        Configuration parameters as read from the (e.g. yaml) inputfile
 * \param hdf5File            HDF5 file where to write any output
 * \param jitterGenerator     Generator of the yaw, pitch roll variations due to spacecraft jitter
 * 
 * \note  The jitterGenerator has been configured in Simulation::Simulation()
 */
Platform::Platform(ConfigurationParameters configParams,
		   HDF5File &hdf5File,
		   JitterGenerator &jitterGenerator)
: HDF5Writer(hdf5File),
  useJitter(true),
  internalTime(0.0),
  jitterGenerator(jitterGenerator)
{
    // Configure the Platfrom object

    configure(configParams);

    // Initialise the HDF5 group(s) in the output file

    initHDF5Groups();

    // Initialise the rotation matrices (should be done after configuration) with the unjittered spacecraft

    rotJitteredSpacecraftToEquatorial = getUnjitteredSpacecraftToEquatorialRotationMatrix();
    rotEquatorialToJitteredSpacecraft = rotJitteredSpacecraftToEquatorial.t();
}







/**
 * \brief Destructor
 */

Platform::~Platform()
{
    flushOutput();
}











/**
 * \brief Configure the Platform object
 * 
 * \param configParams  The configuration parameters from the input parameters file
 */

void Platform::configure(ConfigurationParameters &configParams)
{
  // General configuration parameters
  
  useJitter = configParams.getBoolean("Platform/UseJitter");
  writeACS  = configParams.getBoolean("ControlHDF5Content/WriteACS");
  platformOrientationSource = configParams.getString("Platform/Orientation/Source");

  // Select method of platform pointing
  
  if (platformOrientationSource == "Angles")
    {
      Log.info("Platform: using the (RA, Dec, Kappa) angles to determine the initial platform orientation");
      originalRA  = deg2rad(configParams.getDouble("Platform/Orientation/Angles/RAPointing"));            
      originalDec = deg2rad(configParams.getDouble("Platform/Orientation/Angles/DecPointing"));
      originalKappa = deg2rad(configParams.getDouble("Platform/Orientation/Angles/SolarPanelOrientation"));
      currentRA   = originalRA;
      currentDec  = originalDec;
      Log.debug("Platform: (RA, Dec, Kappa) = (" + to_string(rad2deg(originalRA)) + " deg, " + to_string(rad2deg(originalDec)) + " deg, " + to_string(rad2deg(originalKappa)) + " deg)");
    }
  else if (platformOrientationSource == "Quaternion")
    {
      Log.info("Platform: using the Platform/Orientation/Quaternion to determine the initial platform orientation");
      vector<double> quaternion = configParams.getDoubleVector("Platform/Orientation/Quaternion/Components");
      copy(quaternion.begin(), quaternion.end(), original_q_EQ2PLM.begin());
      tie(originalRA, originalDec) = calculatePlatFormSkyCoordinatesFromQuaternion(original_q_EQ2PLM);         // [rad]
      currentRA  = originalRA;                                                                                 // [rad]
      currentDec = originalDec;                                                                                // [rad]
      Log.debug("Platform: (RA, Dec) from quaternion = (" + to_string(rad2deg(originalRA)) + " deg, " + to_string(rad2deg(originalDec)) + " deg)");
    }
  else
    {
      Log.error("In input yaml file: unsupported Platform/Orientation/Source value. Only Angles or Quaternion are allowed");
    }
}












/**
 * \brief Creates the group(s) in the HDF5 file where the ACS information will be stored. 
 *        These group(s) have to be created once, at the very beginning.
 */

void Platform::initHDF5Groups()
{
  if (writeACS)
    {
      Log.debug("Platform: initialising HDF5 group: ACS");
      hdf5File.createGroup("/ACS");
    }
}











/**
 * \brief Write all recorded information to the HDF5 output file
 */
void Platform::flushOutput()
{
  if (writeACS)
    {
      Log.info("Platform: Flushing output to HDf5 file.");

      if (!hdf5File.hasGroup("/ACS"))
        {
	  Log.warning("Platform.flushOutput: HDF5 file has no ACS group, cannot flush Platform information.");
	  return;
        }

      if (!historyTime.empty())
        {
	  hdf5File.writeArray("/ACS", "time",        historyTime.data(),  historyTime.size());
	  hdf5File.writeArray("/ACS", "platformRA",  historyRA.data(),    historyRA.size());
	  hdf5File.writeArray("/ACS", "platformDec", historyDec.data(),   historyDec.size());
	  hdf5File.writeArray("/ACS", "yaw",         historyYaw.data(),   historyYaw.size());
	  hdf5File.writeArray("/ACS", "pitch",       historyPitch.data(), historyPitch.size());
	  hdf5File.writeArray("/ACS", "roll",        historyRoll.data(),  historyRoll.size());
        }
      else
        {
	  Log.warning("Platform: No ACS history to flush to HDF5 file.");
        }
    }
}











/**
 * \brief Repoints the spacecraft to the given pointing coordinates
 * 
 * \param rightAscencsion  Equatorial right Ascension [see unit]  
 * \param declination      Equatorial declination [see unit] 
 * \param unit             Unit of the input coordinates. Either Angle::degrees or Angle::radians.
 */

void Platform::setPointingCoordinates(double rightAscencsion, double declination, Unit unit)
{
    originalRA = rightAscencsion / unit;
    originalDec = declination / unit;

    currentRA  = originalRA;
    currentDec = originalDec;
}





/**
  * \brief Compute the RA, Declination coordinates given a quaterntion in the equatorial reference frame.
  *
  * \param q   Unit quaternion (q0, qx, qy, qz) describing the passive rotation from the equatorial reference 
  *            system to the platform reference system so that:
  *                 vec_PLM = q^* . vec_EQ . q
  *
  * \return (RA, Dec)   [rad]
  */   
pair<double, double> Platform::calculatePlatFormSkyCoordinatesFromQuaternion(quaternionType q_EQ2PLM)
{
    // In the Platform reference frame, the platform pointing vector corresponds with the z-axis (roll-axis)

    quaternionType v_PLM = {0.0, 0.0, 0.0, 1.0};
    quaternionType v_EQ = qmul(q_EQ2PLM, qmul(v_PLM, conj(q_EQ2PLM)));             // Real part can be ignored

    const double r = sqrt(v_EQ[1]*v_EQ[1] + v_EQ[2]*v_EQ[2] + v_EQ[3]*v_EQ[3]);
    const double dec = PI / 2.0 - acos(v_EQ[3]/r);                                 // Declination     [rad]
    double RA = atan2(v_EQ[2], v_EQ[1]);                                           // Right Ascension [rad]
    if (RA < 0.0) RA += 2 * PI; 
    
    // That's it

    return make_pair(RA, dec);
}






/**
 * \brief Update the pointing coordinates (of the roll axis) of the spacecraft
 * 
 * \param time [s]
 *  
 * \return Nothing
 */

void Platform::updatePlatformOrientation(double time)
{

    double yaw=0.0, pitch=0.0, roll=0.0;
    //    Log.warning("Platform: We use the jitter " + to_string(useJitter));
    if (useJitter)
    {
        // Check if the request is going backwards in time. If so: complain.

        if (time < internalTime)
        {
            Log.warning("Platform: updatePlatformOrientation() at time before previous request: Not Implemented. ");
            return;
        }


        // Check if the given 'time' is the same as the last one we processed (and kept).
        // If so, nothing has to change.

        if (!historyTime.empty())
        {
            if (time == historyTime.back())
            {
                Log.debug("Platform: updatePlatformOrientation(): coordinates up-to-date for requested time " + to_string(time));
                Log.debug("Platform: updatePlatformOrientation(): At time " + to_string(time) + ": (RA, dec) = (" 
                                               + to_string(historyRA.back()) + ", " 
                                               + to_string(historyDec.back()) + ") before applying jitter step.");
                return;
            }
        }

        // We're now in the case that we haven't processed the given time point yet.
        // Let the platfrom jitter until 'time'. Yaw, pitch, and roll are in [rad]
        tie(yaw, pitch, roll) = jitterGenerator.getNextYawPitchRoll(time);

        Log.debug("Platform: updatePlatformOrientation(): At time " + to_string(time) + ": (yaw, pitch, roll) = (" 
                                       + to_string(rad2deg(yaw)*3600.) + ", " 
                                       + to_string(rad2deg(pitch)*3600.) + ", " 
                                       + to_string(rad2deg(roll)*3600.) + ") arcsec");


        // Get the rotation matrix to take into account the jitter.

        arma::mat rotUnjittered2Jittered = getUnjitteredToJitteredRotationMatrix(yaw, pitch, roll);

        // Get the matrix to rotate from the unjittered spacecraft (SC) reference frame to 
        // the equatorial (EQ) reference frame

        arma::mat rotSC2EQ = getUnjitteredSpacecraftToEquatorialRotationMatrix();

        // Store the total rotation matrix and its inverse for later use

        rotJitteredSpacecraftToEquatorial = rotSC2EQ * rotUnjittered2Jittered;
        rotEquatorialToJitteredSpacecraft = rotJitteredSpacecraftToEquatorial.t();

        // Before the jitter, the roll axis has coordinates (0,0,1) in the SC reference frame
        // After jitter it will have slightly rotated. Afterwards, we convert from the SC to the EQ
        // reference frame

        arma::colvec zUnitBeforeJitterSC = {0.0, 0.0, 1.0};
        arma::colvec zUnitAfterJitterEQ = rotJitteredSpacecraftToEquatorial * zUnitBeforeJitterSC;
 
        // Convert from cartesian to sky coordinates

        const double x = zUnitAfterJitterEQ(0);
        const double y = zUnitAfterJitterEQ(1);
        const double z = zUnitAfterJitterEQ(2);

        const double r = sqrt(x*x+y*y+z*z);
        currentDec = PI / 2.0 - acos(z/r);
        currentRA = atan2(y, x);
        if (currentRA < 0.0) currentRA += 2 * PI; 
    }
    else
    {
        Log.info("Platform: No jitter, platform (yaw, pitch, roll) = (0.0, 0.0, 0.0)");
        yaw = 0.0;
        pitch = 0.0;
        roll = 0.0;
        currentRA = originalRA;
        currentDec = originalDec;

        // No need to change rotJitteredSpacecraftToEquatorial and rotEquatorialToJitteredSpacecraft
        // because they were already set in the constructor.
    }

    Log.debug("Platform: At time " + to_string(time) + ": (RA, dec) = (" 
                                   + to_string(rad2deg(currentRA)) + ", " 
                                   + to_string(rad2deg(currentDec)) + ") after applying jitter step.");

    // Update the internal clock

    internalTime = time;

    // Store the computed values so that they can later be saved to HDF5
    // RA & Dec are saved in degrees. Yaw, pitch, roll in arcsec.

    historyTime.push_back(time);
    historyRA.push_back(rad2deg(currentRA));                 // [deg]
    historyDec.push_back(rad2deg(currentDec));               // [deg]
    historyYaw.push_back(rad2deg(yaw) * 3600.);              // [arcsec]
    historyPitch.push_back(rad2deg(pitch) * 3600.);          // [arcsec]
    historyRoll.push_back(rad2deg(roll) * 3600.);            // [arcsec]

    // That's it

    return;
}













/**
 * \brief Return the current (RA, Dec) pointing coordinates of the spacecraft
 * 
 * \return (RA, Dec)  Right Ascension and declination [rad]
 */

pair<double, double> Platform::getCurrentPointingCoordinates()
{
    return make_pair(currentRA, currentDec);
}













/**
 * \brief Return the current (RA, Dec) pointing coordinates of the spacecraft
 * 
 * \return (RA, Dec)  Right Ascension and declination [rad]
 */

pair<double, double> Platform::getInitialPointingCoordinates()
{
    return make_pair(originalRA, originalDec);
}













/**
 * \brief Return the rotation matrix to rotate from cartesian coordinates in the 
 *        jittered spacecraft (SC) reference frame to the cartesian coordinates 
 *        in the equatorial reference frame.
 *        
 * \note This matrix was computed in updateSpacecraftOrientation() with the latest
 *       (yaw, pitch, roll) values.
 *        
 * \return 3x3 rotation matrix
 */

arma::mat Platform::getJitteredSpacecraftToEquatorialRotationMatrix()
{
    return rotJitteredSpacecraftToEquatorial;
}










/**
 * \brief Return the rotation matrix to rotate from cartesian coordinates in the 
 *        equatorial reference frame to the cartesian coordinates in the jittered 
 *        spacecraft (SC) reference frame.
 *        
 * \note This matrix was computed in updateSpacecraftOrientation() with the latest
 *       (yaw, pitch, roll) values.
 *         
 * \return 3x3 rotation matrix
 */

arma::mat Platform::getEquatorialToJitteredSpacecraftRotationMatrix()
{
    return rotEquatorialToJitteredSpacecraft;
}













/**
 * \brief Return the heartbeat interval of Platform. It's the largest interval with which 
 *        small Platform variations (e.g. due to Jitter) can be tracked.
 * 
 */

double Platform::getHeartbeatInterval()
{
    // Currently Platform does not change in any other way than just jittering, so delegate to the jitter generator.

    return jitterGenerator.getHeartbeatInterval();
}














/**
 * \brief Compute rotation matrix to tranform cartesian coordinates in the spacecraft reference frame to 
 *        cartesian coordinates in the equatorial reference frame. That is, return the 3x3 matrix rotSC2EQ
 *        so that 
 *               vecEQ = rotSC2EQ * vecSC
 *        where vecEQ are cartesian coordinates in the equatorial reference frame, and vecSC are the 
 *        cartesian coordinates in the Spacecraft (platform) reference frame.
 *        See PLATO-KUL-PL-TN-0001 for more details on the definition of the rotation matrices.
 *
 * \return rotSC2EQ : 3x3 rotation matrix 
 */

arma::mat Platform::getUnjitteredSpacecraftToEquatorialRotationMatrix()
{
    arma::mat rotEQ2SC = getEquatorialToUnjitteredSpacecraftRotationMatrix();
    arma::mat rotSC2EQ = rotEQ2SC.t();
    return rotSC2EQ;
}














/**
 * \brief Compute rotation matrix to tranform cartesian coordinates in the equatorial reference frame to
 *        cartesian coordinates in the spacecraft reference frame. That is, return the 3x3 matrix rotSC2EQ
 *        so that 
 *               vecSC = rotEQ2SC * vecEQ
 *        where vecEQ are cartesian coordinates in the equatorial reference frame, and vecSC are the 
 *        cartesian coordinates in the Spacecraft (platform) reference frame.
 *        See PLATO-KUL-PL-TN-0001 for more details.
 *
 * \return rotEQ2SC : 3x3 rotation matrix 
 */

arma::mat Platform::getEquatorialToUnjitteredSpacecraftRotationMatrix()
{
    if (platformOrientationSource == "Angles") 
    {
        arma::mat rotEQ2A;
        rotEQ2A <<  cos(originalRA) << sin(originalRA) << 0.0 << arma::endr
                << -sin(originalRA) << cos(originalRA) << 0.0 << arma::endr
                <<        0.0       <<        0.0      << 1.0 << arma::endr;

        arma::mat rotA2B;
        rotA2B << sin(originalDec) << 0.0 << -cos(originalDec) << arma::endr
               <<       0.0        << 1.0 <<        0.0        << arma::endr
               << cos(originalDec) << 0.0 <<  sin(originalDec) << arma::endr;
        
        arma::mat rotB2SC;
        rotB2SC <<  cos(originalKappa) << sin(originalKappa) << 0.0 << arma::endr
                << -sin(originalKappa) << cos(originalKappa) << 0.0 << arma::endr
                <<        0.0          <<        0.0         << 1.0 << arma::endr;

        arma::mat rotEQ2SC = rotB2SC * rotA2B * rotEQ2A;

        return rotEQ2SC;
    }
    else    // platformOrientationSource == "Quaternion"
    {
        // Convert the quaternion into a rotation matrix. 
        
        const double q0 = original_q_EQ2PLM[0];
        const double qx = original_q_EQ2PLM[1];
        const double qy = original_q_EQ2PLM[2];
        const double qz = original_q_EQ2PLM[3];
        arma::mat rotEQ2SC;                                                                         // Passive rotation matrix! 
        rotEQ2SC << 2*(q0*q0+qx*qx)-1 << 2*(qx*qy+q0*qz)   << 2*(qx*qz-q0*qy)   << arma::endr
                 << 2*(qx*qy-q0*qz)   << 2*(q0*q0+qy*qy)-1 << 2*(qy*qz+q0*qx)   << arma::endr
                 << 2*(qx*qz+q0*qy)   << 2*(qy*qz-q0*qx)   << 2*(q0*q0+qz*qz)-1 << arma::endr;

        return rotEQ2SC;
    }
}













/**
 * \brief Transforms the coordinates of a vector after yaw, pitch, and roll rotations.
 * 
 * \details We define the yaw, the pitch, and the roll as rotation angles around respectively 
 *          the X_SC, Y_SC, and Z_SC axes, (SC standing for spacecraft reference frame) such that 
 *          the angles increase with a clockwise rotation, when looking along the positive axes. 
 *          First a roll rotation is done around the Z_SC axis, then a pitch rotation is done 
 *          around the rotated Y_SC axis, and finally a yaw rotation is done around the already
 *          twice-rotated X_SC axis. The combined rotation matrix is: 
 *                \f[ R(yaw, pitch, roll) = R(yaw) * R(pitch) * R(roll) \f]
 *          Which transforms
 *                \f[ vector\_after = R(yaw, pitch, roll) * vector\_before \f]
 *          where vector_before and vector_after are the coordinates of the same point before and
 *          after the yaw.pitch,roll rotations. 
 *          
 * \note See also Technical Note PLATO-KUL-PL-TN-001
 * 
 * \param yaw    Yaw angle   [rad]
 * \param pitch  Pitch angle [rad]
 * \param roll   Roll angle  [rad]
 * \return       3x3 rotation matrix 
 */

arma::mat Platform::getUnjitteredToJitteredRotationMatrix(const double yaw, const double pitch, const double roll)
{
    // Some handy abbreviations

    const double sinYaw   = sin(yaw);
    const double cosYaw   = cos(yaw);
    const double sinPitch = sin(pitch);
    const double cosPitch = cos(pitch);
    const double sinRoll  = sin(roll);
    const double cosRoll  = cos(roll);

    // The rotation matrices

    arma::mat Ryaw;
    Ryaw << 1.0  <<  0.0    <<     0.0   << arma::endr
         << 0.0  <<  cosYaw <<  -sinYaw  << arma::endr
         << 0.0  <<  sinYaw <<   cosYaw  << arma::endr;

    arma::mat Rpitch;
    Rpitch <<  cosPitch  <<  0.0  <<  sinPitch  << arma::endr
           <<   0.0      <<  1.0  <<    0.0     << arma::endr
           << -sinPitch  <<  0.0  <<  cosPitch  << arma::endr;

    arma::mat Rroll;
    Rroll << cosRoll  <<  -sinRoll  <<  0.0  << arma::endr
          << sinRoll  <<   cosRoll  <<  0.0  << arma::endr
          <<  0.0     <<     0.0    <<  1.0  << arma::endr; 

    // That's it

    return Ryaw  * Rpitch * Rroll;
}





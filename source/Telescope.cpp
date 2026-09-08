#include "Telescope.h"

/**
 * Constructor
 * 
 * \param configurationParameters: Configuration parameters for the telescope.
 * \param hdf5File                 Output HDF5 file.
 * \param Platform:                Platform on which the telescope is mounted
 * 
 */

Telescope::Telescope(ConfigurationParameters &configParams, HDF5File &hdf5File, Platform &platform, DriftGenerator &driftGenerator)
: HDF5Writer(hdf5File), originalAzimuthAngle(0.0), originalTiltAngle(0.0), useDrift(true), internalTime(0.0), 
  driftGenerator(driftGenerator), platform(platform)
{

    // Retrieve the Telescope configuration parameters

    configure(configParams);

    // Initialise the HDF5 group(s) in the output file
    if (writeTelescopeACS)
    {  
    initHDF5Groups();
    }

    // Initialise the heartbeat interval of the telescope.
    // The Telescope properties (e.g. the coordinates of the optical axis) are evolving in time, 
    // for example because of thermo-elastic drift, or because of the jitter of the platform it 
    // is mounted on. To properly track these changes one has to use a small enough timestep, 
    // which is called the "heartbeat" interval of the Telescope. Because Telescope depends on 
    // other components, like Platform which in turn may also have a certain heartbeat, the
    // 'global' heartbeat of Telescope is the minimum of its own intrinsic heartbeat and the
    // heartbeat of all the components it depends on.

    heartbeatInterval = min(platform.getHeartbeatInterval(), driftGenerator.getHeartbeatInterval());

    // Initialise the Telescope <-> Spacecraft reference frame rotation matrices with 
    // the unjittered and undrifted versions.

    rotDriftedTelescopeToSpacecraft = getUndriftedTelescopeToPlatformRotationMatrix();
    rotSpacecraftToDriftedTelescope = rotDriftedTelescopeToSpacecraft.t();

    // Initialize the current sky coordinates of the optical axis of the telescope.
    // In the telescope reference frame, the optical axis has cartesian coordinates (0,0,1).

    arma::mat rotSC2EQ = platform.getJitteredSpacecraftToEquatorialRotationMatrix();

    arma::colvec opticalAxisTL = {0.0, 0.0, 1.0};
    arma::colvec opticalAxisEQ = rotSC2EQ * rotDriftedTelescopeToSpacecraft * opticalAxisTL;

    const double x = opticalAxisEQ(0);
    const double y = opticalAxisEQ(1);
    const double z = opticalAxisEQ(2);
 
    const double r = sqrt(x*x + y*y + z*z);
    currentDeltaOpticalAxis = PI / 2.0 - acos(z/r);                               // Dec is measured from equator, not North Pole.  [rad]
    currentAlphaOpticalAxis = atan2(y, x);                                        // [rad]
    if (currentAlphaOpticalAxis < 0.0) currentAlphaOpticalAxis += 2 * PI; 
}










/**
 * Destructor.
 */

Telescope::~Telescope()
{
    flushOutput();
}











/**
 * \brief Configure the Telescope object using the ConfigurationParameters
 * 
 * \param configParam: the configuration parameters 
 **/

 void Telescope::configure(ConfigurationParameters &configParams)
 {
    // Configuration parameters for the Telescope


    string groupID                = configParams.getString("Telescope/GroupID");

    if (groupID == "Custom")
    {
        originalAzimuthAngle      = deg2rad(configParams.getDouble("Telescope/AzimuthAngle"));           // [rad]
        originalTiltAngle         = deg2rad(configParams.getDouble("Telescope/TiltAngle"));              // [rad]
    }
    else if (groupID == "Fast")
    {
        originalAzimuthAngle      = deg2rad(configParams.getDoubleAt("CameraGroups/AzimuthAngle", 4));   // [rad]
        originalTiltAngle         = deg2rad(configParams.getDoubleAt("CameraGroups/TiltAngle", 4));      // [rad]
    }
    else
    {
        int idx = stoi(groupID) - 1;  // Groups are named [1, 2, 3, 4] while the index into vector starts at 0
        originalAzimuthAngle      = deg2rad(configParams.getDoubleAt("CameraGroups/AzimuthAngle", idx)); // [rad]
        originalTiltAngle         = deg2rad(configParams.getDoubleAt("CameraGroups/TiltAngle", idx));    // [rad]
    }

    Log.info("Telescope: selected groupID = " + groupID);
    Log.debug("Telescope: azimuth, tilt = " + to_string(rad2deg(originalAzimuthAngle)) + ", " + to_string(rad2deg(originalTiltAngle)));

    lightCollectingArea       = configParams.getDouble("Telescope/LightCollectingArea") * 1.e-4;            // [m^2]  
    transmissionEfficiencyBOL = configParams.getDouble("Telescope/TransmissionEfficiency/BOL");
    transmissionEfficiencyEOL = configParams.getDouble("Telescope/TransmissionEfficiency/EOL");
    missionDuration           = configParams.getDouble("ObservingParameters/MissionDuration") * 31536000.0; // [s]
    useDrift                  = configParams.getBoolean("Telescope/UseDrift");
    writeTelescopeACS         = configParams.getBoolean("ControlHDF5Content/WriteTelescopeACS");
}








/**
 * \brief Creates the group(s) in the HDF5 file where the Telescope pointing information
 *        will be stored. These group(s) have to be created once, at the very beginning.
 */

void Telescope::initHDF5Groups()
{
    Log.debug("Telescope: initialising HDF5 groups");
    hdf5File.createGroup("/Telescope");
}









/**
 * \brief Write all recorded information to the HDF5 output file
 */

void Telescope::flushOutput()
{

    if (writeTelescopeACS)
    {
      Log.info("Telescope: Flushing output to HDf5 file.");
      if ( ! hdf5File.hasGroup("Telescope") )
	{
	  Log.warning("Telescope::flushOutput: HDF5 file has no Telescope group, cannot flush Telescope information.");
	  return;
	}
    

      if (!historyTime.empty() )
	{
	  hdf5File.writeTelescopeACS(historyTime, historyRA, historyDec, historyYaw, historyPitch, historyRoll);
	}
      else
	{
	  Log.warning("Telescope: No telescope pointing history to flush to HDF5 file.");
	}
    }
}








/**
 * \brief The telescope has some parameters (orientation, temperature, ...)
 *        that can be time dependent. This function updates these parameters
 *        to their value at the given time.
 *
 *
 * \param time: the time point for which the time dependent telescope parameters
 *              need to be updated.
 *
 * \return -
 */

void Telescope::updateParameters(double time)
{
    // Update the orientation of the telescope:
    
    updateTelescopeOrientation(time);

}








/**
 * \brief Update the telescope's pointing coordinates, by evolving the pointing coordinates 
 *        (due to e.g. jitter or thermo-elastic variations) until time point 'time'.  
 */ 



void Telescope::updateTelescopeOrientation(double time)
{
    double yaw=0.0, pitch=0.0, roll=0.0;

    // Check if the request is going backwards in time. If so: complain.

    if (time < internalTime)
    {
        Log.warning("Telescope: updateTelescopeOrientation() at time before previous request: Not Implemented. ");
        return;
    }


    // Check if the given 'time' is the same as the last one we processed (and kept).
    // If so, nothing has to change.

    if (!historyTime.empty())
    {
        if (time == historyTime.back())
        {
            Log.debug("Telescope: updateTelescopeOrientation: coordinates up-to-date for requested time " + to_string(time));
            Log.debug("Telescope: At time " + to_string(time) + ": (RA, dec) = (" 
                                           + to_string(rad2deg(currentAlphaOpticalAxis)) + ", " 
                                           + to_string(rad2deg(currentDeltaOpticalAxis)) + ")");
            return;
        }
    }

    // Get the matrix to rotate from the undrifted telescope (TL) reference frame to 
    // the spacecraft (SC) = platform reference frame

    arma::mat rotTL2SC = getUndriftedTelescopeToPlatformRotationMatrix();

    // Get the rotation matrix to transform the spacecraft coordinates of the (drifted) optical axis 
    // to equatorial coordinates, taking into account that the platform itself may have jittered meanwhile.

    platform.updatePlatformOrientation(time);
    arma::mat rotSC2EQ = platform.getJitteredSpacecraftToEquatorialRotationMatrix();

    // Before any drift is involved, the optical axis has coordinates (0,0,1) in the TL reference frame
    // After the drift it will have slightly rotated.   

    arma::colvec zUnitBeforeDriftTL = {0.0, 0.0, 1.0};      // In telescope reference frame
    arma::colvec zUnitAfterDriftAndJitterEQ;                // In equatorial reference frame

    // The exact transformation from the telescope reference frame to the equatorial reference fraem
    // depends on whether drift is included or not.

    if (useDrift)
    {
        // Let the telescope drift until 'time'. Yaw, pitch, and roll are in [rad]

        tie(yaw, pitch, roll) = driftGenerator.getNextYawPitchRoll(time);

        Log.debug("Telescope: updateTelescopeOrientation(): At time " + to_string(time) + ": (yaw, pitch, roll) = (" 
                                        + to_string(rad2deg(yaw)*3600.) + ", " 
                                        + to_string(rad2deg(pitch)*3600.) + ", " 
                                        + to_string(rad2deg(roll)*3600.) + ") arcsec");


        // Get the rotation matrix to take into account the drift.

        arma::mat rotUndrifted2Drifted = getUndriftedToDriftedRotationMatrix(yaw, pitch, roll);
        
        // Before the drift, the optical axis has coordinates (0,0,1) in the TL reference frame
        // After drift it will have slightly rotated. Afterwards, we convert from the TL to the SC
        // and from the SC to the EQ reference frame

        zUnitAfterDriftAndJitterEQ = rotSC2EQ * rotTL2SC * rotUndrifted2Drifted * zUnitBeforeDriftTL;

        // Store the total rotation matrix and its inverse for later use

        rotDriftedTelescopeToSpacecraft = rotTL2SC * rotUndrifted2Drifted;
        rotSpacecraftToDriftedTelescope = rotDriftedTelescopeToSpacecraft.t();
    }
    else
    {
        Log.info("Telescope: updateTelescopeOrientation(): No drift, telescope (yaw, pitch, roll) = (0.0, 0.0, 0.0)");
        yaw = 0.0;
        pitch = 0.0;
        roll = 0.0;

        // Convert the coordinates in the telescope reference frame into coordinates in 
        // the equatorial reference frame, without taking into account any drift.

        zUnitAfterDriftAndJitterEQ = rotSC2EQ * rotTL2SC * zUnitBeforeDriftTL;

        // Store the total rotation matrix and its inverse for later use, again without drift.

        rotDriftedTelescopeToSpacecraft = rotTL2SC;
        rotSpacecraftToDriftedTelescope = rotDriftedTelescopeToSpacecraft.t();
    }

    // Convert from cartesian to sky coordinates

    const double x = zUnitAfterDriftAndJitterEQ(0);
    const double y = zUnitAfterDriftAndJitterEQ(1);
    const double z = zUnitAfterDriftAndJitterEQ(2);

    const double r = sqrt(x*x+y*y+z*z);
    currentDeltaOpticalAxis = PI / 2.0 - acos(z/r);
    currentAlphaOpticalAxis = atan2(y, x);
    if (currentAlphaOpticalAxis < 0.0) currentAlphaOpticalAxis += 2 * PI; 


    Log.debug("Telescope: updateTelescopeOrientation(): at time " + to_string(time) + ": (RA, dec) = (" 
                                   + to_string(rad2deg(currentAlphaOpticalAxis)) + ", " 
                                   + to_string(rad2deg(currentDeltaOpticalAxis)) + ")");

    // Update the internal clock

    internalTime = time;

    // Store the computed values so that they can later be saved to HDF5
    // RA & Dec are saved in degrees. Yaw, pitch, roll in arcsec.

    historyTime.push_back(time);
    historyRA.push_back(rad2deg(currentAlphaOpticalAxis));                            // [deg]
    historyDec.push_back(rad2deg(currentDeltaOpticalAxis));                           // [deg]
    historyYaw.push_back(rad2deg(yaw) * 3600.);                                       // [arcsec]
    historyPitch.push_back(rad2deg(pitch) * 3600.);                                   // [arcsec]
    historyRoll.push_back(rad2deg(roll) * 3600.);                                     // [arcsec]

    // That's it

    return;

}















/**
 * \brief Return the heartbeat interval of Telescope. It's the largest interval with which 
 *        small Telescope variations (e.g. due to Platform Jitter or due to thermo-elastic
 *        drift variations) can be tracked.
 * 
 */

double Telescope::getHeartbeatInterval()
{
    return min(platform.getHeartbeatInterval(), driftGenerator.getHeartbeatInterval());
}












/**
 * \brief Return the transmission efficiency of the telescope (number between 0 and 1)
 * 
 */

double Telescope::getTransmissionEfficiency(double time)
{
    return transmissionEfficiencyBOL - (transmissionEfficiencyBOL - transmissionEfficiencyEOL) / missionDuration * time;
}












/**
 * \brief Return the effective light collecting area (in [m^2])
 * 
 */


double Telescope::getLightCollectingArea()
{
    return lightCollectingArea;
}











/**
 * \brief Compute the rotation matrix to transform coordinates in the undrifted telescope
 *        reference frame to coordinates in the platform reference frame. That is, 
 *        return the 3x3 matrix rotTL2SC
 *        so that 
 *               vecSC = rotTL2SC * vecTL
 *        where vecSC are cartesian coordinates in the spacecraft (platform) reference frame, 
 *        and vecTL are the cartesian coordinates in the Telescope reference frame.
 *        See PLATO-KUL-PL-TN-0001 for more details.
 *
 * \return rotTL2SC : 3x3 rotation matrix 
 */

arma::mat Telescope::getUndriftedTelescopeToPlatformRotationMatrix()
{
    // Rotating over an azimuth angle around the Z-axis of the platform

    arma::mat rotAzimuth;
    rotAzimuth <<  cos(originalAzimuthAngle) << sin(originalAzimuthAngle) << 0 << arma::endr
               << -sin(originalAzimuthAngle) << cos(originalAzimuthAngle) << 0 << arma::endr
               <<           0                <<          0                << 1 << arma::endr;

    // Rotating over a tilt angle around the Y-axis of the telescope

    arma::mat rotTilt;
    rotTilt <<  cos(originalTiltAngle)  << 0 << -sin(originalTiltAngle)  << arma::endr
            <<            0             << 1 <<             0            << arma::endr
            <<  sin(originalTiltAngle)  << 0 <<  cos(originalTiltAngle)  << arma::endr;

    return rotAzimuth.t() * rotTilt.t() * rotAzimuth;
}











/**
 * \brief Compute the rotation matrix to transform coordinates in the spacecraft (platform)
 *        reference frame to coordinates in the undrifted telescope reference frame. That is, 
 *        return the 3x3 matrix rotSC2TL
 *        so that 
 *               vecTL = rotSC2TL * vecSc
 *        where vecSC are cartesian coordinates in the spacecraft (platform) reference frame, 
 *        and vecTL are the cartesian coordinates in the undrifted telescope reference frame.
 *        See PLATO-KUL-PL-TN-0001 for more details.
 *
 * \return rotSC2TL : 3x3 rotation matrix 
 */

arma::mat Telescope::getPlatformToUndriftedTelescopeRotationMatrix()
{
    arma::mat rotTL2SC = getUndriftedTelescopeToPlatformRotationMatrix();
    arma::mat rotSC2TL = rotTL2SC.t();

    return rotSC2TL;
}













/**
 * \brief Transforms the coordinates of a vector after yaw, pitch, and roll rotations.
 * 
 * \details We define the yaw, the pitch, and the roll as rotation angles around respectively 
 *          the X_TL, Y_TL, and Z_TL axes, (TL standing for telescope reference frame) such that 
 *          the angles increase with a clockwise rotation, when looking along the positive axes. 
 *          First a roll rotation is done around the Z_TL axis, then a pitch rotation is done 
 *          around the rotated Y_TL axis, and finally a yaw rotation is done around the already
 *          twice-rotated X_TL axis. The combined rotation matrix is: 
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

arma::mat Telescope::getUndriftedToDriftedRotationMatrix(const double yaw, const double pitch, const double roll)
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









/**
 * \brief Return the rotation matrix to rotate from cartesian coordinates in the 
 *        drifted telescope (TL) reference frame to the cartesian coordinates 
 *        in the spacecraft (platform) reference frame.
 *        
 * \note This matrix was computed in updateTelescopeOrientation() with the latest
 *       (yaw, pitch, roll) values.
 *        
 * \return 3x3 rotation matrix
 */

arma::mat Telescope::getDriftedTelescopeToPlatformRotationMatrix()
{
    return rotDriftedTelescopeToSpacecraft;
}










/**
 * \brief Return the rotation matrix to rotate from cartesian coordinates in the 
 *        spacecraft (platform) reference frame to the cartesian coordinates in the  
 *        drifted telescope (TL) reference frame.
 *        
 * \note This matrix was computed in updateTelescopeOrientation() with the latest
 *       (yaw, pitch, roll) values.
 *         
 * \return 3x3 rotation matrix
 */

arma::mat Telescope::getPlatformToDriftedTelescopeRotationMatrix()
{
    return rotSpacecraftToDriftedTelescope;
}






#ifndef PLATFORM_H
#define PLATFORM_H

#include <string>
#include <vector>
#include <array>
#include <cmath>

#include "armadillo"

#include "Logger.h"
#include "Units.h"
#include "Constants.h"
#include "Heartbeat.h"
#include "HDF5Writer.h"
#include "HDF5File.h"
#include "SkyCoordinates.h"
#include "Telescope.h"
#include "ConfigurationParameters.h"
#include "JitterGenerator.h"
#include "Quaternion.h"


using namespace std;
using Constants::PI;


class Telescope;  // forward declaration



class Platform : public Heartbeat, HDF5Writer
{
    public:

        Platform(ConfigurationParameters configParams, HDF5File &hdf5File, JitterGenerator &jitterGenerator);
        virtual ~Platform();

        virtual void configure(ConfigurationParameters &configParams);

        void setPointingCoordinates(double rightAscencsion, double declination, Unit unit = Angle::degrees);
        void updatePlatformOrientation(double time);
        pair<double, double> getCurrentPointingCoordinates();
        pair<double, double> getInitialPointingCoordinates();

        arma::mat getJitteredSpacecraftToEquatorialRotationMatrix();
        arma::mat getEquatorialToJitteredSpacecraftRotationMatrix();

        arma::mat getUnjitteredSpacecraftToEquatorialRotationMatrix();
        arma::mat getEquatorialToUnjitteredSpacecraftRotationMatrix();

        virtual double getHeartbeatInterval() override;
        

    protected:

        arma::mat getUnjitteredToJitteredRotationMatrix(const double yaw, const double pitch, const double roll);
        pair<double, double> calculatePlatFormSkyCoordinatesFromQuaternion(quaternionType q_EQ2PLM);
        
        tuple<double, double, double> getNextYawPitchRoll(double time);

        virtual void initHDF5Groups() override;
        virtual void flushOutput() override;

        arma::mat rotJitteredSpacecraftToEquatorial;  // rotation matrix 
        arma::mat rotEquatorialToJitteredSpacecraft;  // rotation matrix 

        bool useJitter;                             // If false, the yaw, pitch, and roll, are always zero.
        double internalTime;                        // [s]
        double currentRA;                           // Current right Ascension of spacecraft pointing axis (zSC-axis)             [rad]
        double currentDec;                          // Current declination     of spacecraft pointing axis (zSC-axis)             [rad]
        double originalRA;                          // Original user-given right Ascension of spacecraft pointing axis (zSC-axis) [rad]
        double originalDec;                         // Original user-given declination     of spacecraft pointing axis (zSC-axis) [rad]
        double originalKappa;                       // Original user-given Platform roll angle before the jitter starts           [rad]
        string platformOrientationSource;           // Either "Angles" or "Quaternion": orientation in the EQ reference frame

        JitterGenerator &jitterGenerator; 

        quaternionType original_q_EQ2PLM;           // Original quaternion to convert from the EQ ref frame to the spacecraft ref frame. 
        vector<double> historyTime;                 // The following vectors stores all computed ASC values to write to HDF5
        vector<double> historyRA;
        vector<double> historyDec;
        vector<double> historyYaw;
        vector<double> historyPitch;
        vector<double> historyRoll;

        bool writeACS;                              // If true write jitter info to the HDF5 file. Don't if false.

    private:

};



#endif

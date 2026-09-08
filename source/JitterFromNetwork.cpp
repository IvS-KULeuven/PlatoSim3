#include "JitterFromNetwork.h"



/**
 * \brief Constructor
 * 
 * \param configParams The configuration parameters from the input parameters file
 */

JitterFromNetwork::JitterFromNetwork(ConfigurationParameters &configParams)
    : context(1)
    , jitterSocket(context, ZMQ_SUB)
{
    // Set the configuration parameters

    configure(configParams);

    // declare a zmq socket and connect it to the jitter server address

    jitterSocket.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    jitterSocket.setsockopt(ZMQ_RCVTIMEO, &jitterSocketTimeout, sizeof(jitterSocketTimeout));

    jitterSocket.connect(jitterAddress);

}










/**
 * \brief Destructor
 */

JitterFromNetwork::~JitterFromNetwork()
{

}







/**
 * \brief Configure this object using the parameters from the input parameters file
 * 
 * \param configParams  The configuration parameters
 */

void JitterFromNetwork::configure(ConfigurationParameters &configParams)
{
    // Note that the inputfile lists the jitter RMS values in [arcsec]

    endOfSimulation = false;

    oldTimeStep = -0.125;
    oldYaw = 0;
    oldPitch = 0;
    oldRoll = 0;

    currentTimeStep = 0;
    currentYaw = 0;
    currentPitch = 0;
    currentRoll = 0;

    endOfSimulation = false;

    jitterAddress = configParams.getString("ControlTcpConnection/JitterServerAddress");

    jitterSocketTimeout = configParams.getInteger("ControlTcpConnection/JitterSocketTimeout");

    if (jitterSocketTimeout < 0)
    {
        jitterSocketTimeout = -1;
    }
    else
    {
        jitterSocketTimeout *= 1000;
    }

}









/**
 * \brief Get the next (yaw, pitch, roll) values using by waiting for a jitter step message from a server
 * 
 * \param time   The time point for which the (yaw, pitch, roll) is requested [s]
 * 
 * \return (newYaw, newPitch, newRoll)  [rad]
 */

tuple<double, double, double> JitterFromNetwork::getNextYawPitchRoll(double time)
{
    // ask for the server thread to get a new jitter step until the new step has a larger time stamp than time

    while(time >= currentTimeStep)
    {
        Log.info("JitterFromNetwork: time: " + to_string(time));
        Log.info("JitterFromNetwork: currentTime: " + to_string(currentTimeStep));

        if (!endOfSimulation)
        {   
            oldTimeStep = currentTimeStep;

            oldYaw = currentYaw;
            oldPitch = currentPitch;
            oldRoll = currentRoll;

            Log.info("JitterFromNetwork: wait for new step");

            // define the message from the jitter socket
            zmq::message_t jitterMessage;

            // wait for a message from server (note: if there is no message, the simulation is stuck here. TODO: implement a timeout)
            jitterSocket.recv(&jitterMessage);

            std::string jitterMessageString = std::string(static_cast<char*>(jitterMessage.data()), jitterMessage.size());

            tuple<double, double, double, double> jitterStep = processJitterMessage(jitterMessageString);

            currentTimeStep = std::get<0>(jitterStep);

            currentYaw = deg2rad((std::get<1>(jitterStep))/3600.);
            currentPitch = deg2rad((std::get<2>(jitterStep))/3600.);
            currentRoll = deg2rad((std::get<3>(jitterStep))/3600.);



            if (abs(time - currentTimeStep) < 0.000001)
            {
                Log.info("JitterFromNetwork: time: " + to_string(currentTimeStep) + "; yaw: " + to_string(currentYaw*1000) + "x10^-3; pitch: " + to_string(currentPitch*1000) + "x10^-3; roll: " + to_string(currentRoll*1000) + "x10^-3");

                return make_tuple(currentYaw, currentPitch, currentRoll);
            }
            else if (currentTimeStep > time)
            {
                // interpolate the jitterstep dependent on the old and new jitterstep and the currentTime

                const double weight1 = (time - oldTimeStep) / (currentTimeStep - oldTimeStep);
                const double weight2 = (currentTimeStep - time) / (currentTimeStep - oldTimeStep);
                const double newYaw   = oldYaw   * weight2 + currentYaw   * weight1;
                const double newPitch = oldPitch * weight2 + currentPitch * weight1;
                const double newRoll  = oldRoll  * weight2 + currentRoll  * weight1;

                Log.info("JitterFromNetwork: time: " + to_string(currentTimeStep) + "; yaw: " + to_string(newYaw*1000) + "x10^-3; pitch: " + to_string(newPitch*1000)+ "x10^-3; roll: " + to_string(newRoll*1000) + "x10^-3");

                return make_tuple(newYaw, newPitch, newRoll);
            }
        }
        else
        {
            // just return empty jitter steps, if there are no more steps from network, but still imagettes to write

            return make_tuple(0.0, 0.0, 0.0);
        }
    }
}

/**
 * \brief Return the heartbeat interval of this network jitter generator
 * 
 * \details The heartbeat interval is the jitter time interval which is set to a fraction of 
 *          the jitter time scale. so that the changes in (yaw, pitch, roll) can still be 
 *          reliably tracked.
 *          
 * \return  heartbeatInterval [s]
 */

double JitterFromNetwork::getHeartbeatInterval()
{
    heartbeatInterval = currentTimeStep - oldTimeStep;

    Log.info("JitterFromNetwork: heartbeatInterval: " + to_string(heartbeatInterval));

    return heartbeatInterval;
}


/**
 * \brief The jitter server message is in a string format - this function converts it to a vector of doubles
 * \      the message consist either of a timestamp and a yaw -, pitch -, roll angle (all double)
 * \      or an empty string, which indicates the end of the simulation
 *  
 */
tuple<double, double, double, double> JitterFromNetwork::processJitterMessage(const std::string jitterString)
{
    // fracture the received string
    std::stringstream ss(jitterString);

    std::vector<double> currentStepVec;

    bool endSimulation;
    double timeStep;
    double yaw;
    double pitch;
    double roll;

    double i;

    // convert the string to double values
    while (ss >> i)
    {
        currentStepVec.push_back(i);

        if (ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }
    }

    // check length of the jitter step

    if (currentStepVec.size() == 4)
    {
        timeStep = currentStepVec.at(0);
        yaw = currentStepVec.at(1);
        pitch = currentStepVec.at(2);
        roll = currentStepVec.at(3);
        
    }
    else
    {
        if(jitterString == "")
        {
            endOfSimulation = true;
        }

        timeStep = 0;
        yaw = 0;
        pitch = 0;
        roll = 0;
    }

    return make_tuple(timeStep, yaw, pitch, roll);

}
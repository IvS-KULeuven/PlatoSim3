#include "ClosedLoopUtility.h"



/**
 * \brief constructor of the closedLoopUtility class
 *        setting up the used connections
 */
ClosedLoopUtility::ClosedLoopUtility(ConfigurationParameters &configParams)
: context(1)
, imagetteSocket(context, ZMQ_DEALER)
, windowPositionSocket(context, ZMQ_DEALER)
{
    configure(configParams);

    // set pid as unique zmq-identity for both sockets in order to establish a relation 
    // between window position and imagette on the receiver side

    pid_t simulationPid = getpid();
    std::string identity = std::to_string(simulationPid);
    
    Log.info("ClosedLoopUtility: ZMQ socket identity string: " + identity);

    // connect to the imagette client, if it is specified in the inputfile

    if (sendImagettesToClient)
    {
        imagetteSocket.setsockopt(ZMQ_IDENTITY, identity.c_str(), identity.length());
        imagetteSocket.connect(imagetteAddress);
    }

    // connect to the window position server, if it is specified in the inputfile

    if (getWindowPositionFromServer)
    {  
      	windowPositionSocket.setsockopt(ZMQ_IDENTITY, identity.c_str(), identity.length());
        windowPositionSocket.connect(windowPositionAddress);
        windowPositionSocket.setsockopt(ZMQ_RCVTIMEO, &windowPositionSocketTimeout, sizeof(windowPositionSocketTimeout));
    }

    firstExposure = true;

}



/**
 * \brief get needed variables from the input file
 *        
 */
void ClosedLoopUtility::configure(ConfigurationParameters &configParams)
{

    sendImagettesToClient           = configParams.getBoolean("ControlTcpConnection/SendImagettesToClients");
    getWindowPositionFromServer     = configParams.getBoolean("ControlTcpConnection/GetWindowPositionsFromServer");
    imagetteAddress                 = configParams.getString("ControlTcpConnection/ImagetteClientAddress");
    windowPositionAddress           = configParams.getString("ControlTcpConnection/WindowPositionServerAddress");
    windowPositionSocketTimeout     = configParams.getInteger("ControlTcpConnection/WindowPositionSocketTimeout");

    if (windowPositionSocketTimeout < 0)
    {
        windowPositionSocketTimeout = -1;
    }
    else
    {
        windowPositionSocketTimeout *= 1000;
    }

}

/**
 * \brief send a specific string to a specific socket
 *        
 */
void ClosedLoopUtility::sendMessageToSocket(const std::string messageString, zmq::socket_t* socketPointer)
{
    // define the message which is to be send
    zmq::message_t message(messageString.length());

    const char *cMessage = messageString.c_str(); 

    // convert the string to a zmq message
    memcpy (message.data (), cMessage, messageString.length());

    // send the message over the socket
    socketPointer->send(message);
}

/**
 * \brief receive a specific string from a specific socket
 *        
 */
std::string ClosedLoopUtility::receiveMessageFromSocket(zmq::socket_t* socketPointer)
{
    // define the message which is to be received
    zmq::message_t reply;

    std::string replyString = "";

    // get the message from the socket
    windowPositionSocket.recv(&reply);

    // convert the reply to a string
    replyString = std::string(static_cast<char*>(reply.data()), reply.size());

    return replyString;
}


/**
 * \brief split a string at empty spaces or commas and return a vector of double with the content
 *        
 */
std::vector<double> ClosedLoopUtility::convertStringToDoubleVec(const std::string message)
{
    // fracture the received string
    std::stringstream ss(message);
        
    std::vector<double> doubleVec;
        
    double i;
       
    // convert the string to double values and save them in a vector
    while (ss >> i)
    {
        doubleVec.push_back(i);
            
        if (ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }
    }

    return doubleVec;
}


/**
 * \brief get the new windo position and return a tuple consisting of bool newPosition, uint numRowsPixelMap, 
 *        uint numColumnsPixelMap, uint subFieldZeroPointRow, uint subFieldZeroPointColumn, double orientationAngle
 */
std::tuple<bool, uint, uint, uint, uint, double> ClosedLoopUtility::getNewWindowPosition(double exposureTime)
{
    // check wether it is the first exposure

    if (firstExposure)
    {
        // formulate a message to the server signaling that the simulation is ready
        // and send the exposureTime to the server

        std::string exposureTimeString = std::to_string(exposureTime);

        sendMessageToSocket(exposureTimeString, &windowPositionSocket);

        // wait for the answer of the window position server
        std::string replyString = receiveMessageFromSocket(&windowPositionSocket);

        std::vector<double> windowPositionVec = convertStringToDoubleVec(replyString);

        if(windowPositionVec.size() == 5)
        {
            Log.info("ClosedLoopUtility: got first win position message");

            // if the window position is set the simulation can start

            firstExposure = false;

            // set the socket to no longer wait for input from server

            uint timeOut = 0;

            windowPositionSocket.setsockopt(ZMQ_RCVTIMEO, &timeOut, sizeof(timeOut));

            // return the received message
            return std::make_tuple(true,
                                   uint(windowPositionVec.at(0)),
                                   uint(windowPositionVec.at(1)),
                                   uint(windowPositionVec.at(2)),
                                   uint(windowPositionVec.at(3)),
                                   windowPositionVec.at(4));

        }
        else
        {
            // end simulation

            exit(1);
        }
    }
    else
    {
        // check whether a new step was received
        std::string windowPositionMessage = receiveMessageFromSocket(&windowPositionSocket);
        
        std::vector<double> windowPositionVec = convertStringToDoubleVec(windowPositionMessage);

        if(windowPositionVec.size() == 5)
        {
            Log.info("ClosedLoopUtility: got window position message");


            return std::make_tuple(true,
                                   uint(windowPositionVec.at(0)),
                                   uint(windowPositionVec.at(1)),
                                   uint(windowPositionVec.at(2)),
                                   uint(windowPositionVec.at(3)),
                                   windowPositionVec.at(4));
        }
        else
        {
            return std::make_tuple(false, 0, 0, 0, 0, 0.0);
        }

    }
}


/**
 * \brief send the created pixelmap to the imagette client
 *        
 */
void ClosedLoopUtility::sendImagetteToClient(arma::Mat<float>* pixelMapPointer, uint exposureNumber)
{

    // convert the pixel map to a string

    std::string pixelMapString = convertMatrixToString(pixelMapPointer, exposureNumber);

    // send the pixelMapString to the imagetteSocket

    sendMessageToSocket(pixelMapString, &imagetteSocket);

}

/**
 * \brief convert a arma mat to a linear, row major string seperated by empty spaces  
 *        the string starts with the imagette number, its rows and cols
 */
std::string ClosedLoopUtility::convertMatrixToString(arma::Mat<float>* pixelMapPointer, const uint exposureNumber)
{
    int rows = pixelMapPointer->n_rows;

    int cols = pixelMapPointer->n_cols;

    // write the values seperated by a white space to the string
    std::string imagetteString = to_string(exposureNumber) + " " + to_string(rows) + " " + to_string(cols) + " ";

    // write every value of the pixelMap to the string
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            imagetteString += to_string(int(pixelMapPointer->at(i, j))) + " ";
        }
    }

    return imagetteString;
}

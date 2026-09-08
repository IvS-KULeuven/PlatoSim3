# a script to test whether platoSim produces the same results independent whether the jitter and 
# position information are comming from file or are sent via zmq connection  

import os
import numpy as np
import subprocess

import zmq
import h5py
import yaml

import time


# get the platosim path

inputDir            = os.getenv("PLATO_PROJECT_HOME")

# get the relevant input and and jitter file

offlineInputFile    = inputDir + "/python/Examples/closedLoopTest/offlineInput.yaml"
onlineInputFile     = inputDir + "/python/Examples/closedLoopTest/onlineInput.yaml"
jitterFile          = inputDir + "/python/Examples/closedLoopTest/shortJitter.txt"

# set the output file names

offlineOutputFile   = "offline.hdf5"
onlineOutputFile    = "online.hdf5"

# load the input data from offline.yaml

offlineInput = open(offlineInputFile)

inputDataMap = yaml.safe_load(offlineInput)

offlineInput.close()

imagetteNumber = inputDataMap["ObservingParameters"]["NumExposures"]


#start an offline simulation

subprocess.Popen(inputDir + "/build/platosim " + offlineInputFile + " " + offlineOutputFile, shell=True)

# get jitter steps from file

with open(jitterFile) as f:
    jitterLine = f.readlines()



# define the sockets on which information is shared with the platosim instance

context             = zmq.Context()

inputSocket         = context.socket(zmq.ROUTER)
inputSocket.bind("tcp://*:5558")

jitterSocket        = context.socket(zmq.PUB)
jitterSocket.bind("tcp://*:5559")

imagetteSocket      = context.socket(zmq.ROUTER)
imagetteSocket.bind("tcp://*:5560")



# initialize poller

poll = zmq.Poller()
poll.register(inputSocket, zmq.POLLIN)
poll.register(imagetteSocket, zmq.POLLIN)

# start an online simulation

subprocess.Popen(inputDir + "/build/platosim " + onlineInputFile + " " + onlineOutputFile + " log2.txt", shell=True)

imagetteList = []

imagetteCounter = 0

# get some information from the offlineInput yaml file

zeroPointRow = inputDataMap["SubField"]["ZeroPointRow"]

zeroPointColumn = inputDataMap["SubField"]["ZeroPointColumn"]

orientation = inputDataMap["CCD"]["Orientation"]

cols = inputDataMap["SubField"]["NumColumns"]

rows = inputDataMap["SubField"]["NumRows"]

# wait for messages from the platosim instance

while (imagetteCounter != imagetteNumber):
    
    print("wait for platosim message")

    socks = dict(poll.poll(10))

    if socks.get(inputSocket):

        identity = inputSocket.recv()
        message = inputSocket.recv()
        print("received platosim ready message")

        # set the reply string

        replyStr = str(rows) + " " + str(cols) + " " + str(zeroPointColumn) + " " + str(zeroPointRow) + " " + str(orientation)

        # send a window postion to platosim

        inputSocket.send_multipart([identity, replyStr.encode()])

        # send 20 jitter steps

        print ("send 20 jitter steps")

        for i in range((imagetteCounter * 20), (imagetteCounter * 20 + 20)):

            jitterStep = jitterLine[i].replace('\t', ' ')

            jitterSocket.send_multipart([identity, jitterStep.encode()])


    if socks.get(imagetteSocket):

        identity = imagetteSocket.recv()
        imagette = imagetteSocket.recv()

        print ("received imagette")

        imagetteList.append(imagette)

        imagetteCounter += 1

        if(imagetteCounter != imagetteNumber):

            # send 20 jitter steps

            print ("send 20 jitter steps")

            for i in range((imagetteCounter * 20), (imagetteCounter * 20 + 20)):

                jitterStep = jitterLine[i].replace('\t', ' ')

                jitterSocket.send_multipart([identity, jitterStep.encode()])

        else:
            
            print ("send last jitter step")
            
            jitterSocket.send_multipart([identity, "".encode()])

            # receive the last imagette just to ignore it

            identity = imagetteSocket.recv()
            imagette = imagetteSocket.recv()

    time.sleep(1)

imagetteIntList = []



unEqual = False

# read the imagettes from the offline hdf5 file

with h5py.File(offlineOutputFile, 'r') as f:

    for i in range(0, 10):

        imagetteString = "image" + "%06d" % (i,)

        imagette = np.array(f['Images'][imagetteString])

        intImagette = imagetteList[i].split()

        for x in range(0, cols):

            for y in range(0, rows):

                offInt = imagette[x][y]

                index = 3 + x * cols + y

                onInt = intImagette[index]

                # set the error status to true and print out the wrong imagettes number

                if int(onInt) != int(offInt):
                   
                    unEqual = True

                    print("Errors occured with imagette: ")

                    print(i)



if unEqual == True:
    
    print("Errors occured!")

else:

    print("Everything is fine!")


os.remove(offlineOutputFile)

os.remove(onlineOutputFile)












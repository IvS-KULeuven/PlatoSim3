#!/usr/bin/env python3

"""
Run the PLATO Simulator from Python.

The Simulation class provides a Python interface to create script that
interactively can tune the input parameters before launching PlatoSim.
The parameters that are available can be inspected by just printing the
Simulation object, i.e. print (sim), which will dump all the parameters
and their current values on the command line.

For usage, see the Jupyter notebook tutorials available at:
PlatoSim/docs/tutorials.
"""

# Built-in
import os
import sys
import ast
import math
import datetime
import subprocess
import warnings

# PlatoSim standard
import yaml
import pyaml
import numpy as np

# PlatoSim functions
import platosim.utilities       as ut
import platosim.referenceFrames as rf
from platosim.simfile import SimFile


#==============================================================#
#                         BEGIN CLASS                          #
#==============================================================#


class Simulation(object):

    """Class for running PlatoSim simulations.

    Simulation class allows running the PLATO simulator interactively from Python
    and tuning the input parameters before each run. For more help, type:

    Example
    -------
    >>> import platosim.simulation as Simulation
   >>> print(Simulation)
    """


    def __init__(self, runName, configurationFile=None, outputDir=None, debug=False):

        """Initialise class variables.
        """

        self.debug   = debug
        self.runName = runName

        # Glag to check if output directory has been specified before running the Simulation

        self.hasTargetLocation = False
        self.targetOutputFilesLocation = None

        # Set output directory

        if outputDir is not None:
            self.outputDir = outputDir
        else:
            self.outputDir = os.getcwd()

        # Set simulation location
        
        self.setSimulatorLocation()

        # Read the YAML input file
        
        if configurationFile:
            self.readConfigurationFile( configurationFile )
        else:
            self.readConfigurationFile( self.originalInputFilesLocation + "/inputfile.yaml" )

        # Placeholders for True and False statements

        self.true  = [True, "yes", "1"]
        self.false = [False, "no", "0"]




        
    def setSimulatorLocation(self):

        """Set the location of the simulation.

        Given the location of the simfile.py module, try to find the build/ folder where
        the simulator executable should be. Then set the default locations for the platosim
        executable (i.e. build directory), and the original input files location.
        """

        # Find the absolute path of the simfile.py. This will allow us to located the other
        # default project directories. Build in a test to check that the build sub-directory
        # exists, otherwise this is probably not a correct default installation.

        path = os.environ["PLATO_PROJECT_HOME"]

        self.platosimLocation = path

        if os.path.exists(path + "/build"):
            self.platosimBuildLocation = self.platosimLocation + "/build"

        elif os.path.exists(path + "/bin"):
            self.platosimBuildLocation = self.platosimLocation + "/bin"

        else:
            raise Exception("Unexpected directory structure for this PlatoSim distribution: " +
                            "No build nor bin sub-directory in $PLATO_PROJECT_HOME")

        # This is the location of the original input files as distributed by the PLATO Simulator

        self.originalInputFilesLocation  = self.platosimLocation + "/inputfiles"
        self.originalOutputFilesLocation = self.platosimLocation + "/outputfiles"




        
    @property
    def outputDir(self):

        """Return the output files location.
        """
        return self.targetOutputFilesLocation




    
    @outputDir.setter
    def outputDir(self, path):

        """Specify the absolute path for the output directory. 

        This directory will contain a copy of the modified input file and
        the HDF5 output file for this simulation. If the output path do not
        exist, it is created.
        """

        if not os.path.exists(path):
            if self.debug:
                print(f"DEBUG: creating output directory {path}")
            self.createDirectory(path)

        self.targetOutputFilesLocation = path
        self.hasTargetLocation = True

        if self.debug:
            print(f"DEBUG: output directory set to {path}")



    def readConfigurationFile(self, filename):

        """Read the YAML input configuration file.
        """
        
        self.configurationFilename = filename

        if self.debug:
            print(f"DEBUG: Parsing YAML configuration file {filename}")

        with open(filename, 'r') as stream:
            try:
                self.yamlDocument = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)





    def getYamlConfiguration(self):

        """Return the YAML configuration as a dictionary.
        """
        
        return self.yamlDocument

    


    
    def showYamlConfiguration(self):

        """Return the YAML configuration as a dictionary.
        """

        YAML = self.getYamlConfiguration()
        print(pyaml.dump(YAML))




        
    def __contains__(self, key):

        """Returns true if the input parameter (key) is known/exists.

        Usage: if "Group/ParameterName" in sim: # do something

        Parameters
        ----------
        key : str
            A string containing the parameter name or "Group/ParameterName" combination.

        Return
        ------
        bool : True if key exists, False otherwise
        """
        
        if key.find('/') == -1:
            nodeNames = [key]
        else:
            nodeNames = key.split("/")

        node = self.yamlDocument

        for nodeName in nodeNames:
            print(f"> {nodeName}, {type(node)}")
            try:
                node = node[nodeName]
            except:
                return False

        return True





    def __getitem__(self, key):

        """Returns the value of the input parameter (key).

        Parameters
        ----------
        key : str
            A string containing the parameter name or "Group/ParameterName" combination.

        Return
        ------
        value : int, float, ndarray, str
            The value of the requested parameter.
        """

        # Split the path into node names
        # E.g. "PSF/MappedGaussian/Sigma" into [PSF, MappedGaussian, Sigma]

        if key.find('/') == -1:
            # FIXME: parentNodeName assigned but never used
            parentNodeName, nodeName = key, None
            print("Usage: the given parameter name (key) should include the group name " +
                  "of the group that contains the parameter. E.g in 'Camera/PlateScale', " +
                  "'Camera' is the group, 'PlateScale' is the parameter.")
            return None
        else:
            nodeNames = key.split("/")

        # Navigate to the deepest node, starting from the document root

        node = self.yamlDocument

        for nodeName in nodeNames:
            if nodeName in node:
                node = node[nodeName]
            else:
                print(f"ERROR: The group '{key}' was not found in the " +
                      f"yaml inputfile '{self.configurationFilename}'")
                return None

        # Node is a string, so cast it to its proper value

        try:
            value = ast.literal_eval(node)
        except Exception:
            
            value = node

        # Return the value of the deepest node

        return value





    def __setitem__(self, key, item):

        """Update a specific node.

        Parameters
        ----------
        key : str
            A string with parent node name and node name seperated by a slash.
        item : str
            A string with the new node value, if not a string the value is converted using str().

        Return
        ------
        bool : True if node could be updated, False otherwise
        """

        # Ensure that the given item is a string

        if type(item) is np.ndarray:
            item = list(item)

        if type(item) != list:
            item = str(item)

        # Split the path into node names
        # E.g. "PSF/MappedGaussian/Sigma" into ["PSF", "MappedGaussian", "Sigma"]

        if key.find('/') == -1:
            print ("USAGE: the given parameter name (key) should include the " +
                   "group name of the group that contains the parameter.")
            print ("       E.g in 'Camera/PlateScale', Camera is the group," +
                   "PlatScale is the parameter.")
            return None
        else:
            nodeNames = key.split("/")

        # Check whether the parent node is in the document. If not, complain

        if nodeNames[0] not in self.yamlDocument:
             print(f"ERROR: no node with the name {nodeNames[0]} found in input yaml file")
             return False

        # If there is only 1 node in the path, we're finished after setting its value

        if len(nodeNames) == 1:
            self.yamlDocument[nodeNames[0]] = item
            return True

        # If we arrive here, there are at least 2 node in the path,
        # check if 2nd parent node exists

        if nodeNames[1] not in self.yamlDocument[nodeNames[0]]:
             print("ERROR: no node with the name " +
                   f"{nodeNames[0]}/{nodeNames[1]} found in input yaml file")
             return False

        # If there are only 2 nodes in the path, we're finished after setting its value

        if len(nodeNames) == 2:
            self.yamlDocument[nodeNames[0]][nodeNames[1]] = item
            return True

        # If we arrive here, there are at least 3 nodes in the path,
        # check if 3rd parent node exists

        if nodeNames[2] not in self.yamlDocument[nodeNames[0]][nodeNames[1]]:
             print("ERROR: no node with the name " +
                   f"{nodeNames[0]}/{nodeNames[1]}/{nodeNames[2]} found in input yaml file")
             return False

        # If there are only 3 nodes in the path, we're finished after setting its value

        if len(nodeNames) == 3:
            self.yamlDocument[nodeNames[0]][nodeNames[1]][nodeNames[2]] = item
            return True

        # If we arrive here, there are at least 4 nodes in the path, check if 4th parent node exists

        if nodeNames[3] not in self.yamlDocument[nodeNames[0]][nodeNames[1]][nodeNames[2]]:
             print("ERROR: no node with the name " +
                   f"{nodeNames[0]}/{nodeNames[1]}/{nodeNames[2]}/{nodeNames[3]} " +
                   "found in input yaml file")
             return False

        # If there are only 4 nodes in the path, we're finished after setting its value

        if len(nodeNames) == 4:
            self.yamlDocument[nodeNames[0]][nodeNames[1]][nodeNames[2]][nodeNames[3]] = item
            return True


        # If we arrive here, there are at least 5 nodes in the path, check if 5th parent node exists

        if nodeNames[4] not in self.yamlDocument[nodeNames[0]][nodeNames[1]][nodeNames[2]][nodeNames[3]]:
             print("ERROR: no node with the name " +
                   f"{nodeNames[0]}/{nodeNames[1]}/{nodeNames[2]}/{nodeNames[3]}/{nodeNames[4]} " +
                   "found in input yaml file")
             return False

        # If there are only 5 nodes in the path, we're finished after setting its value

        if len(nodeNames) == 5:
            self.yamlDocument[nodeNames[0]][nodeNames[1]][nodeNames[2]][nodeNames[3]][nodeNames[4]] = item
            return True

        # If we arrive here, there are at least 6 nodes in the path.
        # Issue a not-implemented error message.

        print(f"Error: Not Implemented: detected 6 or more nodes in the path {key}")
        return False





    def createDirectory(self, path):

        """Create a directory.
        """

        try:
            os.makedirs(path)
        except OSError as ose:
            print (ose)
            if not os.path.isdir(path):
                raise Exception(f"Could not create directory {path}")

        return





    def writeYamlConfigurationFile(self, filename):

        """Write the modified configuration to output file location. 

        This configuration will be loaded by the PLATO Simulator when
        the run() method is executed.

        Parameter
        ---------
        filename : str
            Filename of the output YAML file.
        """
        
        if self.debug:
            print(f"DEBUG: writing the YAML configuration file {filename}")
        with open(filename, 'w') as outfile:
            outfile.write( pyaml.dump(self.yamlDocument, indent=4, width=120) )





    def run(self, removeOutputFile=False, executionTime=False, logLevel=3):

        """Run the PLATO Simulator.

        Parameters
        ----------
        removeOutputFile : bool
            If the outputfile already exists before the run started, simply delete it.
        logLevel : int
            Level of verbosoty: 1 (least verbose) to 3 (most verbose)
        
        Return
        ------
        When PlatoSim fails for some reason and returns an error code (!= 0), an Exception
        is raised.
        """

        if executionTime:
            tic = datetime.datetime.now()

        if not self.hasTargetLocation:
            raise Exception("Output location not set for this Simulation. " +
                            "Set the outputDir before executing the run() method.")

        inputFilename  = f"{self.targetOutputFilesLocation}/{self.runName}.yaml"
        outputFilename = f"{self.targetOutputFilesLocation}/{self.runName}.hdf5"
        logFilename    = f"{self.targetOutputFilesLocation}/{self.runName}.log"

        if removeOutputFile:
            try:
                os.remove(outputFilename)
            except OSError:
                pass

        self.writeYamlConfigurationFile(inputFilename)

        # The run() method was only introduced with Python 3.5.
        # Use the older call() method when running e.g. Python 2.7

        if sys.version_info < (3, 5):
            rc = subprocess.call([self.platosimBuildLocation + "/platosim", inputFilename,
                                  outputFilename, logFilename, str(logLevel)])
            if rc:
                raise Exception(f"Simulation.run(): PlatoSim returned with exit code {rc}.")
        else:
            completedProcess = subprocess.run([self.platosimBuildLocation + "/platosim",
                                               inputFilename, outputFilename, logFilename,
                                               str(logLevel)],
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if completedProcess.returncode:
                print(str(completedProcess.stdout.decode("utf-8")))
                print(str(completedProcess.stderr.decode("utf-8")))
                raise Exception("Simulation.run(): PlatoSim returned with " +
                                f"exit code {completedProcess.returncode}.")

        # Print computation time
        
        if executionTime:
            toc = datetime.datetime.now()
            print(f"Execution time : {toc - tic} [hh:mm:ss]")

        return SimFile(outputFilename)





    def __str__(self):

        """Return a listing of all settings.
        
        This function fetch a listing of all settings from both the CCD
        and the Photometry input files. If a parameter value has been
        updated for this Simulation, [updated] will be printed after the
        value.
        """
        
        root = self.yamlDocument
        msg = "YAML Configuration:\n"
        msg += pyaml.dump(root, indent=4)

        return msg





    def turnOffAllOutput(self):

        """Function to write nothing to the HDF5 file.
        """

        # Fetch names of ControlHDF5Content attributes

        group = "ControlHDF5Content"
        entries = []
        for name, dict_ in self.yamlDocument[group].items():
            entries.append(name)

        # Control the content
        
        for entry in entries:
            self.__setitem__(f"{group}/{entry}", False)

            



    def turnOnAllOutput(self):

        """Function to write all to the HDF5 file.
        """

        # Fetch names of ControlHDF5Content attributes

        group = "ControlHDF5Content"
        entries = []
        for name, dict_ in self.yamlDocument[group].items():
            entries.append(name)

        # Control the content

        for entry in entries:
            self.__setitem__(f"{group}/{entry}", True)
                




    def showAllOutput(self):

        """Function to write all to the HDF5 file.
        """

        # Fetch names of ControlHDF5Content attributes

        group = "ControlHDF5Content"
        entries = []
        for name, dict_ in self.yamlDocument[group].items():
            entries.append(name)

        # Show the control content

        for entry in entries:
            switch = self.__getitem__(f"{group}/{entry}")
            print(f"{group}/{entry} = {switch}")




            
    def controlAllEffects(self, switch):

        """Function to write all or nothing to the HDF5 file.

        Parameters
        ----------
        switch : str, bool
            If 'yes' ('True')  turn on  all effects
            If 'no'  ('False') turn off all effects
        
        Notes
        -----
        Parameters that are turned on/off:
        - Cosmic rays
        - AOCS jitter
        - Thermo-elastic drift (TED)
        - Aberration correction (absolute + differential)
        - Field distortion
        - Charge diffusion
        - Jitter smoothing
        - Flatfield
        - Dark signal
        - Brighter-fatter effect (BFE)
        - Photon noise
        - Readout noise
        - Charge transfer inefficiency (CTI)
        - Open-shutter smearing
        - Overall relative transmissivity
        - Polarisation
        - Particulate contamination
        - Molecular contamination
        - Quantum efficiency
        - Convolution with the PSF (Zemax only)
        - Full-well saturation (blooming)
        - Digital saturation
        - Quantisation
        """
        
        # Sky parameters

        self["Sky/IncludeCosmicsInSubField"]    = switch
        self["Sky/IncludeCosmicsInSmearingMap"] = switch
        self["Sky/IncludeCosmicsInBiasMap"]     = switch

        # Platform parameters

        self["Platform/UseJitter"] = switch

        # Telescope parameters

        self["Telescope/UseDrift"] = switch

        # Camera parameters

        self["Camera/IncludeAberrationCorrection"] = switch
        self["Camera/IncludeFieldDistortion"]      = switch
        self["Camera/IncludePointLikeGhosts"]      = switch
        self["Camera/IncludeExtendedGhosts"]       = switch

        # PSF parameters

        self["PSF/MappedFromFile/IncludeChargeDiffusion"]      = switch
        self["PSF/AnalyticNonGaussian/IncludeChargeDiffusion"] = switch
        self["PSF/MappedFromFile/IncludeJitterSmoothing"]      = switch

        # FEE parameters

        self["FEE/IncludeOverAndUnderShoot"] = switch
        
        # CCD parameters

        self["CCD/IncludeFlatfield"]                = switch
        self["CCD/IncludeDarkSignal"]               = switch
        self["CCD/IncludeBFE"]                      = switch
        self["CCD/IncludePhotonNoise"]              = switch
        self["CCD/IncludeReadoutNoise"]             = switch
        self["CCD/IncludeCTIeffects"]               = switch
        self["CCD/IncludeChargeInjection"]          = switch
        self["CCD/IncludeOpenShutterSmearing"]      = switch
        self["CCD/IncludeQuantumEfficiency"]        = switch
        self["CCD/IncludeRelativeTransmissivity"]   = switch
        self["CCD/IncludePolarization"]             = switch
        self["CCD/IncludeParticulateContamination"] = switch
        self["CCD/IncludeMolecularContamination"]   = switch
        self["CCD/IncludeConvolution"]              = switch
        self["CCD/IncludeFullWellSaturation"]       = switch
        self["CCD/IncludeDigitalSaturation"]        = switch
        self["CCD/IncludeQuantisation"]             = switch
        self["CCD/IncludeGainNonlinearity"]         = switch

        # Straylight
        self["StrayLight/IncludeStraylight"]        = switch




        
    def turnOnAllEffects(self):

        """Function to switch off all writing to the HDF5 file.

        NOTE: function uses controlAllEffects()
        """

        self.controlAllEffects("yes")




        
    def turnOffAllEffects(self):

        """Function to switch off all writing to the HDF5 file.

        NOTE: function uses controlAllEffects()
        """

        self.controlAllEffects("no")





    def useDetectorGain(self, performance="required"):

        """Change the detector gain.

        The parameters are from the Mission Parameter Database (MPD):
        http://ptoops02.esac.esa.int/mpdb/home

        Notes
        -----
        * A mean value of the CCD gain min/max and F/E side value are used from the MPD
        * The FEE gain is then calculated requiring that the:
          Total gain = 1 / (gainFFE * gainCCD) = 25 e-/ADU
        """
        
        if performance == "required":
            # CCD gain F/E side: (min, max) = (1.8, 2.5)
            self.__setitem__("CCD/Gain/RefValueLeft",  "2.15")    # [microV/e-]
            self.__setitem__("CCD/Gain/RefValueRight", "2.15")    # [microV/e-]
            self.__setitem__("FEE/Gain/RefValueLeft",  "0.0186")  # [ADU/microV]
            self.__setitem__("FEE/Gain/RefValueRight", "0.0186")  # [ADU/microV]

        elif performance in ["expected", "designed"]:
            # CCD gain F side (min, max) = (2.08, 2.28) -> 2.18 microV/e- 
            # CCD gain E side (min, max) = (2.04, 2.26) -> 2.15 microV/e-
            self.__setitem__("CCD/Gain/RefValueLeft",  "2.18")
            self.__setitem__("CCD/Gain/RefValueRight", "2.15")
            self.__setitem__("FEE/Gain/RefValueLeft",  "0.018348")
            self.__setitem__("FEE/Gain/RefValueRight", "0.0186")

        else:
            raise ValueError("Not valid performance entry! " +
                             "Use either ['required', 'expected', 'designed']")
        
        return






    def useTimeDependentDetectorNoise(self, performance="required", timeFromBOL=0,
                                      camera="Normal"):

        """Change noise properties of CCD/FEE.
        
        The parameters are from the Mission Parameter Database:
        http://ptoops02.esac.esa.int/mpdb/home

        Notes
        -----
        - Assumption that parameters are the same for F and E side of CCD
        - Assumption that the dark signal for N-CAM and F-CAM (F and E side) are the same        
        - NOTE: 'required' or 'designed' F-FEE readout noise EOL missing
        - NOTE: 
        """
        
        times = [0, self["ObservingParameters/MissionDuration"]*ut.year()]

        # FEE readout noise [e-] -> independent on performance
        
        if camera == "Normal":
            readNoiseFEE = ut.evalLinReg(times, np.array([32.8, 37.7]), timeFromBOL)
        elif camera == "Fast":
            readNoiseFEE = 203.0
        else:
            raise ValueError("Not valid entry! Usage in ['Normal', 'Fast']")

        # Choose performance
        
        if performance == "required":
            readNoiseCCD  = ut.evalLinReg(times, np.array([24.5, 28.0]), timeFromBOL)
            darkCurrent   = ut.evalLinReg(times, np.array([ 1.2,  4.5]), timeFromBOL)
            darkStability = 5.0
            DSNU          = 15.0

        elif performance in ["expected", "designed"]:
            readNoiseCCD  = ut.evalLinReg(times, np.array([23.2,   25.0]), timeFromBOL)
            darkCurrent   = ut.evalLinReg(times, np.array([ 0.544,  4.0]), timeFromBOL)
            darkStability = 0.7
            DSNU          = 13.0
            
        else:
            raise ValueError("Not valid performance entry! " +
                             "Use either ['required', 'expected', 'designed']")

        # Set all parameters from above
        
        self.__setitem__("CCD/DarkSignal/DarkCurrent", f"{darkCurrent}")
        self.__setitem__("CCD/DarkSignal/Stability",   f"{darkStability}")
        self.__setitem__("CCD/DarkSignal/DSNU",        f"{DSNU}")
        self.__setitem__("CCD/ReadoutNoise",           f"{readNoiseCCD}")
        self.__setitem__("FEE/ReadoutNoise",           f"{readNoiseFEE}")

        return
    



    
    def useNormalCamera(self, performance=False, timeFromBOL=0):

        """Change the input parameters to use the N-CAM.
        
        The parameters are from the Mission Parameter Database:
        http://ptoops02.esac.esa.int/mpdb/home
        """

        self.__setitem__("CCD/NumColumns",                "4510")
        self.__setitem__("CCD/NumRows",                   "4510")
        self.__setitem__("ObservingParameters/CycleTime", "25")

        # NOTE These are the same for all requirements

        self.__setitem__("Camera/ThroughputLambdaC",   "750")        
        self.__setitem__("Camera/ThroughputBandwidth", "500")
        
        # If requested, select basic input parameters from MPD

        if performance in ["required", "expected", "designed"]:
            self.useDetectorGain(performance)
            self.useTimeDependentDetectorNoise(performance, timeFromBOL)

        # Other wavelength dependent parameters
            
        if performance == 'required':
            self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.8125")
            self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.7941")
            self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.6267")

        elif performance == 'expected':
            self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.8752")
            self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.8752")
            # TODO No QE values for the "as designed" scenario in MPDB! We use "as designed":
            self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.6552")

        elif performance == 'designed':
            self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.8700")
            self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.8700")
            self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.6552")
     
        return





    def useFastCamera(self, passband, performance=False, timeFromBOL=0):

        """Change the input parameters to use the F-CAM.

        Parameters
        ----------
        
        The parameters are from the Mission Parameter Database:
        http://ptoops02.esac.esa.int/mpdb/home
        
        Notes
        -----
        Effects that are unique for the F-CAMs:
        - FEE overshoot/undershoot (TODO not working properly yet)

        The following parameters are wavelenght dependent:
        - Photometric reference flux (and irradiance -> only PIS)
        - Tranmission efficiency (built-in linear model given BOL->EOL)
        - Throughput bandwidth
        - Central wavelength of the throughput passband
        - Quantum efficiency
        """

        self.__setitem__("Telescope/GroupID",             "Fast")
        self.__setitem__("CCD/NumColumns",                "4510")
        self.__setitem__("CCD/NumRows",                   "2255")
        self.__setitem__("ObservingParameters/CycleTime", "2.5")

        # If requested, set parameters that are independent of the filter
        
        if performance in ["required", "designed"]:

            # Select noise properties for CCD and FEE
            
            self.useDetectorGain(performance)
            self.useTimeDependentDetectorNoise(performance, timeFromBOL, camera="Fast")
        
        # Select time and wavelength dependent parameters
        # NOTE the 'as designed' tranmission efficiency is identical for both F-CAMs
        
        if passband == "blue":

            # Parameters that are the same for all requirements
            self.__setitem__("ObservingParameters/Fluxm0", "1.195e8")
            
            if performance == 'required':
                self.__setitem__("Camera/ThroughputLambdaC",   "578")
                self.__setitem__("Camera/ThroughputBandwidth", "145")
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.6745")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.6549")
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.8505")     
                
            elif performance == 'expected':
                self.__setitem__("Camera/ThroughputLambdaC",   "585")                
                self.__setitem__("Camera/ThroughputBandwidth", "165")                
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.6566")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.6566")
                # TODO No QE values for the "as expected" scenario in MPDB! We use designed:   
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.8986")

            elif performance == 'designed':
                self.__setitem__("Camera/ThroughputLambdaC",   "600")
                self.__setitem__("Camera/ThroughputBandwidth", "200")
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.8578")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.8578")
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.8986")
                
        elif passband == "red":

            # Parameters that are the same for all requirements
            self.__setitem__("ObservingParameters/Fluxm0", "0.923e8")
            self.__setitem__("Camera/ThroughputLambdaC",   "833")                
            self.__setitem__("Camera/ThroughputBandwidth", "335")
           
            if performance == 'required':
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.6980")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.6845")
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.5184")

            elif performance == 'expected':
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.7266")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.7266")
                # TODO No QE values for the "as expected" scenario in MPDB! We use designed:
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.5344")
                
            elif performance == 'designed':
                self.__setitem__("Telescope/TransmissionEfficiency/BOL",        "0.8604")
                self.__setitem__("Telescope/TransmissionEfficiency/EOL",        "0.8604")
                self.__setitem__("CCD/QuantumEfficiency/MeanQuantumEfficiency", "0.5344")

        else:
            errorcode('warning', 'Wrong passband name, use either "blue" or "red"!')    
            
        return




    
    def setSubfieldAroundPixelCoordinates(self, ccdCode,
                                          xCCDpixel, yCCDpixel,
                                          subfieldSizeX, subfieldSizeY,
                                          normal=None):

        """Set the subfield around pixel coordinates.
        
        This function calculate the location of the subField such that it
        is centered on the star with the given pixel coordinates.
        
        Parameters
        ----------
        ccdCode : str
            For nominal camera: either '1', '2', '3', '4'
            For fast camera: either '1F', '2F', '3F', '4F'
        xCCDpixel : int
            X-coordinate (column-number) of the star on the CCD [pixel/float]
        yCCDpixel : int
            Y-coordinate (row-number) of the star on the CCD [pixel/float]
        subfieldSizeX : int
            Width (i.e. number of columns) of the sub-field [pixels]
        SubfieldSizeY : int
            Feight (i.e. number of rows) of the sub-field [pixels]

        Return
        ------
        success: boolean, if the subfield setting was successful or not
        """

        raStar, decStar = rf.pixelToSkyCoordinates(self, ccdCode, xCCDpixel, yCCDpixel)
        success = self.setSubfieldAroundSkyCoordinates(raStar, decStar,
                                                       subfieldSizeX, subfieldSizeY,
                                                       normal)
        return success





    def setSubfieldAroundPixelRow(self, ccdCode, yCCDpixel, subfieldSizeY):

        """Set subfield around a pixel row.

        This function sets the location of the subfield so that its rows are
        centered around the given pixel coordinate.
        
        Paramters
        ---------
        ccdCode : str
            For nominal camera: either '1', '2', '3', '4'.
            For fast camera: either '1F', '2F', '3F', '4F'.
        yCCDpixel : int
            Y-coordinate (row-number) around which to center the subField  [pixel/float].
        subfieldSizeY : int
            Height (i.e. number of rows) of the sub-field [pixels].

        Return
        ------
        None

        Rmarks
        ------
        If it is not possible to center the subfield around the row, because the row
        is too close the the edge, the subfield will be set at that respective edge.
        """

        subfieldSizeX = int(self["SubField/NumColumns"])
        subfieldRowZero = int(self["SubField/ZeroPointColumn"])
        xCCDpixel = subfieldRowZero + subfieldSizeX / 2

        if not (0 <= yCCDpixel <  4510):
            print("Error: we expect input row coordinate in [0, 4510], " +
                  f"but value {yCCDpixel} was given.")
            return

        if not (1 <= subfieldSizeY <= 4510):
            print("Error: we expect size of the row subfield in [1, 4510], " +
                  f"but value {subfieldSizeY} was given.")
            return

        self.setSubfieldAroundPixelCoordinates(ccdCode, xCCDpixel, yCCDpixel, 1, 1)

        deltaY = int(subfieldSizeY / 2)
        self["SubField/NumColumns"] = subfieldSizeX
        self["SubField/NumRows"]    = subfieldSizeY

        self["SubField/ZeroPointColumn"] = subfieldRowZero
        self["SubField/ZeroPointRow"]    = min(max(0, self["SubField/ZeroPointRow"] - deltaY),
                                               4510 - subfieldSizeY)





    def setSubfieldAroundSkyCoordinates(self, raStar, decStar,
                                        subfieldSizeX, subfieldSizeY,
                                        normal=None, ccd=None,
                                        returnInfo=False):

        """Set subfield around stellar coordinates

        Set the location of the sub-field such that it is centred on the star
        with the given sky coordinates.  Depending on the CCD (in nomincal mode:
        "1", "2", "3", or "4"; in fast mode: "1F", "2F", "3F", or "4F"), the
        configuration file for the given simulation is adapted.  These include the
        pre-defined CCD position, the dimensions of the CCD (and also of the subfield,
        although this is not affected by the calculations), the sub-field zeropoint
        and the exposure time.

        Notes
        -----
        - This function calls the referenceFrames.calculateSubfieldAroundCoordinates().
        - It is assumed that the configuration parameters in the sim object contains:
          * A correct platform pointing {alpha, delta, kappa}
          * A correct telescope pointing {tilt, azimuth{
          * A valid values for the {focal length, late scale, pixel size}
          * That the switch to include distortion or not is set correctly
        - The function does not set the exposure time, nor the focal length source, etc.

        Parameters
        ----------
        raStar : float
            Right ascension of the star [radians]
        decStar : float
            Declination of the star [radians]
        subfieldSizeX : int
            Width (i.e. number of columns) of the subiield [pixels]
        subfieldSizeY : int
            Height (i.e. number of rows) of the sub-field [pixels]
        normal : bool
            Is depricated and should no longer be used.
        ccd : dict
            Dictionary with CCD configuration (see referenceFrames.CCD() dict)
        returnInfo : bool
            See Return section below.

        Return
        ------
        If returnInfo is False : bool
            True if subfield fit on one of the CCDs. Else False.
        If returnInfo is True : list
            List containing [ccdCode [str], xCCD [intra-pixel], yCCD [intra-pixel]] 
        """

        # Find the platform pointing [rad]

        if self["Platform/Orientation/Source"] == "Angles":
            alpha  = np.deg2rad(float(self["Platform/Orientation/Angles/RAPointing"]))
            delta = np.deg2rad(float(self["Platform/Orientation/Angles/DecPointing"]))
            kappa = np.deg2rad(self["Platform/Orientation/Angles/SolarPanelOrientation"])
        else:
            q_EQ2PLM = self["Platform/Orientation/Quaternion/Components"]
            alpha, delta, kappa = rf.platformAnglesFromQuaternion(q_EQ2PLM)

        # Find instrumental characteristics from the sim object
        
        telescopeGroupID = self["Telescope/GroupID"]
        if telescopeGroupID == "Custom":
            azimTelescope = np.deg2rad(float(self["Telescope/AzimuthAngle"]))
            tiltTelescope = np.deg2rad(float(self["Telescope/TiltAngle"]))
        elif telescopeGroupID == "Fast":
            azimTelescope = np.deg2rad(self["CameraGroups/AzimuthAngle"][4])
            tiltTelescope = np.deg2rad(self["CameraGroups/TiltAngle"][4])
        else:
            azimTelescope = np.deg2rad(self["CameraGroups/AzimuthAngle"][telescopeGroupID-1])
            tiltTelescope = np.deg2rad(self["CameraGroups/TiltAngle"][telescopeGroupID-1])

        focalLength     = float(self["Camera/FocalLength/ConstantValue"]) * 1000.0 # [m]->[mm]
        focalPlaneAngle =np.deg2rad(float(self["Camera/FocalPlaneOrientation/ConstantValue"]))
        pixelSize       = float(self["CCD/PixelSize"])
        normalCamera    = self["Telescope/GroupID"] != "Fast"

        # TODO This should be removed in the next major release, together with the normal
        #      argument in this function.
        
        if normal is not None:
            warnings.warn("\nThe optional argument 'normal' is depricated! It is now " +
                          "derived from the value of 'Simulation[Telescope/GroupID]'.",
                          category=DeprecationWarning, stacklevel=2)
            if not normal == normalCamera:
                warnings.warn("\nThe value for the argument 'normal' is not consistent with" +
                              " the one specified in the 'Simulation[Telescope/GroupID]'. " +
                              "This function will use the specified normal value, but keep " +
                              "in mind that this is different from value in the inputfile.",
                              category=DeprecationWarning, stacklevel=2)
                normalCamera = normal
                
        # If the PSF is MappedFromFile we need to include mapped field distortion

        if self["PSF/Model"] == "MappedFromFile":
            includeFieldDistortion = True
            mappedDistortion       = True
            pathToPsfFile          = self["PSF/MappedFromFile/Filename"]
            distortionCoefficients = None
        elif self["Camera/IncludeFieldDistortion"] in self.true:
            includeFieldDistortion = True
            mappedDistortion       = False
            pathToPsfFile          = None
            distortionCoefficients = self["Camera/FieldDistortion/ConstantCoefficients"]
        else:
            includeFieldDistortion = False
            mappedDistortion       = False
            pathToPsfFile          = None
            distortionCoefficients = None
            
        # Compute the position of the subfield. xPix and yPix are the CCD coordinates
        # of the star, given a 4510x4510 CCD [colNumber, rowNumber]. The function below
        # also checks if the subfield fits entirely on the CCD. If not: ccdCode is None.

        if ccd is None: ccd = rf.CCD()
        
        data = rf.calculateSubfieldAroundCoordinates(
            subfieldSizeX, subfieldSizeY,
            raStar, decStar,
            alpha, delta, kappa,
            tiltTelescope, azimTelescope,
            focalPlaneAngle, focalLength, pixelSize,
            includeFieldDistortion, normalCamera,
            mappedDistortion, distortionCoefficients,
            pathToPsfFile, ccd=ccd, returnFPA=returnInfo,
        )
        
        if data[0] == None:
            if returnInfo:
                return None, None, None, None, None
            else:
                return False
        else:
            ccdCode = data[0]
            xCCD    = data[1]
            yCCD    = data[2]
            if returnInfo:
                xFP = data[3]
                yFP = data[4]

        # If we arrive here, there is no problem accommodating the entire sufield on the CCD

        self["CCD/Position"] = str(ccdCode)

        # Drop suffix for F-CAM (1F, 2F,.. -> 1, 2,..)
        
        if not normalCamera:
            ccdCode = ccdCode[0]
        
        # Alter CCD geometry if requested
        
        if self["CCD/Position"] != "Custom":
            CCDOriginOffsetX = ccd[ccdCode]["zeroPointXmm"]
            CCDOriginOffsetY = ccd[ccdCode]["zeroPointYmm"]
            CCDOrientation   = ccd[ccdCode]["angle"]
            CCDSizeX         = ccd[ccdCode]["Ncols"]
            CCDSizeY         = ccd[ccdCode]["Nrows"]
            self["CCD/OriginOffsetX"] = str(CCDOriginOffsetX)
            self["CCD/OriginOffsetY"] = str(CCDOriginOffsetY)
            self["CCD/Orientation"]   = str(np.rad2deg(CCDOrientation))
            self["CCD/NumColumns"]    = CCDSizeX
            self["CCD/NumRows"]       = CCDSizeY

        self["Telescope/AzimuthAngle"] = np.rad2deg(azimTelescope)
        self["Telescope/TiltAngle"]    = np.rad2deg(tiltTelescope)

        if telescopeGroupID == "Fast":
            self["CCD/FirstRowExposed"] = str(2255)
        else:
            self["CCD/FirstRowExposed"] = str(0)

        self["SubField/NumRows"]    = str(subfieldSizeY)
        self["SubField/NumColumns"] = str(subfieldSizeX)
        
        # Secure that the target centrally for even-pixel subfields

        dy = yCCD - int(yCCD)
        dx = xCCD - int(xCCD)

        if (subfieldSizeY % 2 == 0):
            if dy >= 0.5:
                subfieldSizeY -= 1

        if (subfieldSizeX % 2 == 0):
            if dx >= 0.5:
                subfieldSizeX -= 1

        self["SubField/ZeroPointRow"]    = str(int(yCCD) - int(subfieldSizeY/2))
        self["SubField/ZeroPointColumn"] = str(int(xCCD) - int(subfieldSizeX/2))

        # That's it
        
        if returnInfo:
            return ccdCode, xCCD, yCCD, xFP, yFP
        else:
            return True





    def createStarCatalogFile(self, ra, dec, mag, starID, starCatalogFile):

        """Create a star catalogue file from equatorial coordinates.
        
        Create a star catalog ascii file given the equatorial coordinates 
        (RA and Dec) of the stars. This is simple copy numpy's option to
        save a ascii file, for the conveniece of the user.

        NOTE: this function sets the stellar catalogue to the simfile object.

        Paramters
        ---------
        ra : ndarray
            Array with right ascensions of the stars [deg]
        dec : ndarray
            Array with declination of the stars [deg]
        mag : ndarray
            Array with Johnson V magnitudes of the stars
        starID : ndarray
            Array with IDs of the star (integers)
        starCatalogPath : str
            Path of the star catalog file that will be written.

        Return
        ------
        A file will be saved, containing, ra, dec, and magnitude of the stars.
        The "ObservingParameters/StarCatalogFile" tag in the yaml tree will be
        changed to the given starCatalogPath
        """

        # Save the sky coordinates to the star catalog file

        np.savetxt(starCatalogFile,
                   np.transpose([ra, dec, mag, starID]),
                   fmt=['%11.6f', '%11.6f', '%8.4f', '%i'])

        # Set the "ObservingParameters/StarCatalogFile" tag in the yaml tree

        self["ObservingParameters/StarCatalogFile"] = starCatalogFile



    

    def createStarCatalogFileFromPixelCoordinates(self, rows, cols, magnitudes, starIDs, starCatalogPath):

        """Create a star catalogue file from the pixel coordinates.
        
        Create a star catalog ascii file given the pixel coordinates 
        (row and column) of the stars. This requires the orientation
        of the spacecraft, telescopes, focal plane, hence it's a member
        function of the Simulation class.
        
        Paramters
        ---------
        rows : ndarray
            Array with fractional row coordinates of the stars (CCD, not subfield) [pix]
        cols : ndarray
            Array with fractional column coordinates of the stars (CCD, not subfield) [pix]
        magnitudes : ndarray
            Array with Johnson V magnitudes of the stars
        starIDs : ndarray
            Array with IDs of the star (integers)
        starCatalogPath : str
            Path of the star catalog file that will be written.

        Return
        ------
        A file will be saved, containing, ra, dec, and magnitude of the stars.
        The "ObservingParameters/StarCatalogFile" tag in the yaml tree will be
        changed to the given starCatalogPath
        """

        # Extract the needed information from the yaml input file
        # Note: groupIDs and ccdIDs start counting from 1...

        if self["Telescope/GroupID"] == "Custom":
            azimuthAngle    = np.deg2rad(self["Telescope/AzimuthAngle"])
            tiltAngle       = np.deg2rad(self["Telescope/TiltAngle"])
        elif self["Telescope/GroupID"] == "Fast":
            azimuthAngle    = np.deg2rad(self["CameraGroups/AzimuthAngle"][4])
            tiltAngle       = np.deg2rad(self["CameraGroups/TiltAngle"][4])
        else:
            groupID = int(self["Telescope/GroupID"])
            azimuthAngle    = np.deg2rad(self["CameraGroups/AzimuthAngle"][groupID-1])
            tiltAngle       = np.deg2rad(self["CameraGroups/TiltAngle"][groupID-1])

        if self["CCD/Position"] == "Custom":
            ccdZeroPointX   = self["CCD/OriginOffsetX"]
            ccdZeroPointY   = self["CCD/OriginOffsetY"]
            CCDangle        = np.deg2rad(self["CCD/Orientation"])
        else:
            ccdID = int(self["CCD/Position"])
            ccdZeroPointX   = self["CCDPositions/OriginOffsetX"][ccdID-1]
            ccdZeroPointY   = self["CCDPositions/OriginOffsetY"][ccdID-1]
            CCDangle        = np.deg2rad(self["CCDPositions/Orientation"][ccdID-1])

        pixelSize       = self["CCD/PixelSize"]                                                               # [micron]

        if self["Platform/Orientation/Source"] == "Angles":
            raPlatform  = np.deg2rad(float(self["Platform/Orientation/Angles/RAPointing"]))
            decPlatform = np.deg2rad(float(self["Platform/Orientation/Angles/DecPointing"]))
            solarPanelOrientation = np.deg2rad(float(self["Platform/Orientation/Angles/SolarPanelOrientation"]))         # [rad]
        else:
            q_EQ2PLM = self["Platform/Orientation/Quaternion/Components"]
            raPlatform, decPlatform, solarPanelOrientation = rf.platformAnglesFromQuaternion(q_EQ2PLM)                   # [rad]

        focalPlaneAngle = np.deg2rad(self["Camera/FocalPlaneOrientation/ConstantValue"])                      # [rad]
        focalLength     = self["Camera/FocalLength/ConstantValue"] * 1000.0                                   # [m] -> [mm]

        if (self["PSF/Model"] == "MappedFromFile"):
            incldueFieldDistortion        = True
            mappedDistortion              = True
            inverseDistortionCoefficients = None
            pathToPsfFile                 = self["PSF/MappedFromFile/Filename"]
        else:
            includeFieldDistortion = self["Camera/IncludeFieldDistortion"]
            mappedDistortion       = False
            inverseDistortionCoefficients = self["Camera/FieldDistortion/ConstantInverseCoefficients"]
            pathToPsfFile          = None


        # Convert the pixel coordinates to focal plane coordinates [mm]

        xFPmm, yFPmm = rf.pixelToFocalPlaneCoordinates(cols, rows, pixelSize, ccdZeroPointX, ccdZeroPointY, CCDangle)

        # If distortion is required in the yaml input file, distort the focal plane coordinates [mm]
        if mappedDistortion:
            for i in range(len(xFPmm)):
                xFPmm[i], yFPmm[i] = rf.mappedDistortedToUndistortedFocalPlaneCoordinates(xFPmm[i], yFPmm[i], pathToPsfFile, focalLength)

        elif (includeFieldDistortion == "yes" or
              includeFieldDistortion == "1"   or
              includeFieldDistortion == True):
            xFPmm, yFPmm = rf.distortedToUndistortedFocalPlaneCoordinates(xFPmm, yFPmm, inverseDistortionCoefficients, focalLength)

        # Convert the focal plane coordinates to equatorial sky coordinates [rad]

        ra, dec = rf.focalPlaneToSkyCoordinates(xFPmm, yFPmm, raPlatform, decPlatform, solarPanelOrientation, tiltAngle, azimuthAngle, \
                                                focalPlaneAngle, focalLength)

        # Convert sky coordinates to degrees

        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)

        # Save the sky coordinates (in [deg]) to the star catalog file

        myFile = open(starCatalogPath, "w")
        myFile.write("# RA Dec Vmag starID\n")
        for n in range(len(ra)):
            myFile.write(f"{ra[n]:.6f} {dec[n]:.6f} {magnitudes[n]:.4f} {starIDs[n]}\n")
        myFile.close()

        # Set the "ObservingParameters/StarCatalogFile" tag in the yaml tree

        self["ObservingParameters/StarCatalogFile"] = starCatalogPath





    def createPhotometryFile(self, starIDs, photometryFile):

        """Create a photometry file list in ascii format and sets it to the YAML input.

        Parameters
        ----------
        starIDs : ndarray
            Array with IDs of the star (integers)
        fileName : str
            Path of the photometry file that will be written.

        Return
        ------
        A file will be saved, containing the star IDs that photometry should be performed on.
        The "Photometry/TargetFileName" tag in the yaml tree will be changed to the given 
        starCatalogPath
        """

        # Check if starIDs are just a number

        if type(starIDs) is int:
            starIDs = [starIDs]

        # Create photometry list file

        np.savetxt(photometryFile, np.transpose(starIDs), delimiter=" ", fmt="%d")

        # Set this to simulation and activate photometry

        self["Photometry/IncludePhotometry"] = True
        self["Photometry/TargetFileName"]    = photometryFile




        
    def createVariableSourceFile(self, time, dmag, variableSourceFile):

        """Create a variable source file of a target star.
        
        This function will automatically create the 'variableSourceList()' needed in order
        for PlatoSim to include the variability.
        The "ObservingParameters/StarCatalogFile" tag in the yaml tree will be
        changed to the given starCatalogPath

        Paramters
        ---------
        time : ndarray
            Array with right ascensions of the stars [deg]
        dmag : ndarray
            Array with declination of the stars [deg]
        variableSourceFile : str
            Path of the star catalog file that will be written.

        Return
        ------
        A file will be saved containing an ascii file with the columns
        time and delta magnitude.
        """

        # Save the sky coordinates to the star catalog file

        np.savetxt(variableSourceFile,
                   np.transpose([time, dmag]),
                   fmt=['%.1f', '%.6f'])       





    def createVariableSourceList(self, starID, variableSourceFile, variableSourceList):

        """Create a variable source file of a target star.
        
        NOTE: This function will automatically create the 'variableSourceList()'
        needed in order for PlatoSim to include the variability. Furthermore, the
        "Sky/IncludeVariableSources" tag in the YAML tree will be changed True.

        Paramters
        ---------
        time : ndarray
            Array with right ascensions of the stars [deg]
        dmag : ndarray
            Array with declination of the stars [deg]
        variableSourceFile : str
            Path of the star catalog file that will be written.

        Return
        ------
        A file will be saved containing an ascii file with the columns
        time and delta magnitude.
        """

        # Check if only a single target is requested
        
        if isinstance(variableSourceFile, str):
            variableSourceFile = [variableSourceFile]

        # Save the sky coordinates to the star catalog file
        
        with open(variableSourceList, 'w') as f:
            for i in range(len(starID)):
                f.write(f'{starID[i]} {variableSourceFile[i]}\n')

        # Set the "Sky" tag in the yaml tree

        self["Sky/IncludeVariableSources"] = True
        self["Sky/VariableSourceList"]     = variableSourceList





        
    def createDriftFile(self, quarter, fileName, model="poly", plot=False):

        """Create a camera drift file in ascii format and sets it to the YAML input.

        Parameters
        ----------
        quarter :
        fileName :
        model : 
        plot :

        Return
        ------
        A file will be saved, containing the star IDs that photometry should be performed on.
        The "Telescope/UseDriftFromFile" tag in the yaml tree will be changed to the given
        DriftFileName.
        """

        # Create TED file
        
        from platosim.noise import getTED  
        getTED(quarter=quarter, model=model, outfile=fileName, plot=plot)

        # Set this to simulation

        self["Telescope/UseDrift"]      = True
        self["Telescope/DriftSource"]   = 'FromFile'
        self["Telescope/DriftFileName"] = fileName





    def createDetectorTemperatureFile(self, time, temp, ccdFileName):

        """Create a CCD(T) file in ascii format and sets it to the YAML input.

        
        Parameters
        ----------
        time : ndarray
            Time points of time series [s]
        temp : ndarray
            CCD temeprature values [K]
        ccdFileName : str
            Filename of the CCD temeprature file

        Return
        ------
        A file will be saved, containing a column of time and CCD temperature.
        The "CCD/TemperatureFileName" tag in the yaml tree will be changed to the
        ccdFileName.

        NOTE: PlatoSim cannot model the FEE(T) yet.
        """

        # Create TED file
                
        np.savetxt(ccdFileName, np.transpose([time, temp]), fmt=['%.1f', '%.6f'])

        # Set this to simulation

        self["CCD/TemperatureFileName"] = ccdFileName
        self["CCD/Temperature"]         = "FromFile"




        
    def getReadoutTime(self):

        """Fetch the readout time.
        
        This function determine the duration of:
          - the readout that takes place before the next exposure starts,
          - and the readout that takes place during the next exposure,
            depending on the camera type (normal / fast) and the readout mode
            (nominal / partial readout).

        For the normal cameras the entire CCD is read out (with open shutter) after
        the exposure during a time interval called 'readoutTimeBeforeNextExposure'.
        Only after this readout, a new exposure is started.
        
        For the fast camera, half of the CCD is first quickly frame-transferred,
        after which it is read out slowly. In this case a new exposure is already
        started after the quick frame-transfer, and starts thus during the slow readout
        of the previous exposure. Hence the need for two parameters: 
        'readoutTimeBeforeNextExposure' and 'readoutTimeDuringNextExposure'.

        Return
        ------
        readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure
        """

        isFastCamera = self["Telescope/GroupID"] == "Fast"
        ccdPosition  = self["CCD/Position"]

        if ccdPosition == "Custom":
            numRows         = self["CCD/NumRows"]         # [pixels]
            numColumns      = self["CCD/NumColumns"]      # [pixels]
            firstRowExposed = self["CCD/FirstRowExposed"] # [pixels]
        else:
            # Positions start with 1, while the index starts at 0
            idx = ccdPosition - 1

            numRows    = self["CCDPositions/NumRows"][idx]     # [pixels]
            numColumns = self["CCDPositions/NumColumns"][idx]  # [pixels]

            if isFastCamera:
                firstRowExposed = self["CCDPositions/FirstRowForFastCamera"][idx]    # [pixels]
            else:
                firstRowExposed = self["CCDPositions/FirstRowForNormalCamera"][idx]  # [pixels]

        readoutMode = self["CCD/ReadoutMode/ReadoutMode"]

        if (readoutMode != "Nominal") and (readoutMode != "Partial"):
            raise ValueError("Simulation::getReadoutTime() Unknown readout mode " +
                             f"specification in configuration file: {readoutMode}")

        serialTransferTime       = self["CCD/SerialTransferTime"]   * 1e-9  # [ns -> s]
        parallelTransferTime     = self["CCD/ParallelTransferTime"] * 1e-6  # [micro s -> s]
        parallelTransferTimeFast = self["CCD/ParallelTransferTimeFast"] * 1e-6 # [micro s -> s]

        numColumnsBiasMap =  self["SubField/NumBiasPrescanColumns"]    # [pixels]
        numRowsSmearingMap = self["SubField/NumSmearingOverscanRows"]  # [pixels]
        # numBiasPrescanRows = self["SubField/NumBiasPrescanRows"]       # [pixels]
        numBiasPrescanColumns = self["SubField/NumBiasPrescanColumns"] # [pixels]

        # Both detector halves are read out simultaneously
        # -> columns read out by the FEE:
        # 		- half of the CCD
        # 		- serial pre-scan
        # 		- (serial over-scan)

        numColumnsReadout = numColumns / 2 + numColumnsBiasMap + numBiasPrescanColumns

        # How many rows will be actually read out by the FEE?
        # 	- nominal mode: image area + parallel over-scan
        #      normal camera: image area = whole CCD
        #      fast camera: image area = lower half of the CCD
        #	- partial readout: configurable
        # The rest of the image area will be dumped

        #--- Fast camera

        if isFastCamera:
            
            # Move the upper half of the CCD down to the lower half, row-by-row

            numRowsFrameTransfer          = numRows - firstRowExposed
            readoutTimeBeforeNextExposure = numRowsFrameTransfer * parallelTransferTimeFast

            # The actual readout of the lower half of the CCD (after frame transfer) is done
            # while the next exposure has already started

            # Nominal mode

            if readoutMode == "Nominal":
                numRowsReadout = firstRowExposed + numRowsSmearingMap
                numRowsDump = 0

            # Rows read out by the FEE: rows in the block (other rows in image area are dumped)
            # Note: no parallel over-scan

            elif readoutMode == "Partial":
                numRowsReadout = self["CCD/ReadoutMode/Partial/NumRowsReadout"]
                numRowsDump    = firstRowExposed - numRowsReadout

            readoutTimeDuringNextExposure = (numRowsDump * parallelTransferTimeFast +
                                             numRowsReadout * (parallelTransferTime +
                                                               numColumnsReadout *
                                                               serialTransferTime))

        #--- Normal camera

        else:
            # Nominal mode (full-frame readout)

            if readoutMode == "Nominal":
                
                # Rows read out by the FEE:
                #  - rows of image area
                #  - parallel over-scan

                numRowsReadout = numRows + numRowsSmearingMap

                # No rows dumped

                numRowsDump = 0

            # Partial readout

            elif readoutMode == "Partial":
                
                # Rows read out by the FEE: rows in the block (other rows in image area are dumped)
                # Note: no parallel over-scan

                numRowsReadout = self["CCD/ReadoutMode/Partial/NumRowsReadout"]
                numRowsDump = numRows - numRowsReadout


            readoutTimeBeforeNextExposure = (numRowsDump * parallelTransferTimeFast +
                                             numRowsReadout * (numColumnsReadout *
                                                               serialTransferTime +
                                                               parallelTransferTime))
            readoutTimeDuringNextExposure = 0

        return readoutTimeBeforeNextExposure, readoutTimeDuringNextExposure





    def getStarsWithinCameraGroup(self, ra, dec, alpha, delta, kappa,
                                  cameraGroup=1, quarter=1, radiusFOV=False,
                                  subfieldSize=6):

        """Fetch all stars within a camera group.

        This function determines if any star from a catalogue is within
        the FOV of a specific PLATO camera group.

        TODO: Could be implemented in a better and faster manner!

        Parameters
        ----------
        ra : list, ndarray
            Right ascension of stars to be checked against [deg]
        dec : list, ndarray
            Declination of stars to be checked against [deg]
        alpha : float
            Right acsension of pointing field [deg]
        delta : float
            Declination of pointing field [deg]
        kappa : float
            Rotation of the platform [deg]
        camGroup : int
            N-CAM camera group ID [1, 2, 3, 4]
        quarter : float
            Mission quarter number (starting from 1)
        subfieldSize : int
            Number of pixels to place retangular subfield

        Return
        ------
        dex : list
           Indices booleans with True being all stars within camera group FOV
        distanceOA : list
           Distance of each from the camera's optical axis [mm]
        """

        # Camera information

        focalLength     = float(self["Camera/FocalLength/ConstantValue"]) * 1000.0  # [mm]
        focalPlaneAngle = np.deg2rad(float(self["Camera/FocalPlaneOrientation/ConstantValue"]))

        # Platform orientation [deg]

        alphaPlatformDeg = self["Platform/Orientation/Angles/RAPointing"]  = alpha
        deltaPlatformDeg = self["Platform/Orientation/Angles/DecPointing"] = delta
        kappaPlatformDeg = self["Platform/Orientation/Angles/SolarPanelOrientation"] = ut.getSolarPanelOrientation(kappa, quarter)

        # Convert coordinates to radians
        
        raTargetsRad  = np.deg2rad(ra)
        decTargetsRad = np.deg2rad(dec)
        alphaPlatformRad = np.deg2rad(alphaPlatformDeg)
        deltaPlatformRad = np.deg2rad(deltaPlatformDeg)
        kappaPlatformRad = np.deg2rad(float(kappaPlatformDeg))
        
        # Loop over each star for this cam-group

        self["Telescope/GroupID"] = cameraGroup
        tiltGroup = np.deg2rad(self["CameraGroups/TiltAngle"][cameraGroup-1])
        azimGroup = np.deg2rad(self["CameraGroups/AzimuthAngle"][cameraGroup-1])

        # Select how large the FOV should be [deg -> rad]

        if not radiusFOV:
            radiusFOV = self['CCD/RelativeTransmissivity/RadiusFOV']
        radiusFOV = np.deg2rad(radiusFOV)
            
        # Start loop over each star
        # Check 1: if star in located on one of the 4 CCDs
        # Check 2: if star is within the "radius" from the OA
        
        dexGroup   = np.zeros(len(ra), dtype=bool)
        distanceOA = np.zeros(len(ra))

        for i in range(len(ra)):
            subfieldIsOnCCD = self.setSubfieldAroundSkyCoordinates(raTargetsRad[i],
                                                                   decTargetsRad[i],
                                                                   subfieldSize,
                                                                   subfieldSize)
            if subfieldIsOnCCD:
                xFP, yFP = rf.skyToFocalPlaneCoordinates(raTargetsRad[i],
                                                         decTargetsRad[i],
                                                         alphaPlatformRad,
                                                         deltaPlatformRad,
                                                         kappaPlatformRad,
                                                         tiltGroup, azimGroup,
                                                         focalPlaneAngle, focalLength)
                
                distanceOA[i] = rf.gnomonicRadialDistanceFromOpticalAxis(xFP, yFP,
                                                                         focalLength)
                if distanceOA[i] < radiusFOV:
                    dexGroup[i] = True
                else:
                    dexGroup[i] = False

        # Return parameters
        
        return dexGroup, distanceOA





    def getSubPixelPositions(self, raPF, decPF, ra, dec, camGroup=1, quarter=1):

        """Fetch the subpixel postion for all stars.

        This function determines the subpixel postion of all stars for the first
        exposure in the simulation. The function also returns the CCD ID of each
        star in the focal plane.

        Parameters
        ----------
        raPF : float
            Right acsension of pointing field [deg]
        decPF : float
            Declination of pointing field [deg]
        ra : list, array
            Right ascension of stars to be checked against [deg]
        dec : list, tuple, array
2            Declination of stars to be checked against [deg]

        Return
        ------
        ccdCode : int
           CCD ID of belonging to the star
        xCCD : float32
           Subpixel position of star in x (row)
        yCCD : float32
           Subpixel position of star in y (column)

        Warning:
        --------
        This functions computes its own version of the solarPanelOrientation, but it does
        NOT change the corresponding value in the yaml config.
        """

        # Telescope config

        if self["Platform/Orientation/Source"] == "Angles":
            raPlatform  = np.deg2rad(float(self["Platform/Orientation/Angles/RAPointing"]))
            decPlatform = np.deg2rad(float(self["Platform/Orientation/Angles/DecPointing"]))
            solarPanelOrientation = np.deg2rad(float(self["Platform/Orientation/Angles/SolarPanelOrientation"]))         # [rad]
        else:
            q_EQ2PLM = self["Platform/Orientation/Quaternion/Components"]
            raPlatform, decPlatform, solarPanelOrientation = rf.platformAnglesFromQuaternion(q_EQ2PLM)                   # [rad]


        focalLength      = float(self["Camera/FocalLength/ConstantValue"]) * 1000.0  # [m] -> [mm]
        focalPlaneAngle  = np.deg2rad(float(self["Camera/FocalPlaneOrientation/ConstantValue"]))

        # NOTE: with the following line the value for the solarPanelOrientation is no longer consistent
        #       with the value in the yaml file.

        solarPanelOrientation = np.deg2rad(math.fmod(quarter * 90., 360.) - 6)

        raTargets  = np.deg2rad(ra)                               # [rad]
        decTargets = np.deg2rad(dec)                              # [rad]

        # Loop over each star for this cam-group

        self["Telescope/GroupID"] = camGroup
        azimuthTelescope = np.deg2rad(self["CameraGroups/AzimuthAngle"][camGroup-1])
        tiltTelescope    = np.deg2rad(self["CameraGroups/TiltAngle"][camGroup-1])

        # CCD properties

        pixelSize = float(self["CCD/PixelSize"])

        ccdCode = np.zeros(len(ra))
        xCCD    = np.zeros(len(ra))
        yCCD    = np.zeros(len(ra))

        for i in range(len(ra)):

            subfieldIsOnCCD = self.setSubfieldAroundSkyCoordinates(raTargets[i], decTargets[i], 6, 6)
            if subfieldIsOnCCD:

                # Fetch CCD code and pixel coordinates (account for field distortion in included)

                if self["Camera/IncludeFieldDistortion"]:
                    includeFieldDistortion = self["Camera/IncludeFieldDistortion"]
                    if self["Camera/FieldDistortion/Type"] == 'FromFile':
                        mappedDistortion = True
                        distortionCoefficients = self["Camera/FieldDistortion/CoefficientsFromFile"]
                    else:
                        mappedDistortion = False
                        distortionCoefficients = self["Camera/FieldDistortion/ConstantCoefficients"]
                else:
                    includeFieldDistortion = False
                    mappedDistortion = False
                    distortionCoefficients = False

                out = rf.getCCDandPixelCoordinates(raTargets[i],
                                                   decTargets[i],
                                                   raPlatform,
                                                   decPlatform,
                                                   solarPanelOrientation,
                                                   tiltTelescope,
                                                   azimuthTelescope,
                                                   focalPlaneAngle,
                                                   focalLength,
                                                   pixelSize,
                                                   includeFieldDistortion,
                                                   normal=True,
                                                   mappedDistortion=mappedDistortion,
                                                   distortionCoefficients=distortionCoefficients)
                ccdCode[i], xCCD[i], yCCD[i] = out[0], out[1], out[2]
                
        # That's it!
        
        return ccdCode, xCCD, yCCD


#!/usr/bin/env python3
"""
This script uses the PIC targets and their contaminants to simulate realistic imagettes
or lightcurves using PlatoSim. It can simulate any of the 24 normal cameras/telescopes,
for an arbitary number of mission quarters. Since a nescessary rotation (along the roll
axis) of the spacecraft platform is required in order to repoint the solar panels every
90 days, simulations cannot realistically extend beyond a quarter of a year. Given an
PIC input sample, the script will automatically simulation all targets being visible by
any camera falling on one of the 4 CCDs. The fact that the PLATO spacecraft will be
equipped with 4 camera groups consisting of 6 cameras each, constrains the efficient use
of nodes and cores on the HPC. In order to make the simulation as realistic as possible,
random seats of various intrinsic- and instrumental effects are included, meaning that
each camera within each camera group needs to be simulated independently (which indeed
are the raw imaging output of PLATO).
"""

# Built-in
import os
import sys
import glob
import shutil
import argparse
import datetime
import warnings
import tracemalloc
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PlatoSim functions
import platosim.utilities       as ut
import platosim.referenceFrames as rf
import platosim.statistics      as st
from platosim.simulation   import Simulation
from platosim.simfile      import SimFile
from platosim.utilities    import errorcode, getPointingField
from platosim.plot         import drawStarInCCDfocalPlane, plotSubfieldAnimation
from platosim.matplotlibrc import setup; setup()

# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace
# pylint: disable=multiple-statements
# pylint: disable=no-member
# pylint: disable=bare-except
# pylint: disable=redefined-outer-name
# pylint: disable=no-name-in-module
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=consider-using-enumerate
# pylint: disable=attribute-defined-outside-init

#==============================================================#
#                        PLATOnium CLASS                       #
#==============================================================#

class PLATOnium(object):
    """Class for running multi-camera and multi-quarter PlatoSim simulations.
    """
    def __init__(self, args):
        
        # PARSED ARGUMENTS

        self.targetNo = args.starID
        self.group    = args.groupID
        self.camera   = args.cameraID
        self.quarter  = args.quarter

        self.seed        = args.seed
        self.performance = args.performance

        self.inputFile     = args.ifil
        self.outputDir     = args.odir
        self.storageDir    = args.sdir
        self.project       = args.project
        self.inputFileName = args.yaml
        self.simPrefix     = args.prefix
        self.sample        = args.sample
        self.starcatFile   = args.starcat
        self.varSourceFile = args.varfile
        self.varSourceList = args.varlist
        self.compress      = args.compress
        self.savePlot      = args.saveplot

        self.field         = args.field
        self.cadence       = args.cadence
        self.simTime       = args.tdur
        self.simBeginTime  = args.bdur
        self.simExposures  = args.nexp
        self.simBeginExp   = args.bexp
        self.picID         = args.pic
        self.mag           = args.mag
        self.conNone       = args.con_none
        self.conDeltaMag   = args.con_dmag
        self.conDisLimit   = args.con_dist
        self.reuseJitter   = args.jit_reuse
        self.fullFrame     = args.fullframe
        self.noAberrCorr   = args.no_aberr_corr

        self.maskUpdate = args.mask
        self.detrend    = args.detrend
        self.poly_deg   = args.poly_deg
        self.clipWotan  = args.clip
        self.clipSigma  = args.clip_sigma
        self.clipWindow = args.clip_window
        self.stitch     = args.stitch
        self.plotPost   = args.check

        self.pipeline         = args.pipeline
        self.pipePsfMethod    = args.pipe_psf
        self.pipeCadence      = args.pipe_cadence
        self.pipeFluxError    = args.pipe_flux_err
        self.pipeAbsCenError  = args.pipe_cen_err
        self.pipePrnuError    = args.pipe_prnu_err
        self.pipeJitDriftOff  = args.pipe_jit_off
        self.pipeExtendedMask = args.pipe_emask
        self.pipePlots        = args.pipe_plots
        
        # Debug L1 without re-simulating all platosim data
        self.l1_only = False

        # Overwrite simulation
        if args.overwrite:
            self.overwrite = True
        else:
            self.overwrite = False

        # MANDATORY PARAMETERS
        
        # Normal vs Fast cameras
        if self.group in [1, 2, 3, 4]:
            if self.camera in [1, 2, 3, 4, 5, 6]:
                self.groupID = self.group
            else:
                errorcode('error', 'Camera can only be [1, 2, 3, 4, 5, 6]')
        elif self.group == 5:
            self.groupID = 'Fast'
            if self.camera == 1:
                self.cameraID = 'blue'
            elif self.camera == 2:
                self.cameraID = 'red'
            else:
                errorcode('error', 'Fast camera can only be [1, 2] = [blue, red]')
        else:
            errorcode('error', 'Camera-group can only be [1, 2, 3, 4, 5] (Fast = 5)')

        # Select full-frame CCD
        if self.fullFrame:
            self.ccdCode = self.targetNo
            if not self.ccdCode in [1, 2, 3, 4]:
                errorcode('error', 'CCD code can only be [1, 2, 3, 4]')

        # Select the camera index [0, 25]
        self.cameraIndex = (self.group - 1) * 6 + (self.camera - 1)

        # OPTIONAL PARAMETERS

        # Verbosity (a.k.a log level) -> Identical to PlatoSim usage:
        # verbose = 0: Cluster mode: Disabling print and warnings, and no log files are saved
        # verbose = 1: Default mode: Print details to bash but do not save log files
        # verbose = 3: Debug   mode: Print details to bash and saves all log files
        if args.verbose == 0:
            self.verbose = 0
            self.verbose_platosim = 0
            # Bash extention to write no output for the pipeline
            self.devnull = '> /dev/null'
            warnings.filterwarnings("ignore")
        elif args.verbose in [None, 1, 2]:
            self.verbose = 2
            self.verbose_platosim = 0
            self.devnull = ''
            warnings.filterwarnings("ignore")
        else:
            self.verbose = 3
            self.verbose_platosim = 3
            self.devnull = ''

        # Save animation
        if args.animation:
            self.animation = True
        else:
            self.animation = False

        # Start software writing
        if self.verbose > 1:
            errorcode('software', '\nPLATOnium')
            
        # I/O PARAMETERS
        
        # Absolute pwd path
        self.path = Path(__file__).parent.resolve()

        # PlatoSim inputfiles
        self.platosimInputDir = self.path.joinpath(os.getenv('PLATO_PROJECT_HOME'), 'inputfiles')
        
        # Sort the workdir input paths and files
        if self.project and self.inputFileName:
            self.simDir    = self.path.joinpath(os.getenv('PLATO_WORKDIR'), self.project)
            self.inputDir  = self.simDir / 'input'
            self.inputFile = self.inputDir / self.inputFileName
        elif self.project:
            self.simDir    = self.path.joinpath(os.getenv('PLATO_WORKDIR'), self.project)
            self.inputDir  = self.simDir / 'input'
            self.inputFile = self.inputDir / 'inputfile.yaml'
        elif self.inputFile:
            self.inputFile = Path(self.inputFile).resolve()
            self.inputDir  = self.inputFile.parents[0]
            self.simDir    = self.inputFile.parents[1]
        else:
            errorcode('error', 'Use {-i, --project, --yaml} to locate the inputfile!')

        # Check if the inputfile.yaml exists
        if not self.inputFile.is_file() and not self.inputFile.is_symlink():
            errorcode('error', 'File inputfile.yaml do not exist! ' +
                      'Alternamtively use {-i, --yaml}')

        # Overwrite simulation
        if args.overwrite:
            self.overwrite = True
        else:
            self.overwrite = False

        # SOURCE CATALOGUE PARAMETERS
        
        # Inclusion thresholds for contaminants
        if not self.conDeltaMag: self.conDeltaMag = 10   # [delta mag]
        if not self.conDisLimit: self.conDisLimit = 60   # [arcsec -> 15 arcsec/pixel]
        
        # PHOTOMETRY AND PIPELINE PARAMETERS

        # Debug L1 without re-simulating all platosim data
        self.l1_only = False
        
        if self.pipeline:
            # Pipeline paths
            self.platoBin = self.path.joinpath(os.getenv('PLATO'), 'bin')
            self.platoLib = Path(os.path.split(sys.executable)[0]).resolve()

            # Check that the sample is parsed
            if self.sample is None:
                errorcode('error', 'The pipeline mode need parsing of --sample argument!')

            # Default L1 pipeline parameters
            self.bsres = 10                  # [subpixel]
            if not self.pipePrnuError:
                self.pipePrnuError = 0.1     # [%]
            if not self.pipeFluxError:
                self.pipeFluxError = 1       # [%]
            if not self.pipeAbsCenError:
                self.pipeAbsCenError = 0.03  # [pixel]
            if self.sample == "P1" and not self.pipeCadence:
                self.pipeCadence = 25
            if self.sample == "P5" and self.pipeCadence not in [25, 50, 600]:
                errorcode('error', 'Must set --pipe_cadence = 25 | 50 | 600')

            # Pipeline PSF method
            if self.pipePsfMethod not in ['microscan', 'library']:
                errorcode('error', 'Must set --pipe_psf = microscan | library. \nNote: P1 ' +
                          'typically uses \'microscan\' and P5 uses \'library\' interpolation')

            # Download the PSF library if it is needed and doesn't exist
            if self.pipePsfMethod == "library":
                self.psfLibraryFilename = 'INVERTED_PSF_LIBRARY_q13_241129.hdf5'
                if not os.path.exists(f"{self.inputDir}/{self.psfLibraryFilename}"):
                    print(f"Downloading PSF library {self.psfLibraryFilename} from KUL FTP")
                    ut.downloadFromFTP(filename=self.psfLibraryFilename,
                                       outputDir=self.inputDir,
                                       server='plato')

        # Check parsing of detrending model
        if not self.detrend in [None, 'poly', 'lowess', 'poly_lowess', 'lowess_theil', 'wotan']:
            errorcode('error', 'Not a valid detrending model!')

        # Check parsing of detrending model
        self.stitchActive = False
        if not self.stitch in [None, 'median', 'lowess']:
            errorcode('error', 'Not a valid stitching model!')

        # Built-in photometric post-processing
        if self.detrend is not None or self.clipWotan:
            self.postProcess = True
        else:
            self.postProcess = False

        # Check if polynomial degree is requested
        if not isinstance(self.poly_deg, int):
            self.poly_deg = False

        # Check clip wotan window
        if self.clipWindow is None:
            self.clipWindow = 0.5
            
        # Monitor script speed
        self.tic  = datetime.datetime.now()
        self.tic0 = datetime.datetime.now()


    def configure_output(self):
        """Module to create, configure, and select the correct output folders.
        """
        # Final destination on the cluster
        if self.storageDir:
            self.storageDir = Path(self.storageDir).resolve()
            # Create directory
            if not self.storageDir.is_dir():
                self.storageDir.mkdir(parents=True, exist_ok=True)
                os.system(f'chmod 755 {self.storageDir}')

        # Default output path
        if self.outputDir is None:
            self.outputDir = self.simDir.joinpath('output')
        else:
            self.outputDir = Path(self.outputDir).resolve()

        # Custom prefix
        if self.simPrefix is not None:
            self.simPrefix = f'{self.simPrefix}_'
        else:
            self.simPrefix = ''

        # Set general output filename
        self.starID = f'{self.targetNo}'.zfill(9)

        # Select suffix of observation
        if self.groupID == 'Fast':
            self.obsPrefix = f'Fcam{self.camera}_Q{self.quarter}'
        else:
            self.obsPrefix = f'Ncam{self.group}.{self.camera}_Q{self.quarter}'

        # Combine file name depending on simulation mode
        # Full frame mode
        if self.fullFrame:
            self.outputFileName = f'{self.obsPrefix}_ccd{self.ccdCode}'

        # Standard mode
        elif not self.pipeline:
            self.outputDir      = self.outputDir / self.starID
            self.outputFileName = f'{self.simPrefix}{self.starID}_{self.obsPrefix}'

        # Pipline mode
        else:
            self.outputFileName = self.starID
            relativeDirStar = f'{self.sample}/Q{self.quarter}/Ncam{self.group}{self.camera}'

            # L1 pipeline paths
            self.pipelineDir  = self.outputDir / relativeDirStar
            self.outputDirStarIDsim = self.pipelineDir / self.starID
            self.outputDirStarIDnew = self.outputDir / 'reduced' / self.sample / self.starID

            # Microscan paths
            self.microscanDir = self.outputDir / 'microscan' / relativeDirStar / self.starID
            self.microscanDirStarID = self.microscanDir / self.starID
            self.microscanDirInvers = self.microscanDir / 'inversion'

            # Create paths
            self.outputDirStarIDsim.mkdir(parents=True, exist_ok=True)
            self.outputDirStarIDnew.mkdir(parents=True, exist_ok=True)
            self.microscanDirStarID.mkdir(parents=True, exist_ok=True)
            self.microscanDirInvers.mkdir(parents=True, exist_ok=True)

            # Normal output path
            self.outputDir = Path(self.outputDirStarIDsim).resolve()

            # Path + Prefix of simulations
            self.microscanSimName = self.microscanDirStarID / self.starID

        # Store final file name
        self.outputSimName = self.outputDir / self.outputFileName

        # Create directory
        if not self.outputDir.is_dir():
            self.outputDir.mkdir(parents=True, exist_ok=True)
            os.system(f'chmod 755 {self.outputDir}')

        # Check if output file exists
        if (not self.overwrite and Path(f'{self.outputSimName}.hdf5').is_file() and
            not self.pipeline):
            errorcode('error', 'HDF5 file already exists and no pipeline run triggered.' +
                      ' Use "-w" to overwrite it')
        elif (not self.overwrite and Path(f'{self.outputSimName}.hdf5').is_file() and
              self.pipeline):
            errorcode('warning', 'HDF5 file already exists and pipeline triggered.' +
                      ' Analysing existing HDF5 file.')
            self.l1_only = True
        else:
            pass


    def _check_source_name(self, df):
        """Utility to check source name in catalogue.
        """
        if 'PIC' in df:
            self.colID = 'PIC'
        elif 'gaiaDR3' in df:
            self.colID = 'gaiaDR3'
        elif 'source_gaia_dr3' in df:
            self.colID = 'source_gaia_dr3'                
        elif 'ID' in df:
            # TODO Backward compatible with PlatoSim =< 3.5.3-19-g18d87597
            self.colID = 'ID'
        else:
            errorcode('error', 'Cannot find ID identifier! ' + 
                      'Valid names are [ID, PIC, gaiaDR3, source_gaia_dr3]')


    def _check_passband_name(self, df):
        """Utility to check passband name in catalogue.
        """
        # Plato passbands
        if ('PBmag' in df) and (self.group == 5) and (self.camera == 1):
            self.passband = 'PBmag'
        elif ('PRmag' in df) and (self.group == 5) and (self.camera == 2):
            self.passband = 'PRmag'
        elif 'Pmag' in df:
            self.passband = 'Pmag'
        # Gaia mean G passband
        elif 'Gmag' in df:
            self.passband = 'Gmag'
        # Johnson-Cousin V passband
        # NOTE to be coherent with Fluxm0 in YAML!
        elif 'mag' in df:
            self.passband = 'mag'            
        else:
            errorcode('error', "No valid passband present in star catalogue! " +
                      "Use ['mag', 'Gmag', 'Pmag', 'PBmag', 'PRmag']")
                
            
    def load_stars(self):
        """Module to load the stellar targets and contaminants.
        """
        # Fetch some information from YAML
        sim = Simulation(self.outputFileName, self.inputFile)

        # Set pointing field from YAML
        if self.field is None:
            self.pointingField = sim['ObservingParameters/StarCatalogFile']        
        else:
            self.pointingField = self.field
            
        # FULL-FRAME CATALOGUE
        
        if self.fullFrame:
            
            # Get pointing field from YAML and load stellar catalogue            
            starcatName = f'starcat**_{self.pointingField}_group{self.group}.ftr'
            try:
                starcat = Path(glob.glob(f'{str(self.inputDir)}/{starcatName}')[0])
            except IndexError:
                errorcode('error', 'No source catalogue found for full-frame mode! '+
                          'Add one or use "picsim --vizier" to produce one')

            # Check for stellar catalogue
            if not starcat.is_file() and not starcat.is_symlink():
                errorcode('error', 'No star catalogue found in the project input directory!')

            # Load source catalogue
            if self.verbose > 1:
                print(f'\nLoading source catalogue: {starcat.name}')
            self.dx = pd.read_feather(starcat)

            # Check source and passband names
            self._check_source_name(self.dx)
            self._check_passband_name(self.dx)

            # Store source catalogue
            self.ds = pd.DataFrame()
            self.ds['ra']  = self.dx.ra
            self.ds['dec'] = self.dx.dec
            self.ds['mag'] = self.dx[self.passband]
            self.ds['ids'] = np.arange(0, len(self.ds.ra)).astype(int)

        # SUBFIELD CATALOGUES
        
        else:
            # Fetch stars from custum catalogue
            if self.starcatFile is not None:

                # Read catalogue
                # TODO Update colID to also incldue source_*
                df = pd.read_feather(self.starcatFile)

                # Define data frames
                self.targetDex = 0
                self.df = df.loc[self.targetDex]
                self.dc = df.iloc[1:]

            # Fetch stars from PIC and Gaia DR3 catalogues
            else:
                
                # Addsample name to use non-default catalogues
                if self.sample is not None:
                    extra_str = self.sample
                else:
                    extra_str = ''

                # Fetch targets and contaminant files
                try:
                    if self.field is None:
                        catTarName = f'{self.inputDir}/starcat**{extra_str}**targets.ftr'
                        catConName = f'{self.inputDir}/starcat**{extra_str}**contaminants.ftr'
                    else:
                        catTarName = f'{self.inputDir}/starcat**{self.pointingField}**{extra_str}**targets.ftr'
                        catConName = f'{self.inputDir}/starcat**{self.pointingField}**{extra_str}**contaminants.ftr'
                except IndexError:
                    errorcode('error', f'Stellar catalogue "{self.sample}" does not exist!')
                else:
                    try:
                        self.catTarFile = glob.glob(catTarName)[0]
                        self.catConFile = glob.glob(catConName)[0]
                    except IndexError:
                        errorcode('error', 'Stellar contaminant catalogue does not exist! ' +
                                  'To generate full-frame CCD images, use "--fullframe"')
                    else:
                        df = pd.read_feather(self.catTarFile)
                        dc = pd.read_feather(self.catConFile)

                # Check if target and contaminant catalogues are consistent
                # NOTE This allows the user to have multiple catalogues in same project folder
                if str(Path(self.catTarFile).stem[:-8]) != str(Path(self.catConFile).stem[:-13]):
                    errorcode('error', f'Target catalogue does not match contaminant catalogues! ' +
                              'Use --field to specify a LOP or alter the YAML')

                # Correct indicing and allow a specific star to be choosen
                if self.targetNo == 0:
                    errorcode('error', 'Star ID indicing starts from 1 and not 0!')
                elif self.picID is not None:
                    if 'gaiaDR3' in df or 'source_gaia_dr3' in df:
                        errorcode('error', "Argument '--pic' is only valid for a PIC identifier!")
                    try:
                        self.targetDex = np.where(df.PIC == self.picID)[0][0]
                    except IndexError:
                        errorcode('error', f'PIC {self.picID} star does not exist in catalogue:' +
                                  f'\n{self.catTarFile}')
                else:
                    # Set index of target
                    self.targetDex = self.targetNo - 1
                            
                # Select target star
                self.df = df.iloc[self.targetDex]

            # Continue information for subfield simulations

            # Check source name and passband
            self._check_source_name(self.df)
            self._check_passband_name(self.df)

            # If requested select only the target, else include contaminants
            if not self.starcatFile:
                if self.conNone:
                    self.dc = dc[dc[self.colID] == 0]
                else:
                    self.dc = dc[dc[self.colID] == self.df[self.colID]]
                    self.dc = self.dc.sort_values(by=['dis'])

            # Change naming
            self.df = self.df.to_frame().T.rename(columns={self.passband:'mag'}).squeeze()
            self.dc = self.dc.rename(columns={self.passband:'mag'})

            # If requested overwrite magnitude of target star
            if self.mag is not None:
                self.df.mag = self.mag

            # Number of contaminants
            if self.conNone:
                self.numCon = 0
            else:
                # Limits for contaminants
                self.dc = self.dc[(self.dc.mag - self.df.mag) < self.conDeltaMag]
                self.dc = self.dc[self.dc.dis < self.conDisLimit]
                self.dc = self.dc.reset_index(drop=True)
                self.numCon = self.dc.shape[0]

            # Save star catalogue
            self.ds = pd.DataFrame()
            self.ds['ra']  = np.append(self.df['ra'],  self.dc['ra'])
            self.ds['dec'] = np.append(self.df['dec'], self.dc['dec'])
            self.ds['mag'] = np.append(self.df['mag'], self.dc['mag'])
            if not self.conNone:
                self.ds['ids'] = np.arange(1, self.numCon+2)
            else:
                self.ds['ids'] = 1

        
    def init_sim(self):
        """Module to initialize the the PlatoSim simulation object.
        """
        # INITIALIZE SIMULATION
        
        # Setting up a test simulation environement
        if self.verbose > 1:
            errorcode('module', '\nInitialize and configure PlatoSim\n')
        sim = Simulation(self.outputFileName, self.inputFile)

        # Start time of simulation [s]
        self.timeStart = round(ut.quarter() * (self.quarter - 1) * 86400)

        # Set general parameters for N-CAM and F-CAM
        # NOTE These functions set multiple default configurations (see simulation.py)
        # NOTE CCD offset is automatically set by sim.setSubfieldAroundCoordinates()
        # NOTE Parameter "normal" is used in the subfield selection
        if self.groupID == 'Fast':
            self.normal = False
            sim.useFastCamera(self.cameraID, self.performance, self.timeStart)
        else:
            self.normal = True
            sim.useNormalCamera(self.performance, self.timeStart)
        
        # CONFIGURE OBSERVING PARAMETERS

        # Cadence of time series [s]
        # NOTE Overwrites sim.useFastCamera() and sim.useNormalCamera()
        cadence = sim['ObservingParameters/CycleTime']
        if self.cadence:
            sim['ObservingParameters/CycleTime'] = self.cadence
            factor = self.cadence / cadence
            if factor > 1 and self.verbose > 1:
                errorcode('warning', f'Only alter the cadence for testing purposes!')
        else:
            self.cadence = cadence
            
        # Check if start time or exposure is parsed
        if self.simBeginTime:
            self.simBeginExp = round(self.simBeginTime * 86400 / self.cadence)
        elif self.simBeginExp is None:
            self.simBeginExp = 0
            
        # Start time of simulation relative mission BOL
        self.beginExposureNr = round(self.timeStart / self.cadence) + self.simBeginExp
        sim['ObservingParameters/BeginExposureNr'] = self.beginExposureNr

        # Duration of time series [exp]
        if self.simExposures:
            # Setting timeseries by N exposures given by user
            self.numExposures = self.simExposures
        elif self.simTime is not None:
            # Setting timeseries by time given by user
            self.numExposures = round(self.simTime * 86400 / self.cadence)
        else:
            # Setting time series to full quarter (assuming a 1-day break)
            self.numExposures = round((ut.quarter() - 1) * 86400 / self.cadence)

        # Secure that F-CAM simulations do not crash due to HDF5 memory overload!
        if not self.normal and self.numExposures > 1071360:
            if (sim['ControlHDF5Content/WritePixelMaps'] or
                sim['ControlHDF5Content/WriteBiasMaps'] or
                sim['ControlHDF5Content/WriteSmearingMaps']):
                errorcode('error', 'A maximum of 1,071,360 F-CAM imagettes can be simulated! ' +
                          'Use --nexp (--tdur) and --bxep (--bdur) to split quarter ' +
                          'into 3 smaller data chunks (maximum of 31 days each). E.g.:\n' +
                          '--nexp 1071360\n' +
                          '--nexp 1071360 --bexp 1071360\n' +
                          '--nexp 978413  --bexp 2142720\n')
            
        # Secure correct zero-point flux for passband
        # Values are from: docs/technical/DLR-TN-0113/zero_point.ipynb
        # NOTE if "mag" column exist the YAML entry "Fluxm0" is used
        if self.passband == 'Pmag':
            sim['ObservingParameters/Fluxm0'] = 6.80208291e7
        elif self.passband == 'PBmag':
            sim['ObservingParameters/Fluxm0'] = 1.07575201e8 
        elif self.passband == 'PRmag':
            sim['ObservingParameters/Fluxm0'] = 5.34942672e7
            
        # PHOTOMETRY ALA MARCHIORI
        
        # The mask-update interval [days] and rate [event after N exposures]
        if self.maskUpdate:
            sim['Photometry/MaskUpdateInterval'] = self.maskUpdate
            self.maskUpdateRate = round(self.maskUpdate * 86400 / self.cadence)
        else:
            self.maskUpdateRate = round(sim['Photometry/MaskUpdateInterval']
                                        * 86400/self.cadence)

        # CONFIGURE PLATFORM

        # Set the pointing of the platform using either ICRS coordinates or Quaternion
        inputFilePLM = self.inputDir.joinpath('instrumentPLM.csv')
        if inputFilePLM.is_file():
            # Use Quaternion to set platform pointing
            # NOTE Included if "instrumentPLM.csv" is available in the input folder
            # NOTE This option allows directly the usage of pointing errors (PRE)
            PLM = pd.read_csv(inputFilePLM)
            if PLM.shape[0] < self.quarter: 
                errorcode('error', 'Cannot apply platform quaternion! ' +
                          f'instrumentPLM.csv does not have {self.quarter} quarter rows..')
            elif PLM.shape[1] != 4:
                errorcode('error', 'Cannot apply platform quaternion! ' +
                          'instrumentPLM.csv needs 4 unit quaternions [q0, qx, qy, qz]..')
            else:
                quaternionPLM = PLM.iloc[self.quarter-1].tolist()
                sim["Platform/Orientation/Source"] = 'Quaternion'
                sim["Platform/Orientation/Quaternion/Components"] = quaternionPLM
                # Print to bash
                if self.verbose > 1:
                    print('Setting platform pointing    (PLM FromFile: Quaternion)')
        else:
            # Select PLATO pointing field from inputfile
            alpha, delta, kappa = getPointingField(self.pointingField)
            sim["Platform/Orientation/Angles/RAPointing"]  = alpha
            sim["Platform/Orientation/Angles/DecPointing"] = delta
            # Solar panel orientation [deg]: Q(N) = {0, 90, 180, 270} + kappa
            solarPanelOrientationDeg = ut.getSolarPanelOrientation(kappa, self.quarter)
            sim["Platform/Orientation/Angles/SolarPanelOrientation"] = solarPanelOrientationDeg
            # Print to bash
            if self.verbose > 1:
                print(f'Setting platform pointing    (PLM FromYAML: {self.pointingField})')
 
        # POINTING ERRORS
        
        # Include Pointing Repeatability Error (PRE) between consecutive quarters
        # NOTE Below we secure that platonium is backward compatible
        inputFilePRE_csv = self.inputDir.joinpath('instrumentPRE.csv')
        inputFilePRE_txt = self.inputDir.joinpath('instrumentPRE.txt')
        block = 'Platform/Orientation/Angles'
        #---------------------------------------------------------------------            
        # New implementation for PlatoSim >= 3.7.0
        if inputFilePRE_csv.is_file():
            PRE = pd.read_csv(inputFilePRE_csv)
            if self.quarter not in PRE.quarter.tolist():
                errorcode('error', 'Cannot apply pointing error model! ' +
                          f'instrumentPRE.csv does not contain quarter {self.quarter}..')
            # Apply PRE from new or old method
            data = PRE.iloc[self.quarter - 1]
            sim[f"{block}/RAPointing"]            += data.alpha
            sim[f"{block}/DecPointing"]           += data.delta
            sim[f"{block}/SolarPanelOrientation"] += data.kappa
            # Print to bash
            if self.verbose > 1:
                print('Applying pointing errors     (PRE FromFile)')
        #---------------------------------------------------------------------
        # Old implementation for PlatoSim < 3.7.0
        elif inputFilePRE_txt.is_file():
            PRE = np.loadtxt(inputFilePRE_txt)
            # Catch one dimentional arrays
            try:
                PRE.shape[1]
            except:
                PRE = np.array([PRE])
            # Apply pointing errors
            index = np.where(PRE[:,0] == self.quarter)[0]
            try:
                index[0]
            except:
                errorcode('error', 'Cannot apply pointing error model! ' +
                          f'instrumentPRE.txt does not contain quarter {self.quarter}..')
            else:
                sim[f"{block}/RAPointing"]            += PRE[index, 1][0]
                sim[f"{block}/DecPointing"]           += PRE[index, 2][0]
                sim[f"{block}/SolarPanelOrientation"] += PRE[index, 3][0]
                # Print to bash
                if self.verbose > 1:
                    print('Applying pointing errors     (PRE FromFile)')
        #---------------------------------------------------------------------                    
            
        # Attitude Orbit Control System (AOCS) jitter
        # First check if file from payload is present
        # NOTE: Included if "instrumentACS.txt" is available in input
        # TODO update to csv!
        inputFileAOCS = self.inputDir.joinpath('instrumentACS.txt')
        if inputFileAOCS.is_file():
            sim["Platform/UseJitter"]      = True
            sim["Platform/JitterSource"]   = 'FromFile'
            sim["Platform/JitterFileName"] = inputFileAOCS

        # Check if "AOCS_Q<quarterNo>.txt" is present to be reused for all quarters
        # NOTE This is not recommended but was used for PLATO-KUL-PL-TN-0023
        elif self.reuseJitter:
            sim["Platform/UseJitter"]    = True
            sim["Platform/JitterSource"] = 'FromFile'
            inputFileAOCS   = self.inputDir.joinpath(sim["Platform/JitterFileName"])
            inputFileAOCS_Q = f'{self.inputDir}/AOCS_Q{self.quarter}.txt'
            # Check if it exists or else create new time column
            if Path(inputFileAOCS_Q).is_file():
                pass
            elif self.quarter == 1:
                os.system(f'cp {inputFileAOCS} {inputFileAOCS_Q}')
            else:
                # Load file given in YAML input
                data = np.loadtxt(inputFileAOCS)
                t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]
                # Generate new time column
                cadence = np.diff(t)[0]  # [s -> 8 Hz]
                t = np.arange(len(t)) * cadence + self.timeStart
                # Save data to input folder
                np.savetxt(inputFileAOCS_Q, np.transpose([t,x,y,z]),
                           fmt=['%.3f', '%.9f', '%.9f', '%.9f'])
            # Set filepath to new file
            sim["Platform/JitterFileName"] = inputFileAOCS_Q
        # Print to bash that jitter is included
        if sim["Platform/UseJitter"] and self.verbose > 1:
            if sim["Platform/JitterSource"] == 'FromFile':
                source = 'FromFile'
            else:
                source = 'RedNoise'
            print(f'Applying platform jitter     (ACS {source})')
            
        # CONFIGURE TELESCOPE
            
        # Set the Camera-group ID, Alt (tilt) [deg], and Az [deg]
        # TODO for now it is not possible to use GroupID = Custom for the F-CAMs since this
        #      uses the N-CAM readout and results in a negative exposure time (and error)
        if self.groupID == 'Fast':
            sim["Telescope/GroupID"] = 'Fast'
        else:
            sim["Telescope/GroupID"]      = 'Custom'
            sim["Telescope/TiltAngle"]    = sim["CameraGroups/TiltAngle"][self.group-1]
            sim["Telescope/AzimuthAngle"] = sim["CameraGroups/AzimuthAngle"][self.group-1]

        # Absolute Pointing Error (APE) due to camera misalignments
        # NOTE Below we secure that platonium is backward compatible
        inputFileAPE_csv = self.inputDir.joinpath('instrumentAPE.csv')
        inputFileAPE_txt = self.inputDir.joinpath('instrumentAPE.txt')
        #---------------------------------------------------------------------
        # New implementation for PlatoSim >= 3.7.0
        if inputFileAPE_csv.is_file():
            if self.verbose > 1:
                print('Applying camera misalignment (APE FromFile)')
            APE = pd.read_csv(inputFileAPE_csv)
            # Use (tilt, azimuth) angles to nudge camera pointing
            # NOTE: Included if "instrumentAPE.csv" is available in the input
            if 'TiltAngle' in APE.columns:
                if self.verbose > 0 and APE.shape != (26, 2): 
                    errorcode('warning', 'Cannot apply camera misalignment (tilt, azimuth)! ' +
                              f'instrumentAPE.csv needs to be shape (26, 2)')            
                else:
                    sim["Telescope/TiltAngle"]    += APE.TiltAngle.iloc[self.cameraIndex]
                    sim["Telescope/AzimuthAngle"] += APE.AzimuthAngle.iloc[self.cameraIndex]
            # Use Quaternion to set camera pointing
            # NOTE In the following approach we assume a nominal camera tilt and azimuth
            # and adjust the platform pointing to get desired camera pointing (misalignment)
            elif 'q0' in APE.columns:
                if self.verbose > 0 and APE.shape != (26, 4): 
                    errorcode('warning', 'Cannot apply camera misalignment (quaternions)! ' +
                              f'instrumentAPE.csv needs to be shape (26, 4)')            
                else:
                    if self.verbose > 1 and inputFilePLM.is_file():
                        errorcode('warning', 'Platform pointing is overwritten by APE! ' +
                                  f'Consider removing the "instrumentPLM.csv" file')
                    quaternionCAM = APE.iloc[self.quarter-1].tolist()
                    alpha1, delta1, kappa1 = rf.platformAnglesFromQuaternion(quaternionCAM)
                    # NOTE we need to account for the fact that PRE has been included too
                    alpha0 = False
                    if inputFilePLM.is_file():
                        alpha0, delta0, kappa0 = rf.platformAnglesFromQuaternion(quaternionPLM)
                    elif inputFilePRE_csv.is_file() or inputFilePRE_txt.is_file():
                        alpha0 = sim["Platform/Orientation/Angles/RAPointing"]
                        delta0 = sim["Platform/Orientation/Angles/DecPointing"]
                        kappa0 = sim["Platform/Orientation/Angles/SolarPanelOrientation"]
                    if alpha0:
                        alpha0 += (alpha0 - alpha1)
                        delta0 += (delta0 - delta1)
                        kappa0 += (kappa0 - kappa1)
                        quaternionCAM = rf.platformQuaternionFromAngles(alpha0, delta0, kappa0)
                    # Apply new camera pointing
                    sim["Platform/Orientation/Source"] = 'Quaternion'
                    sim["Platform/Orientation/Quaternion/Components"] = quaternionCAM
        #---------------------------------------------------------------------
        # Old implementation for PlatoSim < 3.7.0
        elif inputFileAPE_txt.is_file():
            if self.verbose > 1:
                print('Applying camera misalignment (APE FromFile)')
            APE = np.loadtxt(inputFileAPE_txt)
            sim["Telescope/TiltAngle"]    += APE[self.cameraIndex, 0]
            sim["Telescope/AzimuthAngle"] += APE[self.cameraIndex, 1]
        #---------------------------------------------------------------------

        # Thermo-Elastic Drift (TED) due to thermal gradient across optical bench
        # NOTE Included if "instrumentTED**.txt" is available in input
        inputFileTED   = self.inputDir.joinpath('instrumentTED.txt')
        inputFileTED_i = self.inputDir.joinpath(f'instrumentTED_group{self.group}.txt')
        if inputFileTED.is_file() or inputFileTED_i.is_file():
            sim["Telescope/UseDrift"]    = True
            sim["Telescope/DriftSource"] = 'FromFile'
            if inputFileTED.is_file():
                sim["Telescope/DriftFileName"] = inputFileTED
            elif inputFileTED_i.is_file():
                sim["Telescope/DriftFileName"] = inputFileTED_i
        if sim["Telescope/UseDrift"] and self.verbose > 1:
            if sim["Telescope/DriftSource"] == 'FromFile':
                source = 'FromFile'
            else:
                source = 'RedNoise'
            print(f'Applying camera drift        (TED {source})')
                    
        # CONFIGURE CAMERA
        
        # Turn off effect of kinematic aberration
        if self.noAberrCorr:
            sim['Camera/IncludeAberrationCorrection'] = False
        else:
            sim['Camera/IncludeAberrationCorrection'] = True

        # Change constant parameters for CAMERA YAML block
        inputFileCAM = self.inputDir.joinpath('instrumentCAM.csv')
        if inputFileCAM.is_file():
            if self.verbose > 1:
                print('Applying camera parameters   (CAM FromFile)')
            CAM = pd.read_csv(inputFileCAM)
            if CAM.shape[0] != 26:
                errorcode('warning', 'File "instrumentCAM.csv" needs 26 rows (one per camera)!')
            else:            
                if 'FocalLength' in CAM.columns:
                    sim['Camera/FocalLength/Source'] = 'ConstantValue'
                    sim['Camera/FocalLength/ConstantValue'] = CAM.FocalLength.iloc[self.cameraIndex] * 1e-3
                if 'k1' in CAM.columns:
                    coeff_normal = []
                    coeff_normal.append(CAM.k1.iloc[self.cameraIndex])
                    coeff_normal.append(CAM.k2.iloc[self.cameraIndex]) if 'k2' in CAM.columns else coeff_normal.append(0.0)
                    coeff_normal.append(CAM.k3.iloc[self.cameraIndex]) if 'k3' in CAM.columns else coeff_normal.append(0.0)
                    coeff_normal.append(CAM.q1.iloc[self.cameraIndex]) if 'q1' in CAM.columns else coeff_normal.append(0.0)
                    coeff_normal.append(CAM.q2.iloc[self.cameraIndex]) if 'q2' in CAM.columns else coeff_normal.append(0.0)
                    coeff_normal.append(CAM.p1.iloc[self.cameraIndex]) if 'p1' in CAM.columns else coeff_normal.append(0.0)
                    coeff_normal.append(CAM.p2.iloc[self.cameraIndex]) if 'p2' in CAM.columns else coeff_normal.append(0.0)
                    sim["Camera/IncludeFieldDistortion"] = True
                    sim["Camera/FieldDistortion/Source"] = 'ConstantValue'
                    sim["Camera/FieldDistortion/ConstantCoefficients"] = coeff_normal
                else:
                    errorcode('warning', 'No field distortion coefficients found in CAM! ' +
                              'Using default values from YAML file')
                if 'ik1' in CAM.columns:
                    coeff_inverse = []
                    coeff_inverse.append(CAM.ik1.iloc[self.cameraIndex])
                    coeff_inverse.append(CAM.ik2.iloc[self.cameraIndex]) if 'ik2' in CAM.columns else coeff_inverse.append(0.0)
                    coeff_inverse.append(CAM.ik3.iloc[self.cameraIndex]) if 'ik3' in CAM.columns else coeff_inverse.append(0.0)
                    coeff_inverse.append(CAM.iq1.iloc[self.cameraIndex]) if 'iq1' in CAM.columns else coeff_inverse.append(0.0)
                    coeff_inverse.append(CAM.iq2.iloc[self.cameraIndex]) if 'iq2' in CAM.columns else coeff_inverse.append(0.0)
                    coeff_inverse.append(CAM.ip1.iloc[self.cameraIndex]) if 'ip1' in CAM.columns else coeff_inverse.append(0.0)
                    coeff_inverse.append(CAM.ip2.iloc[self.cameraIndex]) if 'ip2' in CAM.columns else coeff_inverse.append(0.0)
                    sim["Camera/FieldDistortion/ConstantInverseCoefficients"] = coeff_inverse
                else:
                    errorcode('warning', 'No inverse distortion coefficients found in CAM! ' +
                              'Using default values from YAML file')
                
        # CONFIGURE CCD
        
        # Change constant parameters for CCD block
        inputFileCCD = self.inputDir.joinpath('instrumentCCD.csv')
        if inputFileCCD.is_file():
            # Units [mm, mm, deg]
            CCD = pd.read_csv(inputFileCCD)
            if CCD.shape != (104, 3):
                errorcode('warning', 'File "instrumentCCD.csv" needs dimention (104, 3)!')
            else:
                if self.verbose > 1:
                    print('Applying CCD positions       (CCD FromFile)')
                OriginOffsetX = []
                OriginOffsetY = []
                Orientation   = sim['CCDPositions/Orientation']
                # Add small offset to orientations
                ccdIndex = 4 * self.cameraIndex
                for i in [ccdIndex, ccdIndex+1, ccdIndex+2, ccdIndex+3]:
                    ccd = CCD.iloc[i]
                    OriginOffsetX.append(ccd.OriginOffsetX)
                    OriginOffsetY.append(ccd.OriginOffsetY)
                    Orientation[i-ccdIndex] += ccd.Orientation
                sim["CCDPositions/OriginOffsetX"] = OriginOffsetX
                sim["CCDPositions/OriginOffsetY"] = OriginOffsetY
                sim["CCDPositions/Orientation"]   = Orientation
        
        # TODO Allow to include default TXT file "cl2bCcds.txt"
        #sim["CCDPositions/UsePositionsFromFile"] = True
        #sim['CCDPositions/PositionsFileName']    = inputFileCCD
                
        # Thermal gain transients due to data gaps
        # NOTE Included if "instrumentGTT.txt" is available in input
        inputFileGTT = self.inputDir.joinpath('instrumentGTT.txt')
        if inputFileGTT.is_file():
            sim["CCD/Temperature"]         = "FromFile"
            sim["CCD/TemperatureFileName"] = inputFileGTT
            if self.verbose > 1:
                print('Applying thermal transients  (GTT FromFile)')
                
        # CONFIGURE FULL-FRAME OR SUBFIELD SIMULATION
        
        if self.fullFrame:
            # Turn off photometry
            sim['Photometry/IncludePhotometry'] = False
            # Argument needed for later
            self.isOnCCD = True
            # Set CCD parameters
            sim["CCD/Position"] = str(self.ccdCode)
            sim["SubField/NumRows"]    = sim["CCDPositions/NumRows"][0]
            sim["SubField/NumColumns"] = sim["CCDPositions/NumColumns"][0]
            # Control output requirements
            sim["ControlHDF5Content/GroupByExposure"]    = True
            sim["ControlHDF5Content/WritePixelMaps"]     = True
            sim["ControlHDF5Content/WriteStarPositions"] = True

            # print(sim['SubField/ZeroPointRow'])
            # print(sim['SubField/ZeroPointColumn'])
            # exit()
            # dex = int(self.ccdCode - 1)
            # if self.groupID == 'Fast':
            #     sim["CCD/FirstRowExposed"] = sim["CCDPositions/FirstRowForFastCamera"][dex]
            # else:
            #     sim["CCD/FirstRowExposed"] = sim["CCDPositions/FirstRowForNormalCamera"][dex]
            
                # TODO readout time is not correct if not GroupID = Fast -> New feature
                # shieldRows = sim["CCDPositions/MetallicShield/ShieldRowCoordinates"]
                # shieldCols = sim["CCDPositions/MetallicShield/ShieldColumnCoordinates"]
                # sim["SubField/ZeroPointRow"]    = shieldRows[0]
                # sim["SubField/ZeroPointColumn"] = shieldCols[0]
                # sim["SubField/NumRows"]         = shieldRows[1] - shieldRows[0]
                # sim["SubField/NumColumns"]      = shieldCols[1] - shieldCols[0]
            # else:
            #     sim["SubField/NumRows"]    = sim["CCD/NumRows"][0]
            #     sim["SubField/NumColumns"] = sim["CCD/NumColumns"][0]
        else:
            # Check for sources in subfield
            self._check_observability(sim)            
            # Create data frame for printing and saving
            c = [self.colID, 'ra [deg]', 'dec [deg]', 'mag',
                 'CCD', 'xCCD [pix]', 'yCCD [pix]',
                 'rOA [deg]', 'xFP [mm]', 'yFP [mm]', 'Ncon']
            d = {self.colID: [self.df[self.colID]], 'mag': [self.df['mag']],
                 'ra [deg]': [self.df['ra']], 'dec [deg]': [self.df['dec']],
                 'CCD': [self.ccdCode], 'xCCD [pix]': [self.xCCD], 'yCCD [pix]': [self.yCCD],
                 'rOA [deg]': [self.rOA], 'xFP [mm]': [self.xFP], 'yFP [mm]': [self.yFP],
                 'Ncon': self.numCon}
            self.df0 = pd.DataFrame(d, columns=c)
            # Print data frame
            if self.verbose > 1:
                print('\nInformation about stellar target')
                print(self.df0)

        # Return simulation object
        return sim

    
    def _check_observability(self, sim):
        """Function to check if the target is observable.
        """        
        numColSubfield = sim["SubField/NumColumns"]
        numRowSubfield = sim["SubField/NumRows"]
        raTargetRad    = np.deg2rad(self.df['ra'])
        decTargetRad   = np.deg2rad(self.df['dec'])

        # Try to set a subfield around the coordinates on one of the 4 CCDs of the camera.
        # This will fail (return = False) if visible by any of the 4 CCDs, i.e.:
        # 1) If the subfield is outside camera FOV; 
        # 2) If the subfield falls in a CCD gap;
        # 3) If the subfield is too large to entirely fit on a CCD.
        # If successful, the CCD and subfield parameters is sets in the 'sim' object.        
        info = sim.setSubfieldAroundSkyCoordinates(raTargetRad, decTargetRad,
                                                   numColSubfield, numRowSubfield,
                                                   normal=self.normal, returnInfo=True)
        if info[0] == None:
            if self.verbose > 0:
                message  = (f"{self.colID} {self.df[self.colID]} (subfield {self.targetNo}) " +
                            'do not fall on any of the CCDs for N-CAM ' +
                            f'{self.group}.{self.camera} and Q{self.quarter}!')
                errorcode('warning', message)
            # Terminate script
            exit()
        else:
            self.isOnCCD = True
            self.ccdCode = info[0]
            self.xCCD    = info[1]
            self.yCCD    = info[2]
            self.xFP     = info[3]
            self.yFP     = info[4]

        # Only continue if ccdCode is found
        if self.ccdCode:
            # Check if string is F-CAM
            if len(self.ccdCode) == 2:
                self.ccdCode = self.ccdCode[0]

            # Add CCD time-shift to time points
            self.timeStart += float(sim['CCDPositions/TimeShift'][int(self.ccdCode)-1])
            self.time = np.arange(self.numExposures) * self.cadence + self.timeStart
        else:
            if self.verbose > 0:
                errorcode('warning', 'Star falls within a CCD gap!')
            # Terminate script
            exit()

        # Gnomonic radial distance of coordinate away from OA
        focalLength = float(sim["Camera/FocalLength/ConstantValue"]) * 1e3 # [m]->[mm]
        self.rOA = np.rad2deg(rf.gnomonicRadialDistanceFromOpticalAxis(self.xFP, self.yFP,
                                                                       focalLength))
        # Account for maximum optical distortion: rAO = 19.8deg -> T = 1%
        if self.rOA > sim['CCD/RelativeTransmissivity/RadiusFOV']:
            if self.verbose > 0:
                message  = (f"{self.colID} {self.df[self.colID]} (subfield {self.targetNo}) " +
                            f'is outside camera FOV (rOA={self.rOA:.2f} deg) ' +
                            f'for N-CAM {self.group}.{self.camera} and Q{self.quarter}!')
                errorcode('warning', message)
            # Terminate script
            exit()
        
    
    def create_seeds(self, sim):
        """Module to select and load the random seeds.
        """
        # Initialise random number generator after user or clock
        if not self.seed:
            seed = 123456789
            rng  = np.random.default_rng()
        else:
            seed = self.seed
            rng  = np.random.default_rng(seed)

        # Seeds needs to be available for L1 pipeline
        self.seedJitter = seed * self.quarter
        self.seedTarget = seed * self.quarter + 1000 * self.targetNo + self.group * self.camera

        # Jitter (relevant for red noise) only depends on the quarter
        if sim["Platform/UseJitter"] == 'yes' and sim["Platoform/JitterSource"] == 'RedNoise':
            sim["RandomSeeds/JitterSeed"] += self.seedJitter

        # Drift (relevant for red noise) depends on quarter, group, and camera
        if sim["Telescope/UseDrift"] and sim["Telescope/DriftSource"] == 'RedNoise':
            sim["RandomSeeds/DriftSeed"] += (seed * self.quarter + 4 * self.group +
                                             24 * self.camera + 100000)

        # Lastly, the remaining seeds needs to differ for all times and for all subfields
        if self.seed:
            sim["RandomSeeds/ReadOutNoiseSeed"] += self.seedTarget
            sim["RandomSeeds/PhotonNoiseSeed"]  += self.seedTarget
            sim["RandomSeeds/DarkSignalSeed"]   += self.seedTarget
            sim["RandomSeeds/FlatFieldSeed"]    += self.seedTarget
            sim["RandomSeeds/CosmicSeed"]       += self.seedTarget
        else:
            sim["RandomSeeds/ReadOutNoiseSeed"] += rng.integers(1e9, size=1)[0]
            sim["RandomSeeds/PhotonNoiseSeed"]  += rng.integers(1e9, size=1)[0]
            sim["RandomSeeds/DarkSignalSeed"]   += rng.integers(1e9, size=1)[0]
            sim["RandomSeeds/FlatFieldSeed"]    += rng.integers(1e9, size=1)[0]
            sim["RandomSeeds/CosmicSeed"]       += rng.integers(1e9, size=1)[0]

            
    def create_inputfiles(self, sim):
        """Function to create ascii input files for PlatoSim.
        """        
        # SAVE STELLAR CATALOGS AND TARGET LISTS

        # Save catalog and load it into the inputfile
        self.starCatalogFile = f'{self.outputDir}/{self.outputFileName}.cat'
        sim.createStarCatalogFile(self.ds.ra, self.ds.dec, self.ds.mag, self.ds.ids,
                                  self.starCatalogFile)

        # Print catalogue
        if self.verbose > 1 and not self.fullFrame:
            print('\nStar catalog used in simulation')
            df1 = pd.DataFrame({'RA [deg]': self.ds.ra,
                                'Dec [deg]': self.ds.dec,
                                'P [mag]': self.ds.mag,
                                'Dis [pix]': np.append(0, self.dc['dis'])/15.})
            print(df1, '\n')

        # SAVE LIST OF VARIABLE SOURCES
        
        # Automatically activate varsource (if the user forgets)
        if self.varSourceFile or self.varSourceList:
            sim['Sky/IncludeVariableSources'] = True

        # NOTE if a user-defined filename for the variable data is parsed
        # then a variable source list is created automatically
        if sim['Sky/IncludeVariableSources'] is True:

            # First priority to varSourceList -> Allows variability for all stars
            if self.varSourceList:
                # Make sure that wrong paths are detected
                self.varSourceList = Path(self.varSourceList).resolve()
                if self.varSourceList.is_file():
                    sim["Sky/VariableSourceList"] = self.varSourceList
                else:
                    errorcode('error', 'VariableSourceList do not exist, check file path!')
                # Print to bash
                if self.verbose > 1:
                    print(f'Applying variable-source list ({self.varSourceList.name})')
                    
            # Second priority to varSourceFile -> Allows variability for target only
            elif self.varSourceFile:
                # If varSourceFile is parsed
                self.varSourceFile = Path(self.varSourceFile).absolute()
                if self.varSourceFile.is_file():
                    # Automaticallt create and save varSourceList to file
                    self.varSourceList = f'{self.outputDir}/{self.outputFileName}.var'
                    sim.createVariableSourceList('1', str(self.varSourceFile), self.varSourceList)
                else:
                    errorcode('error', 'VariableSourceFile do not exist, check file path!')
                # Print to bash
                if self.verbose > 1:
                    print(f'Applying variable-source file ({self.varSourceFile.name})')
                    
        # SAVE PHOTOMETRY FILE
        
        # NOTE if a user defined file name for the photometry file is parsed
        # then a photometry file list is created automatically
        if sim['Photometry/IncludePhotometry'] is True:
            photometryList = self.inputDir.joinpath('photometry.txt')
            if os.path.exists(photometryList) is False:
                np.savetxt(photometryList, np.array([]), header='1', comments='')
            sim["Photometry/TargetFileName"] = photometryList
            self.photometry = True
            # Print to bash
            if self.verbose > 1:
                print(f'Applying on-board photometry  ({photometryList.name})')
        else:
            self.photometry = False

            
    def show_subfield(self, sim):
        """Function to show:
        1) where the target star is situated in the CCD focal plane
        2) the first subfield/imagette of the simulation
        This function exit the simulation.
        """
        # Print to bash
        if self.verbose > 1:
            if self.groupID == 'Fast':
                errorcode('message', '\n[PlatoSim]: Visualizing simulation for ' +
                          f'F-CAM {self.cameraID} Q{self.quarter}\n')
            else:
                errorcode('message', f'\n[PlatoSim]: Visualizing simulation for ' +
                          f'N-CAM {self.group}.{self.camera} Q{self.quarter}\n')

        # Control the content of the hdf5 output files
        sim.turnOffAllOutput()
        sim["ObservingParameters/NumExposures"]      = 1
        sim["ControlHDF5Content/WritePixelMaps"]     = True
        sim["ControlHDF5Content/WriteStarPositions"] = True

        # Add photometric mask to plot if available
        if sim['Photometry/IncludePhotometry']: mask = 1
        else: mask = None

        # Run simulation for first image cadence
        self.outputSimName = self.outputDir.joinpath(self.outputFileName)
        sim.outputDir = self.outputDir
        f = sim.run(removeOutputFile=self.overwrite)
        
        # Plot star in CCD focal plane
        # Fetch info from YAML file since setting the subfied figure the inputfile
        if sim["Platform/Orientation/Source"] == 'Quaternion':
            q_EQ2PLM = sim["Platform/Orientation/Quaternion/Components"]
            alpha, delta, kappa = rf.platformAnglesFromQuaternion(q_EQ2PLM)
            alpha, delta, kappa = np.rad2deg(alpha), np.rad2deg(delta), np.rad2deg(kappa)
        else:
            alpha, delta, kappa = (sim["Platform/Orientation/Angles/RAPointing"],
                                   sim["Platform/Orientation/Angles/DecPointing"],
                                   sim["Platform/Orientation/Angles/SolarPanelOrientation"])
        # TODO plot do not work for F-CAM yet!
        if not self.fullFrame and not self.groupID == 'Fast':
            fig1 = plt.figure(figsize=(12,10))
            drawStarInCCDfocalPlane(fig1, sim,
                                    self.df0['xCCD [pix]'][0], self.df0['yCCD [pix]'][0],
                                    self.df0['CCD'][0], self.group, alpha, delta, kappa)

        # Show subfield for first cadence
        if self.fullFrame:
            figsize = (12,12)
            showStarPositions = False
            showMaskOfStarID  = False
            showGrid          = False
            title             = False
            imgScale          = "auto"
            cmap              = 'cubehelix'
            clipPercentile    = 1
        else:
            figsize = (6,6)
            showStarPositions = 'PIC'
            showMaskOfStarID  = '1'
            title = f'{self.colID} {int(self.df[self.colID])} ({float(self.df.mag):.2f} mag)'
            clipPercentile    = 2
            imgScale          = "auto"
            cmap              = 'Blues_r'
            showGrid          = True

        # Check that if any stars are detected
        try:
            f.getStarCoordinates(self.beginExposureNr)
        except:
            errorcode('error', 'No stars detected within subfield. Check pointing field!')
            showStarPositions = False

        # Plot the subfield
        if self.df0.mag.iloc[0] < 6:
            # Plot for bright stars (GO proposals)
            fig2, _ = f.showExtendedImage(
                self.beginExposureNr,
                showStarPositions=showStarPositions,
                showMaskOfStarID=mask,
                useTitle=f'N-CAM {self.group}.{self.camera} observation of {self.sample} (Gaia DR3 {title[7:27]})',
                #useTitle=f'F-CAM red observation of {self.sample} (Gaia DR3 {title[7:27]})',
                colorMap=cmap,
                colorBar=False,
                imgScale='log', clip=20,
                showGrid=True,
                flipAxes=True,
                ds=self.ds,
                figsize=(15, 5)
            )
        else:
            fig2, _ = f.showImage(
                self.beginExposureNr,
                showStarPositions=showStarPositions,
                clip=clipPercentile,
                showMaskOfStarID=mask,
                useTitle=title,
                colorMap=cmap,
                colorBar=True,
                imgScale=imgScale,
                showGrid=showGrid,
                figsize=figsize
            )
        
        # Save figure if requested
        if self.savePlot:
            if not self.group == 5:
                filename1 = f'{self.outputDir}/focalplane_{self.outputFileName}.png'
                fig1.savefig(filename1, bbox_inches='tight', dpi=200)
            filename2 = f'{self.outputDir}/subfield_{self.outputFileName}.png'
            fig2.savefig(filename2, bbox_inches='tight', dpi=200)
            
        # Remove the output files
        if os.path.isfile(str(self.outputSimName) + '.hdf5'):
            os.system(f'rm {self.outputDir}/{self.outputFileName}*')

    #--------------------------------------------------------------#
    #                         DEFAULT SETUP                        #
    #--------------------------------------------------------------#

    def run_sim_normal(self, sim):
        """Module to run PlatoSim only.
        """
        # Print to bash
        if self.verbose > 1:
            tracemalloc.start()
            if self.fullFrame:
                ccdID = f' CCD {self.ccdCode}'
            else:
                ccdID = ''
            if self.groupID == 'Fast':
                errorcode('message', '\n[PlatoSim]: Simulating' +
                          f'{ccdID} F-CAM {self.cameraID} Q{self.quarter} ' +
                          f'for {self.numExposures} exposures')
            else:
                errorcode('message', '\n[PlatoSim]: Simulating' +
                          f'{ccdID} N-CAM {self.group}.{self.camera} Q{self.quarter} '
                          f'for {self.numExposures} exposures')

        # Select the number of exposures
        sim["ObservingParameters/NumExposures"] = self.numExposures

        # Save images if animation is requested
        if self.animation:
            sim["ControlHDF5Content/WritePixelMaps"]     = True
            sim["ControlHDF5Content/WriteStarPositions"] = True

        # Run simulation
        sim.outputDir = self.outputDir
        simFile = sim.run(removeOutputFile=self.overwrite, logLevel=self.verbose_platosim)

        # Common files to always remove (unless debug mode)
        if self.isOnCCD:
            cat  = Path(self.starCatalogFile)
            log  = Path(str(self.outputSimName) + '.log')
            yaml = Path(str(self.outputSimName) + '.yaml')
            if not self.pipeline and self.verbose < 3:
                if cat.is_file(): cat.unlink()
                if log.is_file(): log.unlink()
                if yaml.is_file(): yaml.unlink()
            if self.varSourceFile:
                Path(self.varSourceList).unlink()

        # Define output file name
        outputFile = f'{self.outputSimName}.hdf5'

        # FULL-FRAME CCD IMAGE
        
        if self.fullFrame:
            # Save full-frame catalogue for first exposure
            # Fetch simulation and stellar positions
            f = SimFile(outputFile)
            ID, row, col, xFP, yFP, flux = f.getStarCoordinates(self.beginExposureNr)
            
            # Make some checks
            if self.dx.shape[0] == 0:
                errorcode('warning', 'Stellar catalogue is empty')
            elif ID is None:
                errorcode('warning', 'No stars detected on the CCD')
            else:
                # Select detected stars
                df = self.dx.iloc[ID]

                # Indices are the star IDs
                df = ut.pdAddColumn(df, df.index, 'starID')
                if 'index' in df: df.drop(columns=['index'], inplace=True)
                df = df.reset_index(drop=True)

                # Add stellar positions
                df['flux'] = flux
                df['xCCD'] = col
                df['yCCD'] = row
                df['xFP']  = xFP
                df['yFP']  = yFP
                
                # Add gnomonic radial distance [deg]
                focalLength = float(sim["Camera/FocalLength/ConstantValue"]) * 1000
                rOA = rf.gnomonicRadialDistanceFromOpticalAxis(df.xFP, df.yFP, focalLength)
                df['rOA'] = np.rad2deg(rOA)

                # Save to file
                df = df.reset_index(drop=True)
                df.to_feather(f'{self.outputSimName}.ftr')

        # SUBFIELD ANIMATION
        
        if self.animation:
            # Adjust number of images to skip and frame rate
            if   self.cadence ==  25.0: fps, nskip = 50, 800
            elif self.cadence ==  50.0: fps, nskip = 25, 500
            elif self.cadence == 600.0: fps, nskip = 25, 50
            plotSubfieldAnimation(outputFile,
                                  outputFileName=str(self.outputSimName),
                                  cadence=self.cadence,
                                  frameRate=fps,
                                  skipNimages=nskip,
                                  numImages=False,
                                  colorMap="magma",
                                  clipPercentile=5.0, 
                                  showStarPositions='PIC',
                                  showMaskOfStarID='1',
                                  useTitle=True,
                                  showGrid=True,
                                  figsize=(6,6))

        # RESOURCES
        
        if self.verbose > 1:
            # Execution time of module
            self.tocPlatoSim = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            # Max RAM memory [Mb]
            self.memRamPlatoSim = np.ceil(tracemalloc.get_traced_memory()[1]/1e6)
            tracemalloc.stop()

            # Storage memory of HDF5 file [Mb]
            self.memDiskPlatoSim = np.ceil(Path(outputFile).stat().st_size/1e6)

            
    def run_reduction(self, sim):
        """Module to perform data reduction.
        """
        # Print to bash
        if self.verbose > 1:
            errorcode('module', '\nPost-processing\n')

        # Load light curve
        from platosim.lightcurve import LightCurve
        lc = LightCurve(f'{self.outputSimName}.hdf5', path=self.outputDir)

        # GAIN TRANSIENTS
        # TODO check exp model! Ask Pierre for correction 

        # Apply step if CCD(T) file exists
        # inputFileGTT = self.inputDir.joinpath('instrumentGTT.txt')

        # if inputFileGTT.is_file():
        #     if self.verbose > 1:
        #         print('Running transient  model : gain(T)')

        #     # Load CCD gain temperature file
        #     dt = pd.read_csv(inputFileGTT, sep=' ', names=['time', 'temp'])
        #     dt = dt.iloc[self.beginExposureNr:self.beginExposureNr+self.numExposures]
        #     temp = dt.temp.to_numpy()

        #     # Fetch the gap durations
        #     inputFileGap = self.inputDir.joinpath('instrumentGAP.tab')
        #     dg = pd.read_feather(inputFileGap)
        #     tdur = dg.td.iloc[0] / 86400

        #     # Use correct gain from either F or E side
        #     tempNominal   = sim['CCD/NominalOperatingTemperature']
        #     gainCCD       = sim['CCD/Gain/RefValueRight']
        #     gainFEE       = sim['FEE/Gain/RefValueRight']
        #     gainStability = sim['FEE/Gain/Stability']

        #     # Perfect correction
        #     lc.correct_gain(temp, tdur, tempNominal, gainCCD, gainFEE, gainStability,
        #                     replace=True, plot=self.plotPost)

        # DETRENDING
        
        if self.detrend is not None:
            if self.verbose > 1:
                print(f'Running detrending model : {self.detrend}')

            # Perform detrending
            lc.detrend(model=self.detrend,
                       poly_degree=self.poly_deg,
                       replace=True,
                       plot=self.plotPost)

            if self.verbose > 1:
                self.tocDetrend = datetime.datetime.now() - self.tic
                self.tic = datetime.datetime.now()


        # STITCH MASK-UPDATES
        
        if self.stitch is not None and len(lc.mask_update_events()) > 1:
            if self.verbose > 1:
                print(f'Running stitching  model : {self.stitch}')

            # Perform stitching
            lc.stitch(method=self.stitch, segment=5, replace=True, plot=self.plotPost)
            self.stitchActive = True

            if self.verbose > 1:
                self.tocStitch = datetime.datetime.now() - self.tic
                self.tic = datetime.datetime.now()

        # OUTLIER REJECTION
        
        if self.clipWotan:
            if self.verbose > 1:
                print('Running sigma-clip model : wotan')

            if self.detrend: flux_unit='ppt'
            else: flux_unit='e/s'

            # Auto select sigma from emperical tests
            # Cuts optimized for N-CAMs of 25s cadence
            if self.clipSigma is not None:
                sigma_lower = sigma_upper = self.clipSigma
            else:
                # Upper is less senitive to variation
                if self.df.mag <= 10:
                    sigma_upper = 5
                elif self.df.mag > 10 and self.df.mag < 11:
                    sigma_upper = 4.5
                else:
                    sigma_upper = 4
                
                # Exoplanets (protect transits)
                if self.detrend == 'wotan':
                    sigma_lower = 10
                # Eclipsing binaries (protect eclipses)
                elif self.detrend == 'lowess':
                    sigma_lower = 6
                else:
                    sigma_lower = sigma_upper

            # Perform sigma-clipping
            lc.clip(model='wotan',
                    sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                    window=self.clipWindow, replace=True, plot=self.plotPost,
                    flux_unit=flux_unit)
            
            if self.verbose > 1:
                self.tocWotanClip = datetime.datetime.now() - self.tic
                self.tic = datetime.datetime.now()

        # Save dataset
        df = lc.data()
        df = df.reset_index(drop=True)
        df.to_feather(f'{self.outputSimName}.ftr')

        #---------------------------------------- Debugging
        # Check reduced data
        # if self.plotPost:
        #     # Compute the residuals
        #     df['flux_res'] = df.flux #(df.flux - 1)*1e6
        #     # Regression model of residuals
        #     import statsmodels.api as sm
        #     lc = df.rename(columns={'time':'x', 'flux_res':'y'})
        #     lc['x'] = lc['x'].subtract(lc['x'].min())
        #     model = 'y ~ x'
        #     lsFit = sm.OLS.from_formula(formula=model, data=lc).fit()
        #     lsFit.summary(alpha=0.05)
        #     # Plot regression model and residuals
        #     st.plot_modelfit(lc, lsFit, model, lsModel='OLS', theme='g',
        #                      xlab='Time [days]', ylab='Residuals [ppt]')
        #     st.plot_residuals(lc, lsFit, theme='g')
        #     st.plot_standardized_residuals(lc, lsFit, K=2, reg='x', lsModel='OLS')
        #-------------------------------------------------------------------
        
    #--------------------------------------------------------------#
    #                    L1 PIPELINE MODULES                       #
    #--------------------------------------------------------------#

    def control_hdf5(self):
        """Module to control HDF5 content for L1 pipeline.
        """
        # Include HDF5 content
        sim["ControlHDF5Content/GroupByExposure"]             = False
        sim["ControlHDF5Content/WritePixelMaps"]              = True
        sim["ControlHDF5Content/WriteBiasMaps"]               = True
        sim["ControlHDF5Content/WriteSmearingMaps"]           = True
        sim["ControlHDF5Content/WriteFlatfieldMap"]           = True
        sim["ControlHDF5Content/WriteThroughputMaps"]         = True
        sim["ControlHDF5Content/WriteTransmissionEfficiency"] = True
        sim["ControlHDF5Content/WriteBackgroundMap"]          = True
        sim["ControlHDF5Content/WriteCTI"]                    = False
        sim["ControlHDF5Content/WriteSubPixelImages"]         = False
        sim["ControlHDF5Content/WriteHighResolutionPSF"]      = True
        sim["ControlHDF5Content/WriteACS"]                    = True
        sim["ControlHDF5Content/WriteTelescopeACS"]           = True
        sim["ControlHDF5Content/WriteStarCatalog"]            = True
        sim["ControlHDF5Content/WriteStarPositions"]          = True
        sim["ControlHDF5Content/WriteGhostPositions"]         = False
        sim["ControlHDF5Content/WriteCosmics"]                = False
        sim["ControlHDF5Content/WriteDiffusedPSF"]            = True

        
    def run_microscan(self, sim):
        """Module to run a microscan sequence with PlatoSim.
        """
        # Print to bash
        if self.verbose > 1:
            errorcode('module', '\nMicroscanning & PSF inversion')

        # Check if mapped PSFs are used and apply correct resolution
        if sim["PSF/Model"] == 'MappedFromFile':
            sim["SubField/SubPixels"] = 64
        else:
            sim["SubField/SubPixels"] = 128

        # Prepare for the Archimedean jitter spiral pattern
        nimages = 428
        sim["Platform/UseJitter"]               = 'yes'
        sim["Platform/JitterSource"]            = 'FromFile'
        sim["ObservingParameters/NumExposures"] = nimages
        sim['ObservingParameters/CycleTime']    = 25

        # Time of microscan simulation needs to match quarter for PlatoSim to run successfully
        # NOTE Here a new file is created with a appropriate time column to match the observation
        # NOTE This is done once per quarter and CCD time-shift and the file is saved to the
        # simulation folder
        spiralFileName     = 'microscan_spiral_8Hz_3h_BC.txt'
        spiralFileBase     = f'{self.platosimInputDir}/{spiralFileName}'
        spiralFileQuarterN = f'{self.microscanDirStarID}/{spiralFileName}'

        # Download microscan file if first run of L1 pipeline
        if not Path(spiralFileBase).is_file():
            print('Downloading miscroscanning file..')
            ut.downloadFromFTP(filename=spiralFileName,
                               outputDir=self.platosimInputDir,
                               server='plato')

        if self.quarter == 1 and int(self.df0['CCD']) == 1:
            os.system(f'cp {spiralFileBase} {spiralFileQuarterN}')
        else:
            # Load file
            data = np.loadtxt(spiralFileBase)
            t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]
             # Generate new time column
            cadence = np.diff(t)[0]  # [s -> 8 Hz]
            t = np.arange(len(t)) * cadence + self.timeStart
            # Save data to input folder
            np.savetxt(spiralFileQuarterN, np.transpose([t,x,y,z]),
                       fmt=['%.3f', '%.9f', '%.9f', '%.9f'])

        # Load Archimedean spiral
        sim["Platform/JitterFileName"] = spiralFileQuarterN

        # HDF5 content to always exclude
        sim["Telescope/UseDrift"]                     = False
        sim["Sky/IncludeVariableSources"]             = False
        sim["Photometry/IncludePhotometry"]           = False
        sim["ControlHDF5Content/WriteSubPixelImages"] = False

        # HDF5 content to always include
        sim["ControlHDF5Content/WritePixelMaps"]              = True
        sim["ControlHDF5Content/WriteBiasMaps"]               = True
        sim["ControlHDF5Content/WriteSmearingMaps"]           = True
        sim["ControlHDF5Content/WriteThroughputMaps"]         = True
        sim["ControlHDF5Content/WriteFlatfieldMap"]           = True
        sim["ControlHDF5Content/WriteHighResolutionPSF"]      = True
        sim["ControlHDF5Content/WriteTransmissionEfficiency"] = True
        sim["ControlHDF5Content/WriteStarPositions"]          = True
        sim["ControlHDF5Content/WriteACS"]                    = True
        sim["ControlHDF5Content/WriteCosmics"]                = False
        sim["ControlHDF5Content/WriteStarCatalog"]            = True
        sim["ControlHDF5Content/WriteTelescopeACS"]           = True
        sim["ControlHDF5Content/WriteCTI"]                    = True
        # NOTE: jmcc psim2datastruc was complaining about WriteDiffusePSF being disabled.
        #       Enabling to see if that helps.
        sim["ControlHDF5Content/WriteDiffusedPSF"]            = True

        # Save catalog and load it into the inputfile
        numStar = self.numCon + 1
        self.ds.ids = np.arange(self.targetNo, self.targetNo + numStar, 1) + 1

        # Run the microscan simulation
        if self.verbose > 1:
            errorcode('message', f'\n[PlatoSim]: Simulating {nimages} imagettes' +
                      ' along Archimedean spiral')
        sim.outputDir = self.microscanDirStarID
        simFile = sim.run(removeOutputFile=self.overwrite, logLevel=self.verbose_platosim)

        # Execution time of module
        if self.verbose > 1:
            self.tocMicroscan = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            
    def run_L1_microscan(self):
        """Splits L1 processing of microscan into its own function.
        This allows better testing of L1
        """
        # Change directory needed to execute scripts
        os.chdir(self.microscanDir)

        if self.verbose > 1:
            errorcode('message', '\n[psim2datastruc]: Pre-processing imagettes')
        mag_err = 2.5*(self.pipeFluxError/100.)/np.log(10.)
        comm = f'psim2datastruc --cam-id {self.cameraID:02d} --prnu_err {self.pipePrnuError} --seed {self.seedTarget} --mag-error {mag_err} --centroid-err {self.pipeAbsCenError} --target_id 1 . {self.starID} {self.starID} 6'
        # Debugging
        if self.verbose > 2:
            print(os.getcwd())
            print(comm)
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('psim2datastruc failed due to the above error!')

        # Get the inverted PSF
        if self.verbose > 1:
            errorcode('message', '\n[gen_psfinv]: Run the PSF inversion')
        comm = f"gen_psfinv --bsres {self.bsres} 1 {self.starID} {self.microscanDirInvers}"
        print(comm)
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('gen_psfinv failed due to the above error!')

        # Check the performance of the inversion!
        if self.verbose > 1:
            errorcode('message', '\n[psfinv_quality]: Check the PSF inversion quality')
        comm = f"psfinv_quality {self.microscanDirInvers}/000000001_inverse_psf.hdf5 {self.starID}/000000001_psf.hdf5"
        print(comm)
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('psfinv_quality failed due to the above error!')

        # Execution time of module
        if self.verbose > 1:
            self.tocInversion = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            
    def run_L1_onground(self):
        """Module to for the on-ground L1 pipeline processing chain.
        """
        # Print to bash
        if self.verbose > 1:
            errorcode('module', '\nOn-ground L1 pipeline')

        # Change directory needed to execute scripts
        os.chdir(self.pipelineDir)

        # PRE-PROCESSING
        if self.verbose > 1:
            errorcode('message', '\n[psim2datastruc]: Pre-processing imagettes')
        mag_err = 2.5*(self.pipeFluxError/100.)/np.log(10.)
        comm = f'psim2datastruc --cam-id {self.cameraID:02d} --prnu_err {self.pipePrnuError} --seed {self.seedTarget} --mag-error {mag_err} --centroid-err {self.pipeAbsCenError} --target_id 1 . {self.starID} {self.starID} 6'
        print(os.getcwd())
        print(comm)
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('psim2datastruc failed due to the above error!')

        # PSF FIITING
        if self.verbose > 1:
            errorcode('message', '\n[gen_pflux_ts]: PSF fitting for light curve generation')

        # build the gen_pflux command
        # NOTE: the psf fittign currently struggles at <0.25 separation with contaminants
        # Reza suggested setting this limit to 0.5 pixels for safety
        if self.pipePsfMethod == 'microscan':
            psf_path = f"{self.microscanDirInvers}/000000001_inverse_psf.hdf5"
            comm = f"gen_pflux_ts --psf {psf_path} --distance-min 0.5"
        else:
            psf_lib_path = f"{self.inputDir}/{self.psfLibraryFilename}"
            comm = f"gen_pflux_ts --psf-library {psf_lib_path} --distance-min 0.5"
        if self.pipePlots:
            comm += " -P"
        if self.noAberrCorr:
            comm += " --ignore-aberration"
        comm += f" 1 {self.starID} {self.starID}"
        print(comm)

        # run the gen_pflux command
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('gen_pflux_ts failed due to the above error!')

        # PROLOGUE
        # Execution time of module
        if self.verbose > 1:
            self.tocOnground = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            
    def run_L1_onboard(self):
        """Module to for the on-board L1 pipeline processing chain.
        """
        # onboard cadences are 50 or 600s, determine --n-average for gen_aflux from the configured cadence
        n_average = int(self.pipeCadence/25)
        if n_average not in [1, 2, 24]:
            self.failed(f'Onboard photometry is done at 25, 50 or 600s (n_average = 1 | 2 | 24)\nCurrent n_average={n_average}')

        # Print to bash
        if self.verbose > 1:
            errorcode('module', '\nOn-board L1 pipeline')

        # Change directory needed to execute scripts
        os.chdir(self.pipelineDir)

        # PRE-PROCESSING
        if self.verbose > 1:
            errorcode('message', '\n[psim2datastruc]: Pre-processing imagettes')
        mag_err = 2.5*(self.pipeFluxError/100.)/np.log(10.)
        comm = f'psim2datastruc --cam-id {self.cameraID:02d} --prnu_err {self.pipePrnuError} --seed {self.seedTarget} --mag-error {mag_err} --centroid-err {self.pipeAbsCenError} --target_id 1 . {self.starID} {self.starID} 6'
        print(os.getcwd())
        print(comm)
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('psim2datastruc failed due to the above error!')

        # APERTURE PHOTOMETRY
        if self.verbose > 1:
            errorcode('message', '\n[gen_aflux_ts]: Aperture photometry ala Marchiori+2019')

        # build the gen_aflux command
        if self.pipePsfMethod == "microscan":
            psf_path = f"{self.microscanDirInvers}/000000001_inverse_psf.hdf5"
            comm = f"gen_aflux_ts --onboard-lc --n-average {n_average} --psf {psf_path}"
        else:
            psf_lib_path = f"{self.inputDir}/{self.psfLibraryFilename}"
            comm = f"gen_aflux_ts --onboard-lc --n-average {n_average} --psf-library {psf_lib_path}"

        if self.pipeExtendedMask:
            comm += " --emask"
        if self.pipePlots:
            comm += " -P"
        if self.noAberrCorr:
            comm += " --ignore-aberration"
        comm += f" 1 {self.starID} {self.starID}"
        print(comm)

        # run the gen_aflux command
        cmd = os.system(comm)
        if cmd != 0:
            self.failed('gen_aflux_ts failed due to the above error!')

        # JITTER AND DRIFT CORRECTION
        if not self.pipeJitDriftOff:
            if self.verbose > 1:
                errorcode('message', '\n[apply_ltdjit_corr]: Jitter & Drift Correction')

            # build apply_ltdjit_corr command
            if self.pipePsfMethod == "microscan":
                psf_path = f"{self.microscanDirInvers}/000000001_inverse_psf.hdf5"
            else:
                psf_path = f"{self.outputDirStarIDsim}/000000001_interpolated_psf.hdf5"

            comm = f"apply_ltdjit_corr --psf {psf_path}"
            if self.pipeExtendedMask:
                comm += " --emask"
            if self.pipePlots:
                comm += " -P"
            comm += f" 1 {self.starID} {self.starID}"
            print(comm)

            # run apply_ltdjit_corr command
            cmd = os.system(comm)
            if cmd != 0:
                self.failed('apply_ltdjit_corr failed due to the above error!')

        # PROLOGUE
        # Execution time of module
        if self.verbose > 1:
            self.tocOnboard = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

    #--------------------------------------------------------------#
    #                            OUTPUTS                           #
    #--------------------------------------------------------------#

    def create_sim_table(self, odir):
        """Module to create a overview table of the simulation.
        """
        # Write PlatoSim info to a table
        filename = f'{odir}/{self.outputFileName}.table'
        data = {"ID":       self.targetNo,
                self.colID: self.df[self.colID],
                "ra":       self.df.ra,
                "dec":      self.df.dec,
                "mag":      self.df.mag,
                "group":    self.group,
                "camera":   self.camera,
                "quarter":  self.quarter,
                "ccd":      self.ccdCode,
                "xCCD":     self.xCCD,
                "yCCD":     self.yCCD,
                "rOA":      self.rOA,
                "xFP":      self.xFP,
                "yFP":      self.yFP,
                "ncon":     self.numCon,
        }
        df1 = pd.DataFrame(data, index=[0])

        # Add SPR if available
        if self.photometry:
            f = SimFile(f'{self.outputSimName}.hdf5')
            try:
                mask = f.getApertureMask(1)
            except:
                pass
            else:
                df1['SPR'] = np.mean(mask[5])
                df1['NSR'] = np.mean(mask[4])

        # Save simulation table
        df1.to_feather(filename)

        
    def failed(self, message):
        """
        Create table of failed pipeline simulations.
        """
        # Open the file in append & read mode ('a+')
        with open(self.inputDir / 'failed.txt', 'a+') as f:
            # Move read cursor to the start of file.
            f.seek(0)
            # If file is not empty then append '\n'
            if len(f.read(100)) > 0: f.write("\n")
            # Append text at the end of file
            f.write(f'{int(self.starID)} {self.group} {self.camera} {self.quarter}')
        # Stop script with errorcode
        errorcode('error', message)

        
    def sort_output_normal(self):
        """Sort output files for default setup.
        """
        # Create a info table of simulation
        if not self.fullFrame:
            self.create_sim_table(self.outputDir)

        # Remove HDF5 file for pipeline mode
        if self.postProcess and self.verbose < 3:
            os.remove(f'{self.outputSimName}.hdf5')

        # Give full read and write access to output files
        os.system(f'chmod 755 {self.outputSimName}*')

        # Compress files
        if (self.compress and os.path.isfile(f'{self.outputSimName}.ftr') or
            self.compress and os.path.isfile(f'{self.outputSimName}.hdf5')):

            if self.verbose > 1:
                errorcode('module', '\nRestructuring data output\n')
                print('Compressing files')

            os.system(f'zip -j {self.outputSimName}.zip {self.outputSimName}* ' +
                      f'{self.devnull}')

            # Give read and write access to file
            os.system(f'chmod 755 {self.outputSimName}.zip')

            # Remove non-compressed files
            if not self.postProcess:
                os.remove(f'{self.outputSimName}.hdf5')
            if not self.fullFrame:
                os.remove(f'{self.outputSimName}.table')
            if self.postProcess or self.fullFrame:
                os.remove(f'{self.outputSimName}.ftr')

        # If requested move file to final output directory (for cluster)
        if self.storageDir:
            os.system(f'mv {self.outputSimName}.* {self.storageDir}')

        # Execution time of module
        if self.verbose > 1:
            self.tocPrologue = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            
    def sort_output_pipeline(self):
        """Sort output files for pipeline setup.
        """
        if self.verbose > 1:
            errorcode('module', '\nPrologue')

        if self.verbose > 1:
            errorcode('message', '\nRestructuring data output')
            print(f'L1 light curve is saved to {self.outputDirStarIDnew}')

        # Select prefix-files
        self.outputFileName = f'{self.starID}_{self.obsPrefix}'
        prefixInversion = self.microscanDirInvers / '000000001'
        prefixStarIDsim = self.outputDirStarIDsim / self.starID
        prefixStarIDnew = self.outputDirStarIDnew / self.outputFileName
        print(f"prefixInversion {prefixInversion}")
        print(f"prefixStarIDsim {prefixStarIDsim}")
        print(f"prefixStarIDnew {prefixStarIDnew}")

        # Create a info table of simulation
        print(f"create_sim_table({self.outputDirStarIDnew})")
        self.create_sim_table(self.outputDirStarIDnew)

        # Fetch P1 light curve
        if args.sample == 'P1':
            lc_file = f"{self.outputDirStarIDsim}/LIGHTCURVE_L1A_IMAGETTE_c{self.cameraID:02d}_p000000001.hdf5"
            cob_file = f"{self.outputDirStarIDsim}/COB_OG_c{self.cameraID:02d}_p000000001.hdf5"
            skypos_file = f"{self.outputDirStarIDsim}/SKYPOS_L1A_IMAGETTE_c{self.cameraID:02d}_p000000001.hdf5"
            star_file = f"{self.outputDirStarIDsim}/000000001_target_star.hdf5"
            yaml_file = f"{self.outputDirStarIDsim}/{self.starID}.yaml"
            if self.pipePsfMethod == "microscan":
                psf_file = f"{prefixInversion}_inverse_psf.hdf5"
            else:
                psf_file = f"{self.outputDirStarIDsim}/000000001_interpolated_psf.hdf5"
            pbkg_plot = f"{self.outputDirStarIDsim}/000000001_pBKG.png"
            pcobx_plot = f"{self.outputDirStarIDsim}/000000001_pCOBx.png"
            pcoby_plot = f"{self.outputDirStarIDsim}/000000001_pCOBy.png"
            pflux_plot = f"{self.outputDirStarIDsim}/000000001_pFLUX.png"

            lc_file_out = f"{prefixStarIDnew}_LIGHTCURVE_L1A_IMAGETTE.hdf5"
            cob_file_out = f"{prefixStarIDnew}_COB_OG.hdf5"
            skypos_file_out = f"{prefixStarIDnew}_SKYPOS_L1A_IMAGETTE.hdf5"
            star_file_out = f"{prefixStarIDnew}_target_star.hdf5"
            yaml_file_out = f"{prefixStarIDnew}.yaml"
            if self.pipePsfMethod == "microscan":
                psf_file_out = f"{prefixStarIDnew}_inverse_psf.hdf5"
            else:
                psf_file_out = f"{prefixStarIDnew}_interpolated_psf.hdf5"
            pbkg_plot_out = f"{prefixStarIDnew}_pBKG.png"
            pcobx_plot_out = f"{prefixStarIDnew}_pCOBx.png"
            pcoby_plot_out = f"{prefixStarIDnew}_pCOBy.png"
            pflux_plot_out = f"{prefixStarIDnew}_pFLUX.png"

            # copy the main files to a long term area with the correct filenames
            print(f"Move {lc_file} -> {lc_file_out}")
            print(f"Move {cob_file} -> {cob_file_out}")
            print(f"Move {skypos_file} -> {skypos_file_out}")
            print(f"Move {star_file} -> {star_file_out}")
            print(f"Move {yaml_file} -> {yaml_file_out}")
            print(f"Move {psf_file} -> {psf_file_out}")
            # Move the phot files to sotrage
            try:
                shutil.copy(lc_file, lc_file_out)
                shutil.copy(cob_file, cob_file_out)
                shutil.copy(skypos_file, skypos_file_out)
                shutil.copy(star_file, star_file_out)
                shutil.copy(yaml_file, yaml_file_out)
                shutil.move(psf_file, psf_file_out)
            except:
                self.failed('Moving PSF photometry files failed...')

            if self.pipePlots:
                print(f"Move {pbkg_plot} -> {pbkg_plot_out}")
                print(f"Move {pcobx_plot} -> {pcobx_plot_out}")
                print(f"Move {pcoby_plot} -> {pcoby_plot_out}")
                print(f"Move {pflux_plot} -> {pflux_plot_out}")
                try:
                    shutil.move(pbkg_plot, pbkg_plot_out)
                    shutil.move(pcobx_plot, pcobx_plot_out)
                    shutil.move(pcoby_plot, pcoby_plot_out)
                    shutil.move(pflux_plot, pflux_plot_out)
                except:
                    self.failed('Moving PSF photometry plots failed...')

        # Fetch P5 light curve
        if args.sample == 'P5':
            if self.pipeExtendedMask:
                lc_file1 = f"{self.outputDirStarIDsim}/E-LIGHTCURVE_L0_c{self.cameraID:02d}_p000000001.hdf5"
                lc_file2 = f"{self.outputDirStarIDsim}/E-LIGHTCURVE_L1A_c{self.cameraID:02d}_p000000001.hdf5"
                cob_file = f"{self.outputDirStarIDsim}/E-COB_L0_c{self.cameraID:02d}_p000000001.hdf5"
                skypos_file = f"{self.outputDirStarIDsim}/E-SKYPOS_L1A_c{self.cameraID:02d}_p000000001.hdf5"
            else:
                lc_file1 = f"{self.outputDirStarIDsim}/LIGHTCURVE_L0_c{self.cameraID:02d}_p000000001.hdf5"
                lc_file2 = f"{self.outputDirStarIDsim}/LIGHTCURVE_L1A_c{self.cameraID:02d}_p000000001.hdf5"
                cob_file = f"{self.outputDirStarIDsim}/COB_L0_c{self.cameraID:02d}_p000000001.hdf5"
                skypos_file = f"{self.outputDirStarIDsim}/SKYPOS_L1A_c{self.cameraID:02d}_p000000001.hdf5"
            if self.pipePsfMethod == "microscan":
                psf_file = f"{prefixInversion}_inverse_psf.hdf5"
            else:
                psf_file = f"{self.outputDirStarIDsim}/000000001_interpolated_psf.hdf5"
            star_file = f"{self.outputDirStarIDsim}/000000001_target_star.hdf5"
            yaml_file = f"{self.outputDirStarIDsim}/{self.starID}.yaml"
            acobx_plot = f"{self.outputDirStarIDsim}/000000001_aCOBx.png"
            acoby_plot = f"{self.outputDirStarIDsim}/000000001_aCOBy.png"
            spr_plot = f"{self.outputDirStarIDsim}/000000001_SPR_TOT-TS.png"
            valid_plot = f"{self.outputDirStarIDsim}/000000001_Valid_points.png"
            abkg_plot = f"{self.outputDirStarIDsim}/000000001_aBKG.png"
            aflux_plot = f"{self.outputDirStarIDsim}/000000001_aFLUX.png"
            aflux_corr_plot = f"{self.outputDirStarIDsim}/000000001_aFLUX-CORR.png"

            if self.pipeExtendedMask:
                lc_file1_out = f"{prefixStarIDnew}_E-LIGHTCURVE_L0.hdf5"
                lc_file2_out = f"{prefixStarIDnew}_E-LIGHTCURVE_L1A.hdf5"
                cob_file_out = f"{prefixStarIDnew}_E-COB_L0.hdf5"
                skypos_file_out = f"{prefixStarIDnew}_E-SKYPOS_L1A.hdf5"
            else:
                lc_file1_out = f"{prefixStarIDnew}_LIGHTCURVE_L0.hdf5"
                lc_file2_out = f"{prefixStarIDnew}_LIGHTCURVE_L1A.hdf5"
                cob_file_out = f"{prefixStarIDnew}_COB_L0.hdf5"
                skypos_file_out = f"{prefixStarIDnew}_SKYPOS_L1A.hdf5"
            if self.pipePsfMethod == "microscan":
                psf_file_out = f"{prefixStarIDnew}_inverse_psf.hdf5"
            else:
                psf_file_out = f"{prefixStarIDnew}_interpolated_psf.hdf5"
            star_file_out = f"{prefixStarIDnew}_target_star.hdf5"
            yaml_file_out = f"{prefixStarIDnew}.yaml"
            acobx_plot_out = f"{prefixStarIDnew}_aCOBx.png"
            acoby_plot_out = f"{prefixStarIDnew}_aCOBy.png"
            spr_plot_out = f"{prefixStarIDnew}_SPR_TOT-TS.png"
            valid_plot_out = f"{prefixStarIDnew}_Valid_points.png"
            abkg_plot_out = f"{prefixStarIDnew}_aBKG.png"
            aflux_plot_out = f"{prefixStarIDnew}_aFLUX.png"
            aflux_corr_plot_out = f"{prefixStarIDnew}_aFLUX-CORR.png"

            # copy the main files to a long term area with the correct filenames
            print(f"Move {lc_file1} -> {lc_file1_out}")
            print(f"Move {lc_file2} -> {lc_file2_out}")
            print(f"Move {cob_file} -> {cob_file_out}")
            print(f"Move {skypos_file} -> {skypos_file_out}")
            print(f"Move {psf_file} -> {psf_file_out}")
            print(f"Move {star_file} -> {star_file_out}")
            print(f"Move {yaml_file} -> {yaml_file_out}")
            try:
                shutil.copy(lc_file1, lc_file1_out)
                shutil.copy(lc_file2, lc_file2_out)
                shutil.copy(cob_file, cob_file_out)
                shutil.copy(skypos_file, skypos_file_out)
                shutil.copy(psf_file, psf_file_out)
                shutil.copy(star_file, star_file_out)
                shutil.copy(yaml_file, yaml_file_out)
            except:
                self.failed('Moving aperture photometry files failed...')

            if self.pipePlots:
                print(f"Move {acobx_plot} -> {acobx_plot_out}")
                print(f"Move {acoby_plot} -> {acoby_plot_out}")
                print(f"Move {spr_plot} -> {spr_plot_out}")
                print(f"Move {valid_plot} -> {valid_plot_out}")
                print(f"Move {abkg_plot} -> {abkg_plot_out}")
                print(f"Move {aflux_plot} -> {aflux_plot_out}")
                print(f"Move {aflux_corr_plot} -> {aflux_corr_plot_out}")
                try:
                    shutil.copy(acobx_plot, acobx_plot_out)
                    shutil.copy(acoby_plot, acoby_plot_out)
                    shutil.copy(spr_plot, spr_plot_out)
                    shutil.copy(valid_plot, valid_plot_out)
                    shutil.copy(abkg_plot, abkg_plot_out)
                    shutil.copy(aflux_plot, aflux_plot_out)
                    shutil.copy(aflux_corr_plot, aflux_corr_plot_out)
                except:
                    self.failed('Moving aperture photometry plots failed...')

        # Remove microscan-starID and simulation folder (and all its content)
        if self.verbose < 3:
            print(f"Removing {self.microscanDirStarID} {self.microscanDirInvers} {self.outputDirStarIDsim}")
            shutil.rmtree(self.microscanDirStarID)
            shutil.rmtree(self.microscanDirInvers)
            shutil.rmtree(self.outputDirStarIDsim)

        # Give full read/write access
        print(f"chmod 755 {prefixStarIDnew}*")
        os.system(f'chmod 755 {prefixStarIDnew}*')

        # Compress files
        if self.compress:
            print("Compressing...")
            if self.verbose > 1:
                print('Compressing files')
            comm1 = f'zip -j {prefixStarIDnew}.zip {prefixStarIDnew}* {self.devnull}'
            comm2 = f"find {self.outputDirStarIDnew} -type f -not -name '*.zip' -delete {self.devnull}"
            comm3 = f'chmod 755 {prefixStarIDnew}.zip'
            print(comm1)
            os.system(comm1)
            print(comm2)
            os.system(comm2)
            print(comm3)
            os.system(comm3)

        # If requested move file to final output directory (for cluster)
        if self.storageDir:
            comm4 = f'mv {prefixStarIDnew}.* {self.storageDir}'
            print(comm4)
            os.system(comm4)

        # Execution time of module
        if self.verbose > 1:
            self.tocPrologue = datetime.datetime.now() - self.tic
            self.tic = datetime.datetime.now()

            
    def resources(self):
        """Module to print resources used by PLATOnium.
        """
        if self.verbose > 1:
            errorcode('message', '\nSimulation statistics')
            print('------------------------------------------------------------')
            print(f'Max RAM memory for PlatoSim      : {self.memRamPlatoSim} MB')
            print(f'Storage memory for PlatoSim      : {self.memDiskPlatoSim} MB')
            print(f'Execution time for PlatoSim      : {self.tocPlatoSim} [h:mm:ss]')
            if self.detrend:
                print(f'Execution time for detrending    : {self.tocDetrend} [h:mm:ss]')
            if self.stitchActive:
                print(f'Execution time for stitching     : {self.tocStitch} [h:mm:ss]')
            if self.clipWotan:
                print(f'Execution time for Wotan clip    : {self.tocWotanClip} [h:mm:ss]')
            if self.pipeline:
                if self.pipePsfMethod == "microscan":
                    print(f'Execution time for Microscanning : {self.tocMicroscan} [h:mm:ss]')
                    print(f'Execution time for PSF inversion : {self.tocInversion} [h:mm:ss]')
                if self.sample == 'P1':
                    print(f'Execution time for L1 On-ground  : {self.tocOnground} [h:mm:ss]')
                if self.sample == 'P5':
                    print(f'Execution time for L1 On-board   : {self.tocOnboard} [h:mm:ss]')
                print(f'Execution time for Prologue      : {self.tocPrologue} [h:mm:ss]')
            print('------------------------------------------------------------')
            print(f'Total execution time             : {datetime.datetime.now() - self.tic0} [hh:mm:ss]\n')

#==============================================================#
#               PARSING COMMAND-LINE ARGUMENTS                 #
#==============================================================#

parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-p', '--plot',      action='store_true',  help='Flag to plot 1st subfield (this will not save the simulation)')
parser.add_argument('-w', '--overwrite', action='store_true',  help='Flag to overwrite an existing HDF5 output file that may exist')
parser.add_argument('-a', '--animation', action='store_true',  help='Flag to generate an animation of the subfield simulation')
parser.add_argument('-v', '--verbose', metavar='NR', type=int, help='Verbosity level [0, 1, 3] (Default: 1)')

man_group = parser.add_argument_group('MANDATORY PARAMETERS')
man_group.add_argument('starID',   type=int, help='Star ID in target list (or CCD in [1, 2, 3, 4] using --fullframe)')
man_group.add_argument('groupID',  type=int, help='Camera group ID [1, 2, 3, 4, 5] (F-CAM = 5)')
man_group.add_argument('cameraID', type=int, help='N-CAM in [1, 2, 3, 4, 5, 6]; F-CAM in {1, 2}')
man_group.add_argument('quarter',  type=int, help='Mission quarter in [1, 2, 3, 4, ..]')

rec_group = parser.add_argument_group('RECOMMENTED PARAMETERS')
rec_group.add_argument('--seed',        metavar='INT',  type=int, help='Option to bootstrap seeds ro reproduce results')
rec_group.add_argument('--sample',      metavar='NAME', type=str, help='Option to select a PIC sample catalogue to use')
rec_group.add_argument('--performance', metavar='MODE', type=str, help='Option to set basic input parameters {required, expected}')

out_group = parser.add_argument_group('I/O PARAMETERS')
out_group.add_argument('-i', '--ifil', metavar='FILE', type=str, help='Path to YAML input file')
out_group.add_argument('-o', '--odir', metavar='PATH', type=str, help='Path to output directory')
out_group.add_argument('-s', '--sdir', metavar='PATH', type=str, help='Path to final output directory for storage')
out_group.add_argument('--project',    metavar='NAME', type=str, help='Name of project folder within $PLATO_WORKDIR')
out_group.add_argument('--yaml',       metavar='NAME', type=str, help='Name of yaml file within $PLATO_WORKDIR/input')
out_group.add_argument('--prefix',     metavar='NAME', type=str, help='Prefix added to output filename')
out_group.add_argument('--starcat',    metavar='FILE', type=str, help='Path to stellar catalog file -> see PlatoSim docs')
out_group.add_argument('--varfile',    metavar='FILE', type=str, help='Path to variable source file -> see PlatoSim docs')
out_group.add_argument('--varlist',    metavar='FILE', type=str, help='Path to variable source list -> see PlatoSim docs')
out_group.add_argument('--compress',   action='store_true',      help='Flag to compress output files')
out_group.add_argument('--saveplot',   action='store_true',      help='Flag to save any plot produced')

sim_group = parser.add_argument_group('SIM PARAMETERS')
sim_group.add_argument('--field',         metavar='LOP',  type=str,   help='PLATO pointing field [tLOPS2, LOPS2, LOPN1, SPF, NPF]')
sim_group.add_argument('--cadence',       metavar='SEC',  type=float, help='Cadence for each exposure (default: 25 seconds)')
sim_group.add_argument('--tdur',          metavar='DAY',  type=float, help='Duration of shortened quarter time series [days]')
sim_group.add_argument('--bdur',          metavar='DAY',  type=float, help='Duration of time to start qurter simulation [days]')
sim_group.add_argument('--nexp',          metavar='NO.',  type=int,   help='Number of exposures of shortened quarter time series')
sim_group.add_argument('--bexp',          metavar='NO.',  type=int,   help='Number of exposure to start from beginning of quarter')
sim_group.add_argument('--pic',           metavar='ID',   type=int,   help='Option to overwrite starID and select PIC identifier')
sim_group.add_argument('--mag',           metavar='MAG',  type=float, help='Option to overwrite target magnitude in inputfile')
sim_group.add_argument('--con_dmag',      metavar='MAG',  type=float, help='Threshold in dmag of contaminant(s) (Default: 10 mag)')
sim_group.add_argument('--con_dist',      metavar='AS',   type=float, help='Threshold in dist of contaminant(s) (Default: 60 as)')
sim_group.add_argument('--con_none',      action='store_true',        help='Flag to ignore all contaminants in subfield')
sim_group.add_argument('--no_aberr_corr', action='store_true',        help='Flag to turn of aberration correction (no DKA)')
sim_group.add_argument('--jit_reuse',     action='store_true',        help='Flag to reuse an AOCS jitter file across all quarters')

phot_group = parser.add_argument_group('PHOTOMETRY PARAMETERS')
phot_group.add_argument('--mask',        metavar='DAY',  type=float,  help='Option to overwrite the mask-update in inputfile [days]')
phot_group.add_argument('--detrend',     metavar='NAME', type=str,    help='Detrending method [poly, lowess, poly_lowess, lowess_theil, wotan]')
phot_group.add_argument('--poly_deg',    metavar='INT',  type=int,    help='Degree of polynomial of trend (use with --detrend poly)')
phot_group.add_argument('--stitch',      metavar='NAME', type=str,    help='Name of stitching method to activate [lowess, median]')
phot_group.add_argument('--clip',        action='store_true',         help='Flag to activate outlier rejection')
phot_group.add_argument('--clip_sigma',  metavar='FLOAT', type=float, help='MAD std to use for outlier rejection (Default: 4)')
phot_group.add_argument('--clip_window', metavar='FLOAT', type=float, help='Window to use for outlier rejection (Default: 0.5 d)')
phot_group.add_argument('--check',       action='store_true',         help='Flag to plot the requested post-processing steps')

pip_group = parser.add_argument_group('PIPELINE PARAMETERS')
pip_group.add_argument('--pipeline',      action='store_true',             help='Flag to activate proto-type pipeline')
pip_group.add_argument('--pipe_psf',      metavar='NAME',      type=str,   help='Pipeline PSF method [microscan | library]')
pip_group.add_argument('--pipe_cadence',  metavar='INT',       type=int,   help='Cadence for pipeline (P1=25s, P5=50 or 600s)')
pip_group.add_argument('--pipe_flux_err', metavar='PERCENT',   type=float, help='Error assumption of target and contaminant(s) flux (Default: 1 %%)')
pip_group.add_argument('--pipe_cen_err',  metavar='PIXEL',     type=float, help='Error assumption of target centroid (Default: 0.03 pixel)')
pip_group.add_argument('--pipe_prnu_err', metavar='PERCENT',   type=float, help='Error assumption of PRNU knowledge (Default: 0.1 %%)')
pip_group.add_argument('--pipe_jit_off',  action='store_true',             help='Flag to turn-off the jitter/drift correction in apply_ltdcorr')
pip_group.add_argument('--pipe_emask',    action='store_true',             help='Flag to turn-on the extended flux mask in gen_aflux_ts')
pip_group.add_argument('--pipe_plots',    action='store_true',             help='Enable pipeline output plots')

full_group = parser.add_argument_group('FULL-FRAME CCD PARAMETERS')
full_group.add_argument('--fullframe', action='store_true', help='Flag to simulate a full-frame CCD -> CCDcode = starID')

args = parser.parse_args()

# Load and run modules
p = PLATOnium(args)
p.configure_output()
p.load_stars()
sim = p.init_sim()
p.create_seeds(sim)

# skip if only doing L1
if not p.l1_only:
    p.create_inputfiles(sim)

if args.plot:
    # Only show subfield
    p.show_subfield(sim)

elif args.pipeline and args.sample == 'P1':
    # Run on-ground L0-L1 pipeline chain
    p.control_hdf5()
    if not p.l1_only:
        p.run_sim_normal(sim)
        if p.pipePsfMethod == "microscan":
            p.run_microscan(sim)
            p.run_L1_microscan()
    p.run_L1_onground()
    p.sort_output_pipeline()

elif args.pipeline and args.sample == 'P5':
    # Run on-board L0-L1 pipeline chain
    p.control_hdf5()
    if not p.l1_only:
        p.run_sim_normal(sim)
        if p.pipePsfMethod == "microscan":
            p.run_microscan(sim)
            p.run_L1_microscan()
    p.run_L1_onboard()
    p.sort_output_pipeline()

else:
    # Run PlatoSim time series
    p.run_sim_normal(sim)
    # Run simple post-processing
    if args.detrend or args.stitch or args.clip:
        p.run_reduction(sim)
    # Prologue
    p.sort_output_normal()

# Finito!
if not args.plot:
    p.resources()

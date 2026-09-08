#!/usr/bin/env python3
"""
This script is an integrated part of PlatoSim's toolkit PLATOnium.
The usage consist of three main query methods:
  1) Create a customized PIC catalogue
  2) Create a single-star catalogue quering Simbad
  3) Create a PLATO pointing field catalogue for each camera FOV

General information:
  - One of the three query methods needs to be used to run this script.
  - All output files are directly useable by PALTOnium.
  - The arguments "--dmag" and "--dist" only apply to method 1 and 2.

1) Customized PIC catalogue (--pic):
  This option uses the PIC input catalog to create a smaller custimized
  stellar catalogue both for the PIC targets and contaminants. Moreover,
  this script can also be used to re-plot a old "--pic" catalog produced
  by this software. Usage example:

  $ picsim --pic 100 P1 LOPS2 --project <project_name> -p

2) Single-star cataloge (--simbad):
  This option can be used to feth a single target star by name, which
  is recognizable by the CDS Simbad query. By default is select stars
  within 45 arcsec (2 pixel) from the target star: Usage examples:

  $ picsim --simbad Mizar --project <project_name> -p

3) PLATO pointing field catalogue (--vizier):
  This option creates a coherent PLATO catalogue that covers the
  FoV of each camera group. It uses the Gaia DR3 catalogue and
  makes query of 16 grid points distribution around the requested
  PLATO pointing field. The output catalogues (one for each group)
  is recognized be by "platonium --fullframe" for the generation
  of full-frame CCD images. Usage examples:

  $ picsim --vizier LOPS2 --project <project_name> -p
"""
# Built-in
import os
import shutil
import random
import argparse
import datetime
import warnings
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from prettytable import PrettyTable

# Platonium extra
import ligo.skymap.plot

# PlatoSim functions
import platosim.plot      as pt
import platosim.utilities as ut
import platosim.starquery as sq
import platosim.referenceFrames as rf
from platosim.utilities  import errorcode
from platosim.simulation import Simulation
from platosim.matplotlibrc import setup; setup()         

# Turn of specific Pandas warning
pd.options.mode.chained_assignment = None

#==============================================================#
#                         BEGIN CLASS                          #
#==============================================================#

class PicSim(object):
    """Class to generate customised (PIC) catalogues.
    """ 
    def __init__(self, args):

        # I/O PARAMETERS

        # Plotting flag
        self.plot = args.plot
        
        # Verbosity (a.k.a log level) -> Identical to PlatoSim usage
        if args.verbose == 0:
            self.verbose = 0            
            warnings.filterwarnings("ignore")
        elif args.verbose is None:
            self.verbose = 2
        else:
            self.verbose = args.verbose

        # Output directory
        if (args.outdir is not None) or (args.project is not None):

            # NOTE "--outdir" overwrites "--project"
            if args.outdir is not None:
                self.outputDir = Path(args.outdir).resolve()
            elif args.project is not None:
                self.outputDir = Path(os.getenv('PLATO_WORKDIR')) / args.project / 'input'

            # Create directory if is doesn't exist
            self.outputDir.mkdir(parents=True, exist_ok=True)
        else:
            self.outputDir = args.outdir

        # GENERIC INPUT PARAMETERS
        
        # Contaminant magnitude limit
        if args.dmag is None:
            self.dmagConLimit = 5
        else:
            self.dmagConLimit = args.dmag

        # Warning for single star query
        if args.simbad and self.dmagConLimit > 21:
            errorcode('warning', 'Gaia DR3 is only complete to G < 21 mag!')

        # Contaminant distance limit (default 2 pixel)
        if args.dist is None:
            self.disConLimit = 45
        else:
            if args.pic is not None:
                if ((args.pic[0][2] in ['LOPS2', 'LOPN1']) and
                    (args.dist not in [30, 45, 60])):
                    errorcode('error', 'Not a valid contaminant-to-target distance! ' +
                              'Use {30, 45, 60} arcsec')
                elif (args.pic[0][2] in ['tLOPS2']) and (args.dist > 60):
                    errorcode('error', 'Maximum contaminant-to-target distance exceeded! ' +
                              'Use dist <= 60 arcsec')
            # For Simbad any value is accepted
            self.disConLimit = args.dist
        
    #--------------------------------------------------------------#
    #                        PIC OF DESTINY                        #
    #--------------------------------------------------------------#            

    def printNotesPIC(self):
        """Function to show the notes about the PIC.
        """
        errorcode('message', '\nOverview notes for the PLATO Input Catalogue (PIC)')
        print("""
Notes on parsed argument "--pic":
  "star"   : Number of targets saved in catalog (Select "all" for all stars)
  "sample" : PIC samples [P1, P2, P4, P5, all]  (Select "all" for all samples)
  "field"  : PLATO pointing field [LOPS2, LOPN1]
       
Information about PIC 2.1.0:
- Number of stars in fields:
  --------------------------
  LOPS2      : 315,385
    - tPIC   : 218,820
      - P1   :  10,967
      - P2   :     690
      - P4   :  12,738
      - P5   : 157,723
    - fgPIC  :   5,283
    - cPIC   :  53,321
    - scvPIC :  37,961
  --------------------------
- Notes about the catalogue:
  - Magnitude completeness of contaminants    : Pmag < 19 mag
  - Maximum radial target-contaminat distance : Rmax < 60 arcsec
  - Calibration methods: Same as PIC 200.

Information about PIC 2.0.0:
- Number of stars in fields:
  --------------------------
  LOPS2    : 179,564
    - P1   :   8,835
    - P2   :     678
    - P4   :  12,026
    - P5   : 157,543
  --------------------------
  LOPN1    : 179,564
    - P1   :   9,369
    - P2   :     680
    - P4   :  12,026
    - P5   : 152,818
  --------------------------
- Notes about the catalogue:
  - Magnitude completeness of contaminants    : Pmag < 17 mag
  - Maximum radial target-contaminat distance : Rmax < 45 arcsec
  - Calibration methods: The Gaia DR3 colour information (BP-RP)
    and extinction maps are used to derive the PLATO magnitudes.

General notes:
  - The P2 is a bright sub-sample of P1, and P1 of P5.
        """)

        
    def initPIC(self):
        """Initialise the PIC input parameters.
        """
        if self.verbose > 1:
            errorcode('software', '\nPIC of Destiny')

        # Optinal parameters without checks
        self.saveAscii     = args.save
        self.inputFiles    = args.incat
        self.oldCatalogue  = args.unique
        self.numTimeseries = args.ntime
            
        # MANDATORY PARAMETERS
            
        # PIC arguments
        self.stars, self.sample, self.field = args.pic[0]

        # Check field
        field = ['LOPS2', 'LOPN1']
        if self.field == 'LOPN1' and args.release in [None, 'PIC210']:
            errorcode('error', f'{self.field} is not in PIC210 yet! Use "--release PIC200"')
        if self.field not in field:
            errorcode('error', f'{self.field} is not defined! Use {field}')
        
        # Select PIC release (default is the latest version)
        if args.release in [None, 'PIC210']:
            self.pic = 'PIC210'
            self.mag_column = 'Pmag'
            # Select the field
            if self.field == 'LOPS2':
                self.numPIC = 218162
            else:
                errorcode('error', f'{self.field} is not defined! Use LOPS2')
            # Select the sample
        elif args.release == 'PIC200':
            self.pic = 'PIC200'
            self.mag_column = 'mag'
            
            # Select the field
            if self.field == 'LOPS2':
                self.numPIC = 179564
            elif self.field == 'LOPN1':
                self.numPIC = 175325 # Original: 175,597
            else:
                errorcode('error', f'{self.field} is not defined! Use {LOPS2, LOPN1}')
        # TODO The PIC110 is not used anymore, but change to PLATO-CS (PCS100)
        elif args.release == 'PIC110':
            self.pic = 'PIC110'
            self.mag_column = 'mag'
            sample = ['P1', 'P1', 'P4', 'P5']
            # Select the field
            if self.field == 'SPF':
                self.numPIC = 137052
            elif self.field == 'NPF':
                self.numPIC = 144507
            else:
                errorcode('error', f'{self.field} is not defined! Use {SPF, NPF}')
        else:
            errorcode('error', f'No release named {args.release}! Usage in [PIC210, PIC200]')
                
        # Available samples for release
        if self.pic == 'PIC210':
            sample = [
                'tPIC', 'P1', 'P2', 'P4', 'P5',
                'fgPIC', 'fgFb', 'fgFr',
                'cPIC', 'R1F', 'R2F', 'R3F', 'R4F', 'R5F', 'R1N', 'R2N', 'R3N', 'R4N', 'R5N',
                'scvPIC', 'SCV1a', 'SCV1b', 'SCV1c', 'SCV1d', 'SCV1e',
                'SCV2a', 'SCV2b', 'SCV3a', 'SCV3b', 'SCV4a', 'SCV4b', 'SCV5', 'SCV6'
            ]
        else:
            sample = ['P1', 'P1', 'P4', 'P5']
        # Check the sample
        if self.sample == 'all':
            self.sample = None
        elif self.sample not in sample:
            errorcode('error', f'Not a valid PIC sample! Use {sample}')
    
        # Check number of stars
        self.numTargets = self.stars
        if self.numTargets[0].isdigit():
            self.numTargets = int(self.numTargets)
            if self.numTargets > self.numPIC:
                errorcode('error', 'More stars selected than available in the PIC! ' +
                          'See -h for PIC notes')
        elif self.stars == 'all':
            if self.numTimeseries is not None:
                self.numTargets = int(self.numTimeseries/6.) + 1
            else:
                self.numTargets = self.numPIC

        # OBSERVATIONAL PARAMETERS
        
        # Select magnitude range or select all stars
        if args.mag is None:
            self.magRange = [0, 21]
        else:
            self.magRange = ut.convertMagnitudeRange(args.mag)
        
        # Visibility by N-cams
        if args.ncams in [None, 6, 12, 18, 24]:
            self.ncams = args.ncams
        else:
            errorcode('Not valid N-CAM visibility! Use {6, 12, 18, 24}')

        # Check cam-group entry
        if args.group in [None, 1, 2, 3, 4]:
            self.group = args.group
        else:
            errorcode('error', 'Not valid group number! Use {1, 2, 3, 4}')

        # Check evolutionary stage 
        if args.lum in [None, 'V', 'IV']:
            self.lumClass = args.lum
        else:
            errorcode('error', 'Not valid evolutionary stage! Use {V, IV}')

        # Check spectral type
        if args.spec in [None, 'F', 'G', 'K']:
            self.specType = args.spec
        else:
            errorcode('error', 'Not valid spectral type! Use {F, G, K}')
                                
        # EXTRA I/O PARAMETERS
        
        # Input directory        
        self.inputDir = Path(os.getenv("PLATO_PROJECT_HOME")) / 'inputfiles/data_picsim'
        if not self.inputDir.is_dir():
            errorcode('message', '\nInuaguration: Welcome to PIC of Destiny!')

        # Name space for output files
        # Default is to use the latest PIC catalog saved
        # Else several old catalogs can be parsed for replotting
        # E.g. use: --intar starcat**targets.txt --incon starcat**contaminants.txt
        if self.outputDir:

            # Prefix name of output files
            self.outputPrefix = 'starcat'
            if self.sample: self.outputPrefix += '_'      + self.sample
            if self.field:  self.outputPrefix += '_'      + self.field
            if self.group:  self.outputPrefix += '_Group' + str(self.group)
            if self.ncams:  self.outputPrefix += '_Ncam'  + f'{self.ncams:0=2d}'
            if self.inputFiles is not None: self.outputPrefix += '_NewCat'
            self.outputPrefixTar = self.outputPrefix + '_targets.ftr'
            self.outputPrefixCon = self.outputPrefix + '_contaminants.ftr'

            # Save either a PlatoSim (ascii) or a PLATOnium (feather) calogue
            if self.saveAscii:
                self.outputFileCat = self.outputDir / f'{self.outputPrefix}.txt'
            else:
                self.outputFileTar = self.outputDir / self.outputPrefixTar
                self.outputFileCon = self.outputDir / self.outputPrefixCon
                
        # PLOT PARAMETERS
            
        # Generic title to use for plotting
        self.title = f'{self.pic}, {self.field}, {self.sample} sample'
         
               
    def loadPIC(self):
        """Fetch PIC targets from feather files.
        """
        # Files names
        inputFileTar = self.inputDir / f'{self.pic}_{self.field}_targets.ftr'
        inputFileCon = self.inputDir / f'{self.pic}_{self.field}_contaminants.ftr'

        # Fetch PIC catalogue from FTP server
        if not inputFileTar.is_file() or not inputFileCon.is_file():
            print(f'Downloading {self.pic} catalogue..')
            ut.downloadFromFTP(inputFileTar.name, self.inputDir, 'plato')
            ut.downloadFromFTP(inputFileCon.name, self.inputDir, 'plato')

        # Load catalogues
        if self.verbose > 1:
            print('Loading stellar catalogues..')
        self.df0 = pd.read_feather(inputFileTar)
        self.dc0 = pd.read_feather(inputFileCon)
        self.dx = self.df0
    
        
    def loadOldPIC(self):
        """Load old PIC catalog to re-plot it.
        """
        # Check if one or more catalogs are parsed
        # NOTE using *.ftr will order in alphabetic order
        if len(self.inputFiles) > 2:
            inputFileTar = self.inputFiles[1::2]
            inputFileCon = self.inputFiles[0::2]
        else:
            inputFileTar = [self.inputFiles[1]]
            inputFileCon = [self.inputFiles[0]]
            
        # Append each catalog if several are parsed
        N = int(len(self.inputFiles)/2.)
        for i in range(N):
            if i == 0:
                self.df = pd.read_feather(inputFileTar[i])
                self.dc = pd.read_feather(inputFileCon[i])
            else:
                self.df = pd.concat([self.df, pd.read_feather(inputFileTar[i])])
                self.dc = pd.concat([self.dc, pd.read_feather(inputFileCon[i])])

        # For --incat and sample distribution
        self.dx  = self.df0
        self.df0 = self.df

                
    def getStellarClass(self, df):
        """Classifier of spectral type and luminosity class.
        """
        # Seperate dwarf (MS) and sub-gaint (post MS) stars
        ds = df[df.R < ut.getMainSequenceLimit(df.Teff)]
        sg = df[df.R > ut.getMainSequenceLimit(df.Teff)]
        
        # Seperate dwarf spectral types
        dK = ds[ ds.Teff < 5300]
        dG = ds[(ds.Teff > 5300) & (ds.Teff < 5900)]
        dF = ds[ ds.Teff > 5900]
        
        # Seperate sub-giants spectral types
        sgK = sg[ sg.Teff < 5300]
        sgG = sg[(sg.Teff > 5300) & (sg.Teff < 5900)]
        sgF = sg[ sg.Teff > 5900]
        
        return df, ds, sg, dK, dG, dF, sgK, sgG, sgF


    def queryTargetsPIC(self):
        """Fetch PIC targets from feather files.
        """
        if self.verbose > 1:
            errorcode('module', '\nPIC targets')
        
        # Check parsing of old catalogue to select new unique targets
        if self.oldCatalogue:
            try:
                df_old = pd.read_feather(self.oldCatalogue)
            except:
                errorcode('error', 'Old picsim target catalogue do not exist!')
            else:
                cond = df['PIC'].isin(df_old['PIC']) # Boolen
                df   = df.drop(df[cond].index)
        else:
            df = self.df0

        # QUERY CUTS

        # Check sample flag
        if self.pic == 'PIC210':
            # tPIC samples
            if self.sample == 'tPIC':
                df = df.loc[df.source & 4 == 4]
            elif self.sample == 'P1':
                df = df[df.tPIC == 1]
            elif self.sample == 'P2':
                df = df[df.tPIC == 3]
            elif self.sample == 'P4':
                df = df[df.tPIC == 8]
            elif self.sample == 'P5':
                df = df[df.tPIC == 4]
            # fgPIC samples        
            elif self.sample == 'fgPIC':
                df = df.loc[df.source & 8 == 8]
            elif self.sample == 'fgFb':
                df = df[df.fgPIC & 1 == 1]
            elif self.sample == 'fgFr':
                df = df[df.fgPIC & 2 == 2]
            # cPIC samples
            elif self.sample == 'cPIC':
                df = df.loc[df.source & 16 == 16]
            elif self.sample == 'R1F':
                df = df[df.cPIC & 1 == 1]
            elif self.sample == 'R2F':
                df = df[df.cPIC & 2 == 2]
            elif self.sample == 'R3F':
                df = df[df.cPIC & 4 == 4]
            elif self.sample == 'R4F':
                df = df[df.cPIC & 8 == 8]
            elif self.sample == 'R5F':
                df = df[df.cPIC & 16 == 16]
            elif self.sample == 'R1N':
                df = df[df.cPIC & 32 == 32]
            elif self.sample == 'R2N':
                df = df[df.cPIC & 64 == 64]
            elif self.sample == 'R3N':
                df = df[df.cPIC & 128 == 128]
            elif self.sample == 'R4N':
                df = df[df.cPIC & 256 == 256]
            elif self.sample == 'R5N':
                df = df[df.cPIC & 512 == 512]
            # scvPIC samples
            elif self.sample == 'scvPIC':
                df = df.loc[df.source & 32 == 32]
            elif self.sample == 'SCV1a':
                df = df[df.scvPIC & 1 == 1]
            elif self.sample == 'SCV1b':
                df = df[df.scvPIC & 2 == 2]
            elif self.sample == 'SCV1c':
                df = df[df.scvPIC & 4 == 4]
            elif self.sample == 'SCV1d':
                df = df[df.scvPIC & 8 == 8]
            elif self.sample == 'SCV1e':
                df = df[df.scvPIC & 16 == 16]
            elif self.sample == 'SCV2a':
                df = df[df.scvPIC & 32 == 32]
            elif self.sample == 'SCV2b':
                df = df[df.scvPIC & 64 == 64]
            elif self.sample == 'SCV3a':
                df = df[df.scvPIC & 128 == 128]
            elif self.sample == 'SCV3b':
                df = df[df.scvPIC & 256 == 256]
            elif self.sample == 'SCV4a':
                df = df[df.scvPIC & 512 == 512]                    
            elif self.sample == 'SCV4b':
                df = df[df.scvPIC & 1024 == 1024]                    
            elif self.sample == 'SCV5':
                df = df[df.scvPIC & 2048 == 2048]                    
            elif self.sample == 'SCV6':
                df = df[df.scvPIC & 4096 == 4096]                    
        # Check stellar sample
        else: 
            df = df[df['sample'] == self.sample]

        # After cur store for plot
        self.dx = df
        
        # Fetch stellar classifications
        df, ds, sg, dK, dG, dF, sgK, sgG, sgF = self.getStellarClass(df)
                
        # Check evolutionary stage 
        if self.lumClass == 'V':  df = ds
        if self.lumClass == 'IV': df = sg

        # Check spectral type
        if self.specType is not None:
            if self.specType == 'K':
                df = dK if self.lumClass == 'V' else (sgK if self.lumClass == 'IV' else pd.concat([dK, sgK]))
            elif self.specType == 'G':
                df = dG if self.lumClass == 'V' else (sgG if self.lumClass == 'IV' else pd.concat([dG, sgG]))
            elif self.specType == 'F':
                df = dF if self.lumClass == 'V' else (sgF if self.lumClass == 'IV' else pd.concat([dF, sgF]))

        # Select stars depending on camera visibility
        # NOTE this will be done properly later but here it is done to catch
        #      the failure when the stars requested are larger than catalogue
        if self.ncams is not None:
            df = df[df['ncams'] == int(self.ncams)]

        # Allow to select only one cam-group
        if self.group:
            sim = Simulation('picsim')
            # Fetch PLATO pointing field [deg]
            alpha, delta, kappa = ut.getPointingField(self.field, unit='deg')
            dex = sim.getStarsWithinCameraGroup(df.ra.to_numpy(), df.dec.to_numpy(),
                                                alpha, delta, kappa,
                                                self.group)
            # Select max number of stars if too many is requested
            max_stars = np.sum(dex[0])
            if self.numTargets > max_stars:
                self.numTargets = max_stars
                if self.verbose > 0:
                    errorcode('warning', f'Only {max_stars} stars are available after cuts!')
            df = df[dex[0]]

        # Check P passband magnitude range
        if self.magRange is not None:
            df = df[(df[self.mag_column] > self.magRange[0]) & (df[self.mag_column] < self.magRange[1])]

        # Check if too many stars are selected
        if len(df) < self.numTargets:
            if (self.stars != 'all'):
                if self.verbose > 0:
                    errorcode('warning', f'Only {len(df)} stars are available after cuts!')
            self.numTargets = len(df)
            
        # RANDOM CATALOGUE SELECTION

        # Only after the above cuts can we select randomly a sub-sample (if applicable)
        # Use random.choices(data, weights=some weight, k=numTargets) if a weigthing is needed
        # NOTE args.stars being a string is needed to select "all" in one of the samples too
        # NOTE we need to shuffle the sample if a specific number of timeseries are needed
        if (self.stars != 'all') or (self.numTimeseries is not None):
            df = df.sample(n = self.numTargets)

        # Redefine the number fo targets
        numTargets = len(df['PIC'])

        # Add ID column to df
        df  = df.reset_index()
        df  = df.drop(columns=['index'])
        ids = np.arange(1, len(df.ra)+1)
        IDs = [f'{i}'.zfill(9) for i in ids]
        df  = ut.pdAddColumn(df, IDs, 'ID')

        # Randomly choose sample until the number of timeseries are met
        if self.numTimeseries:
            numImagettes = 0
            for i in range(numTargets):
                # Add camera visibilities
                numImagettes += df['ncams'].to_numpy()[i]
                # The data it cut by the index that reached the limit
                if numImagettes >= self.numTimeseries:
                    print(i)
                    df = df.drop(df.index[:i])
                    break

        # MAKE AN OVERVIEW TABLE

        cameras = [6, 12, 18, 24]
        countStars = [df[df['ncams'] == 6].count()[0],  df[df['ncams'] == 12].count()[0],
                      df[df['ncams'] == 18].count()[0], df[df['ncams'] == 24].count()[0]]
        countImagettes = [countStars[i] * cameras[i] for i in range(4)]

        # Last collection of stars
        numTargets = np.sum(countStars)

        # Make table
        self.t = PrettyTable(['N-cams', 'Targets', 'Time series'])
        for i in range(4):
            self.t.add_row([str(cameras[i]), countStars[i], countImagettes[i]])
        if self.verbose > 1:
            print(f'User catalog contains {numTargets} stars of ' +
                  f'{np.sum(countImagettes)} time series drawn from:')
            print(self.t)

        # Store copy of modified df
        self.df = df

        
    def queryContaminantsPIC(self):        
        """Fetch and select PIC contaminants.

        Function to select stellar contaminants belonging to specific 
        PIC target stars within a certain radial distance. Note that
        there are more than 8 million contaminants stars in the original
        catalog, hence, this step can take some time if the large
        catalog are to be used.
        """        
        if self.verbose > 1:
            errorcode('module', '\nPIC contaminants')
            print(f'Fetching contaminants within {self.disConLimit} arcsec ' +
                  f'and {self.dmagConLimit} mag from each target:')

        # Store data frames
        df = self.df
        dc = self.dc0

        # Create empty data-frame to append to 
        dc1 = pd.DataFrame()
        numConPerTar = np.zeros(self.numTargets)

        for starNo in tqdm(range(self.numTargets), bar_format=ut.tqdmBar()):

            # Find corresponding contaminants to our target
            dcc = dc[dc['PIC'] == df['PIC'].iloc[starNo]]

            # We will only add stars within a user-defined distance of our target
            dcc = dcc[dcc['dis'] < self.disConLimit]

            # Add only contaminants brigther than threshold
            dcc = dcc[dcc[self.mag_column] < df[self.mag_column].iloc[starNo] + self.dmagConLimit]

            # Add contaminats to contaminant list
            if dc1.empty and not dcc.empty:
                dc1 = dcc                
            elif not dc1.empty and not dcc.empty:
                dc1 = pd.concat([dc1, dcc])

            # Store number of contaminats per target for statistics below
            numConPerTar[starNo] = len(dcc)

        # Add the number of contaminats to the target catalog
        self.df['ncon'] = numConPerTar
        self.dc = dc1

        
    def plotTargetsPIC(self):
        """Plot function for PIC targets.
        """
        df = self.df
        
        # Plot targets in aitoff Galactic sky projection
        self.fig0, ax = pt.drawStarsInSkyAitoff(df.ra, df.dec, df[self.mag_column])
        if self.plot: plt.show()

        # Plot Zoom-in on FOV
        if df.shape[0] > 1000: mag = None
        else: mag = df[self.mag_column].to_numpy()
        self.fig1, ax = pt.plotPlatoFOV(self.field, ncamStars=True, title=self.title,
                                        raStars=df.ra, decStars=df.dec, magStars=mag,
                                        lw=0.2, ncamMap=self.pic)
        if self.plot: plt.show()
        
        # Plot sample distribution in Teff vs. Radius
        _, ds, sg, dK, dG, dF, sgK, sgG, sgF = self.getStellarClass(self.dx)
        self.fig2, ax = pt.plotTeffvsRadius(ds, dK, dG, dF, sg, sgK, sgG, sgF, df,
                                            self.title)
        if self.plot: plt.show()
        
        
    def plotContaminantsPIC(self):
        """Plot function for PIC contaminants.
        """
        df = self.df
        dc = self.dc
        
        # Necessary to define again in case a PIC sample is defined
        magRange = [df[self.mag_column].min(), df[self.mag_column].max()]
        self.fig3 = plt.subplots(2, 2, figsize=(14,8))
        pt.plotStellarSampleDistributions(self.fig3, magRange,
                                          df[self.mag_column].to_numpy(),  dc[self.mag_column].to_numpy(),
                                          df.ncon.to_numpy(), dc.dis.to_numpy())
        if self.plot: plt.show()


    def prologuePIC(self):
        """Handle output for --pic flag.
        """
        if self.verbose > 1:
            errorcode('module', '\nPrologue')

        # Save output if requested
        if self.outputDir is not None:
            # Copy the YAML file to the project if it doesn't exist
            if self.verbose > 1:
                print(f"Copying YAML configuration file")
            ut.copyInputYAML(self.field, self.outputDir)

            # Save textfile with information about PIC stars
            log = ('PIC Catalogue include\n' +
                   '----------------------------------\n' +
                   f'Catalogue             : {self.pic}\n' +
                   f'PLATO field           : {self.field}\n' +
                   f'PLATO sample          : {self.sample}\n' +
                   f'PLATO camera-group    : {self.group}\n' +
                   f'Magnitude range       : {self.magRange[0]:.2f}-{self.magRange[1]:.2f}\n' +
                   f'Luminosity class      : {self.lumClass}\n' +
                   f'Spectral type         : {self.specType}\n' +
                   f'Contaminant distance  : {self.disConLimit} arcsec\n' +
                   f'Contaminant magnitude : {self.dmagConLimit} mag\n' +
                   self.t.get_string() + '\n')
            with open(self.outputDir / f'{self.outputPrefix}.log','w') as file:
                file.write(log)

            # Save figures
            resolution = 200
            if self.verbose > 1:
                print('Saving all plots to {0}'.format(self.outputDir))
            self.fig0.savefig(self.outputDir / f'plot_{self.outputPrefix}_allsky.png',
                              bbox_inches='tight', dpi=resolution)
            self.fig1.savefig(self.outputDir / f'plot_{self.outputPrefix}_pointing.png',
                              bbox_inches='tight', dpi=resolution)
            self.fig2.savefig(self.outputDir / f'plot_{self.outputPrefix}_TeffvsRadius.png',
                              bbox_inches='tight', dpi=resolution)
            if not self.fig3 is False:
                self.fig3[0].savefig(self.outputDir / f'plot_{self.outputPrefix}_distribution.png',
                                     bbox_inches='tight', dpi=resolution)            
                
            # Save ascii catalog (PlatoSim) or feathers (PLATOnium)
            if self.saveAscii:
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileCat}')
                df0        = pd.concat([self.df.ra,  self.dc.ra])
                df0['dec'] = pd.concat([self.df.dec, self.dc.dec])
                df0['mag'] = pd.concat([self.df[self.mag_column], self.dc[self.mag_column]])
                df0.to_csv(self.outputFileCat, sep=' ', header=False, float_format='%.6f')
            
            else:
                # We reset the index in order to save to feather
                df = self.df.reset_index(drop=True)
                dc = self.dc.reset_index(drop=True)
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileTar}')
                df.to_feather(self.outputFileTar)
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileCon}')
                dc.to_feather(self.outputFileCon)

                # Output file name of HPC data
                outputFileVarSim = f'{self.outputDir}/cluster_{self.outputPrefix}.data'
                if self.verbose > 1:
                    print(f'Saving file {outputFileVarSim}')
                headerVar = 'ID,PIC,M,R,Teff'
                df_hpc = pd.DataFrame()
                df_hpc['ID']   = df.ID
                df_hpc['PIC']  = df.PIC
                df_hpc['Teff'] = df.Teff
                df_hpc['R']    = df.R
                df_hpc['M']    = df.M
                df_hpc = df_hpc.reset_index(drop=True)
                df_hpc.to_csv(outputFileVarSim, sep=',', index=False)

    #--------------------------------------------------------------#
    #                   SINGLE GAIA STAR QUERY                     #
    #--------------------------------------------------------------#            
    
    def initSimbad(self):
        """Initialise the Simbad input parameters.
        """
        if self.verbose > 1:
            errorcode('software', '\nSimbad target query')

        # Arguments for GaiaDR3 star query
        self.simbad = args.simbad
        self.field  = args.pipe_field
        self.saveAscii = args.save
        self.inputFiles = args.incat
        self.outputPrefix = f'starcat_'

        # We need to set a sample because it is needed by platonium for the pipeline
        # (P1 treated differently from P5)
        if args.pipe_sample in ['P1', 'P2', 'P4', 'P5', None]:
            self.pipeSample = args.pipe_sample
            if self.pipeSample is not None:
                self.outputPrefix += f"{self.pipeSample}_"
        else:
            errorcode('error', 'Not a valid PIC sample! Use --pipe_sample {P1, P2, P4, P5}')

        # We need to set a field because it is needed by platonium for the pipeline
        # (to dictate the pointing)
        if self.field in ['SPF', 'NPF', 'LOPS2', 'LOPN1', None]:
            self.field = args.pipe_field
            if self.field is not None:
                self.outputPrefix += f'{self.field}_'
        else:
            errorcode('error', 'Not a valid field! Use --pipe_field {SPF, NPF, LOPS2, LOPN1}')

        # Replace a few strings
        self.outputPrefix += self.simbad.replace(' ', '_')
        self.outputPrefixTar = self.outputPrefix + '_targets.ftr'
        self.outputPrefixCon = self.outputPrefix + '_contaminants.ftr'

        # Save either a PlatoSim (ascii) or a PLATOnium (feather) catalogue
        if self.saveAscii:
            self.outputFileCat = self.outputDir / f'{self.outputPrefix}.txt'
        else:
            self.outputFileTar = self.outputDir / self.outputPrefixTar
            self.outputFileCon = self.outputDir / self.outputPrefixCon

        
    def querySimbad(self):
        """Function to query a Gaia DR3 star and its contaminants.
        """
        # Query Gaia DR3 star
        self.df_all = sq.simbadQuery(self.simbad, radius=self.disConLimit)
        if self.verbose > 1:
            print(f'\nGaia sources in the vicinity of {self.simbad}:')
            print(self.df_all)
            
        # Set df to just the first row
        self.df = self.df_all.iloc[:1]
        
        # If there are contaminants, set dc to the rest of the rows
        if len(self.df_all) > 1:
            self.dc = self.df_all.iloc[1:]
            # Apply contamination limits
            self.dc = self.dc[self.dc['Pmag'] < (self.df.iloc[0]["Pmag"] + self.dmagConLimit)]
            # Platonium needs the gaiaDR3 ID to be set to that of the target
            # (to identify contaminants)
            self.dc["gaiaDR3"] = self.df["gaiaDR3"].iloc[0]
        else:
            # Set self.dc to a panda df with same columns as self.df
            self.dc = pd.DataFrame(columns=self.df.columns)

        # Print actual catlogue saved
        if self.verbose > 1:
            print(f'\nCatalogue for {self.simbad}:')
            print(self.df)
            print(f'\nCatalogue for {self.simbad} contaminants:')
            print(self.dc)
            
        # Plot if requested
        if self.plot:
            c = 'yellow'
            cat = SkyCoord(self.df.ra, self.df.dec, frame='icrs', unit=u.deg)
            if cat.galactic.b < 0:
                self.pic   = 'PIC210'
                self.field = 'LOPS2'
            else:
                self.pic   = 'PIC200'
                self.field = 'LOPN1'
            # Plot targets in aitoff Galactic sky projection
            fig, ax = pt.drawStarsInSkyAitoff(self.df.ra, self.df.dec, color=c, ms=50)
            plt.show()
            # Plot Zoom-in on FOV
            fig, ax = pt.plotPlatoFOV(self.field, ncamStars=True, clabel=r'$P$ [mag]', s=150, lw=0.3)
            ax.scatter(cat.ra.deg, cat.dec.deg, marker='o', c=c, ec='k',
                       transform=ax.get_transform('icrs'), s=150)
            plt.show()
            
            
    def prologueSimbad(self):
        """Handle output for --simbad flag.
        """
        if self.verbose > 1:
            errorcode('module', '\nPrologue')

        if self.outputDir is not None:
            # Copy the YAML file to the project if it doesn't exist and a field is parsed
            if self.field is not None:
                if self.verbose > 1:
                    print(f"Copying YAML configuration file")
                ut.copyInputYAML(self.field, self.outputDir)

            # Save ascii catalog (PlatoSim) or feathers (PLATOnium)
            if self.saveAscii:
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileCat}')
                df0 = pd.DataFrame()
                df0['ra']   = pd.concat([self.df.ra, self.dc.ra])
                df0['dec']  = pd.concat([self.df.dec, self.dc.dec])
                df0['Pmag'] = pd.concat([self.df.Pmag, self.dc.Pmag])
                df0 = df0.reset_index()
                df0.to_csv(self.outputFileCat, sep=' ', header=False, float_format='%.6f')
            else:
                # We reset the index in order to save to feather
                df = self.df.reset_index(drop=True)
                dc = self.dc.reset_index(drop=True)
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileTar}')
                df.to_feather(self.outputFileTar)
                if self.verbose > 1:
                    print(f'Saving file {self.outputFileCon}')
                dc.to_feather(self.outputFileCon)

    #--------------------------------------------------------------#
    #                       PLATO CAMERA FOV                       #
    #--------------------------------------------------------------#            

    def initVizier(self):
        """Initialise the Simbad input parameters.
        """
        if self.verbose > 1:
            errorcode('software', '\nVizier PLATO FOV query\n')

        # Arguments for GaiaDR3 star query        
        self.field    = args.vizier
        self.bright   = args.yale_stars
        self.stellar  = args.gaia_stellar
        self.variable = args.gaia_variable
        self.quasar   = args.gaia_quasar
        self.fcam     = args.fcam
        
        # Magnitude limits
        if args.mag_min is None:
            self.mag_min = 0
        else:
            self.mag_min = args.mag_min
        if args.mag_max is None:
            self.mag_max = 15
        else:
            self.mag_max = args.mag_max
            
        # Check if output folder exist
        if self.outputDir:
            self.filename = self.outputDir / f'starcat_GaiaDR3_{self.field}'
        else:
            errorcode('error', 'Option --vizier needs a output destination!')

        # Check for inconsitent query
        if self.bright and self.stellar:
            errorcode('error', 'Flags "yale_stars" and "gaia_astro" cannot be parsed at once!')
            
        # Constants [deg]
        self.rGroup = 19.0                       # Max radius to query stars within a group
        self.cGroup = 9.2                        # Diagonal opening angle of groups
        self.aGroup = np.sqrt(self.cGroup**2/2)  # Horizontal/vertical opening angle of group
        
        # Get pointing of platform [rad] 
        self.alpha, self.delta, self.kappa = ut.getPointingField(self.field, unit='rad')

        # Find the camera group pointings [rad]
        raGroups, decGroups = rf.getCameraGroupCoordinates(self.alpha, self.delta, self.kappa)
        self.raGroups  = np.rad2deg(np.append(raGroups,  self.alpha))
        self.decGroups = np.rad2deg(np.append(decGroups, self.delta))

        # Make a grid in azimuth and tilt angles and find the sky coordinates [deg]
        # Grid constants [deg]
        r = 2.80                   # Radius of grid point search
        a = 3.95                   # Distance between equidistant grid points
        c = np.sqrt(2*a**2)        # Diagonal distance of grid points
        b = np.sqrt(a**2+(2*a)**2) # Semi-diagonal distance of grid points
        n = 6                      # Half-grid points

        # Create Cartesian grid
        gridCC = []
        for xx in range(-n, n+1):
            for yy in range(-n, n+1):
                # Cut-off cornors
                if ((xx==-n and yy==-n) or (xx==-n and yy==-n+1) or (xx==-n+1 and yy==-n) or
                    (xx==-n and yy==+n) or (xx==-n and yy==+n-1) or (xx==-n+1 and yy==+n) or 
                    (xx==+n and yy==-n) or (xx==+n and yy==-n+1) or (xx==+n-1 and yy==-n) or
                    (xx==+n and yy==+n) or (xx==+n and yy==+n-1) or (xx==+n-1 and yy==+n)):
                    pass
                else:
                    gridCC.append((xx*a, yy*a))

        # Get grid points in sky coordinates [deg]   
        x = np.zeros(len(gridCC))
        y = np.zeros(len(gridCC))
        for i in range(len(gridCC)):
            gridEQ = ut.cart2pol(gridCC[i][0], gridCC[i][1])
            x[i], y[i] = rf.platformToTelescopePointingCoordinates(
                self.alpha, self.delta, self.kappa,
                np.deg2rad(gridEQ[0]),
                np.deg2rad(gridEQ[1])
            )
        self.raGrid  = np.rad2deg(x)
        self.decGrid = np.rad2deg(y)

        # Shorten names
        self.r, self.a, self.c = r, a, c
        self.gridCC, self.gridEQ = gridCC, gridEQ

        # Plot grid used for query
        if self.plot:
            self.plotVizier()


    def plotVizier(self):
        """Plot grid in EQ used to make Gaia DR3 source query.
        """
        # PLOT GRID IN EQUATORIAL COORDINATES
        # Shorten names
        r, a, c = self.r, self.a, self.c
        rg, ag  = self.rGroup, self.aGroup
        gridCC = self.gridCC
        gridEQ = self.gridEQ
        # Plot the grid in cartesian coordinates
        fig, ax = plt.subplots(figsize=(7,7))
        for i in range(len(gridCC)):
            ax.add_artist(plt.Circle((gridCC[i][0], gridCC[i][1]), r, color='k', alpha=.2))
        ax.add_artist(plt.Circle(( ag, ag), rg, color='b', alpha=.2))
        ax.add_artist(plt.Circle(( ag,-ag), rg, color='b', alpha=.2))
        ax.add_artist(plt.Circle((-ag,-ag), rg, color='b', alpha=.2))
        ax.add_artist(plt.Circle((-ag, ag), rg, color='b', alpha=.2))
        # Settings
        ax.set_title('Cartesian query grid')
        ax.set_xlabel('x [deg]')
        ax.set_ylabel('y [deg]')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        plt.show()

        # PLOT GRID IN EQUATORIAL COORDINATES
        fig = plt.figure(figsize=(7,7))
        # Plot the grid points on the sky
        platform = SkyCoord(np.rad2deg(self.alpha), np.rad2deg(self.delta),
                            frame='icrs', unit=u.deg)
        ax = plt.axes(projection='astro degrees zoom', center=platform,
                      radius='35 deg', rotate='180 deg')
        # Plot pointing of platform
        ax.plot(platform.ra.deg, platform.dec.deg, '*', c='k', mfc='magenta', ms=25,
                    transform=ax.get_transform('world'))
        # Plot the pointing of each camera group
        colors = ['b', 'limegreen', 'yellow', 'r']
        for i, c in zip(range(4), colors):
            ax.plot(self.raGroups[i], self.decGroups[i], 'o', ms=13, c=c, mec='k',
                    transform=ax.get_transform('world'))
        # Plot grid points
        grid = SkyCoord(self.raGrid, self.decGrid, frame='icrs', unit=u.deg)
        ax.plot(grid.ra.deg, grid.dec.deg, 'o', ms=10, c='k',mec='k',
                transform=ax.get_transform('world'))      
        # Settings
        ax.scalebar((0.05, 0.05), 10 * u.deg).label()
        ax.compass(0.95, 0.05, 0.1)
        ax.grid(color='gray')
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        plt.show()

        
    def queryVizier(self):
        """Query function for Gaia DR3.

        This function query a circular area from the Gaia DR3 given a
        equatorial pointing of the PLATO spacecraft. It uses a grid of 
        9 circular regions to search for stars (in order not to exceed
        the RAM memory) and concatenates these grids into a final Gaia
        custom catalogue spanning each of the camera groups.

        Authors: Juan Cabrera & Nicholas Jannsen
        https://www.cosmos.esa.int/web/gaia-users/archive/programmatic-access
        """
        self.flag_combine = True

        # QUERY CATALOGUE
        
        # Check if catalogue already exist
        starcat = Path(self.outputDir) / f'starcat_GaiaDR3_{self.field}_group1.ftr'
        if starcat.is_file():
            errorcode('warning', 'Camera group output files already exists, skipping query')
        else:
            if self.verbose > 1:
                print(f'Adding stellar  columns: {self.stellar}')
                print(f'Adding variable columns: {self.variable}')
                print(f'Adding quasar   columns: {self.quasar}')
                print(f'\nQuery Gaia DR3 for G-magnitudes: {self.mag_min} - {self.mag_max}')

            # Query stars within the FOV of each grid
            for i in tqdm(range(len(self.raGrid)), bar_format=ut.tqdmBar()):
                df0 = sq.gaiaQueryRegion(self.raGrid[i], self.decGrid[i], radius=self.r,
                                         mag_min=self.mag_min, mag_max=self.mag_max,
                                         flag_stellar=self.stellar,
                                         flag_variable=self.variable,
                                         flag_quasar=self.quasar,
                                         ofile=f'{self.filename}.vot')
                # Concatenate catalogue
                if i == 0: df = df0
                else:      df = pd.concat([df, df0])

            # Remove duplicate stars (from overlapping grid)
            df = df.drop_duplicates(subset=['source_gaia_dr3'])
            if self.verbose > 1:
                print(f'Number of objects in stellar catalogue: {df.shape[0]}')

            # If requested, add bright stars not available in the Gaia catalogue (G > 2)
            # All information if from CDS and magnitudes are in V Johnson-Cousin
            if self.bright:
                if self.verbose > 2:
                    print('[DEBUG]: Adding bright star catalogue')
                self.inputDir = Path(os.getenv("PLATO_PROJECT_HOME")) / 'inputfiles/data_picsim'
                filename = self.inputDir / f'bright_star_catalogue.csv'
                # Download file if not exisiting
                if not filename.is_file():
                    print(f'Downloading {filename.name}')
                    ut.downloadFromFTP(filename.name, self.inputDir, 'plato')
                # Add catalogue to exisiting Gaia sources
                db = pd.read_csv(filename)
                indices = np.arange(db.shape[0]).tolist()
                dx = pd.DataFrame(np.nan, index=indices, columns=df.columns.tolist())
                # We assume that Vmag is similar to Gmag
                dx.source_gaia_dr3 = indices
                dx.ra    = db.ra
                dx.dec   = db.dec
                # Emperical transformation to Gaia colours
                # NOTE Approximate relations from Gaia's docs: Evans et al. relations
                for i in range(db.shape[0]):
                    B_V = db.Bmag.iloc[i] - db.Vmag.iloc[i]
                    dx.BP_RP.iloc[i] = 1.25 * B_V
                    if B_V < 1:
                        dx.Gmag.iloc[i] = db.Vmag.iloc[i] - (0.05 + 0.35 * B_V)
                    if B_V >= 1:
                        dx.Gmag.iloc[i] = db.Vmag.iloc[i] - (0.35 + 0.05 * B_V)
                # Concatenate data frames
                df = pd.concat([df, dx])

            # Keep only stars within the camera group FOV
            if self.verbose > 1:
                print(f'\nCreating catalogue for each camera group')
            for i in range(5):
                # Calculate angular distance [deg]
                dOA = ut.radialDistance(self.raGroups[i], self.decGroups[i],
                                        df.ra.to_numpy(), df.dec.to_numpy())
                df0 = df[dOA < 20]
                # Select output filename
                ofile = f'{self.filename}_group{i+1}.ftr'
                # Save new catalogue
                df0.reset_index(drop=True, inplace=True)
                df0.to_feather(ofile)
            
        # CREATE FINAL CATALOGUE

        # Run PLATOnium simulations to create final catalogue
        if self.flag_combine:
            if self.verbose > 1:
                print(f'\nRunning PLATOnium to find stars within the focal plane')
                
            # Copy YAML to output
            ut.copyVizierInputYAML(self.field, self.outputDir)
            platonium = os.getenv('PLATO_PROJECT_HOME')+'/python/platosim/platonium/platonium.py'

            # Query stars within the FOV of each grid
            if self.fcam:
                ngroup = 1 # NOTE only one since F-CAM groups have identical FOV
                groups = range(5,6)
                string = 'Fcam'
                sims = self.outputDir / 'Fcam1_Q1_ccd1.ftr'
            else:
                ngroup = 4
                groups = range(1,5)
                string = 'Ncam'
                sims = self.outputDir / 'Ncam1.1_Q1_ccd1.ftr'
            # Check for or run simulations
            if sims.is_file():
                errorcode('warning', 'Camera group simulations already exists! ' +
                          'skipping simulations..')
            else:
                for ccd in tqdm(range(1,5), bar_format=ut.tqdmBar()):
                    for group in groups:
                        os.system(f'python {platonium} {ccd} {group} 1 1 --fullframe ' +
                                  f'-i {self.outputDir}/inputfile_vizier.yaml ' +
                                  f'-o {self.outputDir} --nexp 1 -v 0 -w')

            # Load full-frame stellar catalogues
            if self.verbose > 1:
                print(f'\nCombing catalogues into a final PLATO {self.field} catalogue')
            df = pd.DataFrame()
            if self.fcam:
                groups = [1]
            for group in groups:
                for ccd in range(1,5):
                    if self.fcam:
                        sims = f"{self.outputDir}/{string}{group}_Q1_ccd{ccd}.ftr"
                    else:
                        sims = f"{self.outputDir}/{string}{group}.1_Q1_ccd{ccd}.ftr"
                    try:
                        df0 = pd.read_feather(sims)
                    except FileNotFoundError:
                        errorcode('error', f'No stars found on CCD {ccd} of {string} {group}.1')
                    else:
                        df = pd.concat([df, df0])
                                                    
            # Drop a few columns
            df = df.drop(columns=['starID', 'flux', 'xCCD', 'yCCD', 'xFP', 'yFP', 'rOA'])

            # Drop dublicates and keep highest count
            df = df.drop_duplicates(subset=['source_gaia_dr3'])

            # Replace any inf with nan
            df = df.replace(np.inf, np.nan)

            # Merge undefined spec types into the unknown spec type
            if self.stellar:
                df.spec.loc[df[df.spec == ''].index] = 'unknown'
            
            # Replace missing Gaia colors
            if df.BP_RP.isna().sum() > 0:
                if self.quasar:
                    # Mean value from PLATO fields 
                    if self.verbose > 2:
                        print(f'\nDEBUG: Replacing BP_RP = NaN with 0.6 (mean value of Quasars)')
                    df.BP_RP[df.BP_RP.isna()] = 0.6
                else:
                    # Assuming M0 dwarfs for stars
                    if self.verbose > 2:
                        print(f'\nDEBUG: Replacing BP_RP = NaN with 2.0 (mean M0 dwarf star)')
                    df.BP_RP[df.BP_RP.isna()] = 2.0

            # Convert Gmag to Pmag
            dex = df.columns.get_loc('Gmag')
            if self.quasar:
                df.insert(dex, 'Gmag', ut.passbandConversionG2P(df.Gmag, df.BP_RP))
                df = df.rename(columns={'Pmag': 'Gmag'})
                pass
            else:
                Pmag  = ut.passbandConversionG2P(df.Gmag, df.BP_RP)
                PBmag = ut.passbandConversionG2P(df.Gmag, df.BP_RP, camera='fast_blue')
                PRmag = ut.passbandConversionG2P(df.Gmag, df.BP_RP, camera='fast_red')
                df.insert(dex,   'Pmag',  Pmag)
                df.insert(dex+1, 'PBmag', PBmag)
                df.insert(dex+2, 'PRmag', PRmag)
                
            # Remove stars with bad colour solutions
            df.BP_RP.loc[df[(df.BP_RP == 2.000)].index] = np.nan
                
            # Add distances [pc]
            dex = df.columns.get_loc('plx_err'); df.insert(dex+1, 'd', 1/(df.plx/1e3))
            dex = df.columns.get_loc('d'); df.insert(dex+1, 'd_err', 1/(df.plx_err/1e3))

            # Add absolute magnitude (zero extinction if not available)
            Ag = df.Ag.fillna(0.0)
            Mg = df.Gmag - 5 * np.log10(df.d) + 5 - Ag
            dex = df.columns.get_loc('Ag')
            df.insert(dex+1, 'Mg', Mg)
            
            # Sort data frame (add N-CAM visibility)
            if self.fcam:
                # Sort after ncams and Pmag
                df = df.sort_values(by=['Pmag'])                
            else:
                # Fetch N-CAM group visibility
                N = df.shape[0]
                cams = np.zeros(N)
                if self.verbose > 1:
                    print(f'\nFetching the {string} visibility for each star:')

                for i in tqdm(range(N), bar_format=ut.tqdmBar()):
                    # Fetch 4 values ahead (since max 4/2 groups for Ncam/Fcam)
                    dx = df.iloc[i:i+ngroup]
                    # Subtract star ID and count zeros = N-CAM visibility:
                    # Row with highest ncams value is the one we keep below
                    diff = np.array(dx.source_gaia_dr3).astype(int) - int(dx.source_gaia_dr3.iloc[0])
                    cams[i] = np.count_nonzero(diff==0)

                # Add column
                dex = df.columns.get_loc('Pmag')
                df.insert(dex, 'ncam', (cams * 6).astype(int))
            
                # Sort after ncams and Pmag
                df = df.sort_values(by=['ncam', 'Pmag'])

            # Save catalogue be used by varsim
            df.reset_index(drop=True, inplace=True)
            df.to_feather(f'{self.outputDir}/starcat_PlatoCS_{string}_{self.field}.ftr')

            # Remove all files again after loaded
            # os.system(f'rm {self.outputDir}/{string}*')
            # os.system(f'rm {self.outputDir}/inputfile_vizier.yaml')
            
#==============================================================#
#               PARSING COMMAND-LINE ARGUMENTS                 #
#==============================================================#

parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

out_group = parser.add_argument_group('I/O PARAMETERS')
out_group.add_argument('-p', '--plot',    action='store_true',      help='Flag to plot each step of the software')
out_group.add_argument('-v', '--verbose', type=int, metavar='INT',  help='Verbosity level [0, 1, 3] (Default: 1)')
out_group.add_argument('-o', '--outdir',  type=str, metavar='STR',  help='Output directory to save catalogue and plots')
out_group.add_argument('--project',       type=str, metavar='NAME', help='Name of project folder within $PLATO_WORKDIR')

pic_group = parser.add_argument_group('PIC QUERY (PLATO FIELDS)')
pic_group.add_argument('--pic',    type=str, nargs=3, metavar=('STARS', 'SAMPLE', 'FIELD'), action="append", help='Mandatory arguments (Check "--notes")')
pic_group.add_argument('--release', type=str, metavar='STR',  help='PIC release [PIC210 (default), PIC200]')
pic_group.add_argument('--ncams',  type=int, metavar='INT',   help='N-CAM visibility [6, 12, 18, 24]')
pic_group.add_argument('--group',  type=int, metavar='INT',   help='Camera-group w.r.t. Q1 [1, 2, 3, 4]')
pic_group.add_argument('--mag',    type=str, metavar='RANGE', help='P magnitude range of PIC targets (float or dash-range)')
pic_group.add_argument('--spec',   type=str, metavar='STR',   help='Spectral type [F, G, K]')
pic_group.add_argument('--lum',    type=str, metavar='STR',   help='Luminosity class [V, IV]')
pic_group.add_argument('--ntime',  type=int, metavar='INT',   help='Number of individual timeseries to be generated')
pic_group.add_argument('--incat',  type=str, nargs='*',       help='PIC target and contaminant input files [*.ftr, *.txt]')
pic_group.add_argument('--unique', type=str, metavar='STR',   help='Parse old target catalogue to select new unique stars [*.ftr])')
pic_group.add_argument('--save',   action='store_true',       help='Flag to save stars into ascii PlatoSim-like catalogue')
pic_group.add_argument('--notes',  action='store_true',       help='Flag to show the notes of the available PIC catalogues')

bad_group = parser.add_argument_group('SIMBAD QUERY (SKY REGION)')
bad_group.add_argument('--simbad',      type=str, metavar='NAME', help='Simbad target name')
bad_group.add_argument('--pipe_sample', type=str, metavar='INT',  help='PLATO sample for platonium pipeline processing [P1, P2, P4, P5]')
bad_group.add_argument('--pipe_field',  type=str, metavar='INT',  help='PLATO fields for platonium pipeline processing [LOPS2, LOPN1]')

viz_group = parser.add_argument_group('VIZIER QUERY (PLATO FOV)')
viz_group.add_argument('--vizier', type=str,   metavar='FIELD', help='PLATO pointing field')
viz_group.add_argument('--mag_min', type=float, metavar='MAG',  help='Min magnitude to query (Default: 0 mag)')
viz_group.add_argument('--mag_max', type=float, metavar='MAG',  help='Max magnitude to query (Default: 15 mag)')
viz_group.add_argument('--yale_stars',    action='store_true',  help='Flag to add the Yale bright stars catalogue')
viz_group.add_argument('--gaia_stellar',  action='store_true',  help='Flag to add stellar parameters to catalogue')
viz_group.add_argument('--gaia_variable', action='store_true',  help='Flag to add variabe parameters to catalogue')
viz_group.add_argument('--gaia_quasar',   action='store_true',  help='Flag to add Quasars parameters to catalogue')
viz_group.add_argument('--fcam',          action='store_true',  help='Generate catalogue for F-CAMs instead')

que_group = parser.add_argument_group('GENERIC QUERY OPTIONS')
que_group.add_argument('--dmag', type=int, metavar='MAG',    help='Delta magnitude target-to-contaminant limit (Default: 5 mag)')
que_group.add_argument('--dist', type=int, metavar='ARCSEC', help='Radial distance target-to-contaminant limit (Default: 45 arcsec)')

args = parser.parse_args()


#--------------------------------------------------------------#
#                            WORKFLOW                          #
#--------------------------------------------------------------#

# Start time tracking
tic = datetime.datetime.now()

# Initialize instance of class
p = PicSim(args)

# Print notes only
if args.notes:
    p.printNotesPIC()
    exit()

# Query single star using Simbad
if args.simbad:
    p.initSimbad()
    p.querySimbad()
    p.prologueSimbad()

# Query a PLATO pointing field
elif args.vizier:
    p.initVizier()
    p.queryVizier()
    
# Query stars from the PIC     
elif args.pic:

    p.initPIC()
    p.loadPIC()
    
    # Old picsim catalogue    
    if args.incat:    
        p.loadOldPIC()
        p.queryTargetsPIC()
        p.plotTargetsPIC()
        p.plotContaminantsPIC()
        p.prologuePIC()
        
    # New picsim catalogue
    else:
        p.queryTargetsPIC()
        p.plotTargetsPIC()
        p.queryContaminantsPIC()
        p.plotContaminantsPIC()
        p.prologuePIC()
    
# Finish with output
if (args.verbose is None) or (args.verbose > 1):
    toc = datetime.datetime.now()
    print(f'\nTotal execution time: {toc-tic} [hh:mm:ss]\n')

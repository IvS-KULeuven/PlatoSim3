#!/usr/bin/env python3

"""
Run the PLATOnium functions in parallel (CPUs).
"""

# Python standard
import os
import shutil
import random
import argparse
import datetime
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PLATOnium standard
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config

# PlatoSim imports
import platosim.utilities as ut
from platosim.utilities import errorcode


#==============================================================#
#                   HIGH PERFORMANCE COMPUTING                 #
#==============================================================#


class HPC(object):

    """Class for parallisation of PLATOnium functions.
    """
    
    def __init__(self, project, cpus=6, backend='threading'):

         # PATHS AND INPUT

        # Global variables
        self.cpus    = cpus
        self.backend = backend
        self.project = project
        
        # Absolute pwd path
        self.path = Path(__file__).parent.resolve()
        self.idir = self.path.joinpath(os.getenv('PLATO_PROJECT_HOME'), 'inputfiles')
        self.pdir = self.path.joinpath(os.getenv('PLATO_WORKDIR'), project)
        self.vardir = self.pdir / 'varsource'

        # Software
        simpath = os.getenv('PLATO_PROJECT_HOME') + '/python/platosim/platonium/'
        self.VARSIM    = simpath + 'varsim.py'
        self.PLATONIUM = simpath + 'platonium.py'




        
    def run(self, script, param_file=False, odir=False, vdir=False,
            sim_range=False, kwargs=None):

        """Function to run the parallelisation.
        """

        # Output folder
        if not odir:
            self.odir = self.pdir / 'output'
        else:
            self.odir = odir
            
        # Parse additional arguments
        self.kwargs = kwargs
            
        # Parsing of parameters
        if param_file:
            param_file = self.pdir / 'input' / param_file 
            params = pd.read_csv(param_file).to_numpy()
            N, M = params.shape
            sim_range = range(N)
        elif sim_range:
            sim_range = range(sim_range[0]-1, sim_range[1])
        else:
            errorcode('error', 'Use either "param_file" or "sim_range"!')

        # Configure parallel computing
        
        with parallel_config(backend=self.backend, n_jobs=self.cpus):

            # RUN PLATONIUM
            
            if script == 'platonium':
                Parallel()(delayed(self.run_platonium)(i, N, M, params)
                           for i in tqdm(sim_range, bar_format=ut.tqdmBar()))

            # RUN VARSIM
                
            if script == 'varsim' and self.project == 'mocka':
                Parallel()(delayed(self.run_varsim_mocka)(i)
                           for i in tqdm(sim_range, bar_format=ut.tqdmBar()))

            elif script == 'varsim' and self.project == 'cs-smbhb':
                Parallel()(delayed(self.run_varsim_smbhb)(i)
                           for i in tqdm(sim_range, bar_format=ut.tqdmBar()))                

            elif script == 'varsim':
                Parallel()(delayed(self.run_varsim)(i, N, params[i])
                           for i in tqdm(sim_range, bar_format=ut.tqdmBar()))

            
    #--------------------------------------------------------------#
    #                        HPC FOR PLATONIUM                     #
    #--------------------------------------------------------------#

    
    def run_platonium(self, i, N, M, params):

        """Function to run the PLATOnium in parallel.
        """
        
        S = int(params[i,0])
        G = int(params[i,1])
        C = int(params[i,2])
        Q = int(params[i,3])
        starID = f'{S}'.zfill(9)

        # Testing MOCKA locally
        #varfile = self.vardir / f'{starID}' / f'varsource_001.txt'
        #vararg = f'--varfile {varfile}'
        
        # Parse arguments
        # if vdir:
        #     starID9 = f'{S}'.zfill(9)
        #     varfile = self.vardir / f'varsource_{starID}.txt'
        #     varlist = self.vardir / f'{starID}' / f'varSourceList.txt'
        #     if varfile.is_file():
        #         vararg = f'--varfile {varfile}'
        #     elif varlist.is_file():
        #         vararg = f'--varlist {varlist}'
        
        # Run PlatoSim simulation
        os.system(f'{self.PLATONIUM} {S} {G} {C} {Q} --project {self.project} ' +
                  f'-o {self.odir} {self.kwargs} -v 0 -w')





    #--------------------------------------------------------------#
    #                         HPC FOR VARSIM                       #
    #--------------------------------------------------------------#                


    def run_varsim_smbhb(self, i):

        """Function to run the parallelisation.
        """

        starID = f'{i+1}'.zfill(9)
        ofile  = f'{self.odir}/varsource_{starID}.txt'
        os.system(f'{self.VARSIM} --binary SMBH --quarter 1-8 -o {ofile} -v 0')
        




    def run_varsim_mocka(self, i):

        """Function to run the parallelisation.
        """
        
        os.system(f'{self.VARSIM} --mocka {self.project} gDor {i+1} no {self.odir} ' +
                  '--quarter 1-8 -v 0')

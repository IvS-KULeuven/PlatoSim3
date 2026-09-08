#!/usr/bin/env python3

"""
Library of functions used for SLURM computations.

NOTE This class needs the Poetry install: 
     >> poetry install --with platonium 
"""

# Python standard
import os
import math 
import argparse
import datetime
from textwrap import dedent

# PlatoSim standard
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# PlatoSim imports
import platosim.utilities as ut
from platosim.utilities import errorcode


#==============================================================#
#                        SLURM UTILITIES                       #
#==============================================================#


def getJobScript(ids, groups, cameras, quarters,
                 nodes=1, cpusPerNode=36, memoryPerCore=184,
                 memoryPerSim=300, timePerSim=15, timeSafeLimit=1,
                 ofile=False):

    """Function to create a SLURM job script to be used on the VSC.

    Parameters
    ----------
    ids : range()
        Python range function with IDs (stars or CCDs)
    groups : range()
        Python range function with camera group IDs {1, 2, 3, 4}
    cameras : range()
        Python range function with camera IDs {1, 2, 3, 4, 5, 6}
    quarters : range()
        Python range function with mission quarters {1, 2, ...}
    ofile : string
        Absolute path to the output ascii file saved
    nodes : int
        Number of nodes
    cpusPerNode : int
        Number of node-cores (cpus)
    memoryPerNode : float
        Max RAM and storage memory for the node [Gb]
        NOTE: This should be the maximum memory allowed -8GB.
    memoryPerSim : float
        Expected memory consumption by the heaviest job [Mb]
    timePerSim : float
        Expected execution time by the slowest job [min]
    timeSafeLimit : float
        Overestimation of calculated memory and walltime [%]
    ofile : string
        Absolute path to the output ascii file saved

    Returns
    -------
    script : string
        A string containing the requested job script
    """

    # Number of idividual simulations
    numSim = ids[-1] * groups[-1] * cameras[-1] * quarters[-1]

    # Estimate number of nodes
    nodes = int(math.ceil(numSim/float(cpusPerNode)))

    # Estimate number of node-cores
    if nodes == 1:
        cores = cpusPerNode
    elif nodes > nodes:
        nodes = nodes
        cores = int(cpusPerNode * nodes)
    else:
        cores = int(math.ceil(numSim/nodes))

    # Estimate memory per core-node [Mb]
    memSim = memoryPerSim / (nodes * cores)
    mem    = int(math.ceil(memSim + timeSafeLimit/100 * memSim))

    # Check if simulation quarter is larger than 5Gb per node-core
    if memSim > 5000:
        errorcode('warning', 'Memory per node-core exceedes 5 GB!')

    # Check if if the total memory exceedes that of the cluster [Gb]
    memSimPerNode = cpusPerNode * memSim * 1e-3

    if memSimPerNode > (memoryPerCore*cpusPerNode)-8:
        errorcode('error', 'Simulation node-memory of {int(memPerNode)} GB ' +
                  'exceedes the cluster node-memory of {pmemCluster}-8 GB!')

    # Estimate wall time given the resources above:
    walltimeSafe = timePerSim + timeSafeLimit/100 * timePerSim
    walltime     = str(datetime.timedelta(seconds=int(walltimeSafe)))

    # Generate job script

    script = \
    f"""
    #!/bin/bash -l

    #SBATCH -A <account>               # Account name of cluster (mandatory)
    #SBATCH --mail-user=<email>        # User email
    #SBATCH --mail-type=END            # Get notified by email if script ends successfuly
    #SBATCH --mail-type=FAIL           # Get notified by email if script fails
    #SBATCH -o <stdout>                # Name of standard output
    #SBATCH -e <stderr>                # Name of standard errors
    #SBATCH -M <cluster>               # Cluster name (VSC: genius or wice)
    #SBATCH -p <partition>             # Cluster software (VSC: batch_debug,batch,bigmen,etc).
    #SBATCH --nodes={nodes}            # Number of nodes
    #SBATCH --ntasks-per-node={cores}  # Number of CPU per node
    #SBATCH --mem-per-cpu={memSim}M    # Amount of RAM memory per CPU
    #SBATCH -t {walltime}              # Totoal execution of slurm job 

    cd $SLURM_SUBMIT_DIR
    module purge                       # Pruge and refresh modules
    module restore <module_env>        # Name of cluster module environment

    # User defined parameters
    WORKDIR=<workdir_name>             # PLATOnium working directory
    PROJECT=<project_name>             # PLATOnium project directory
    SIMDIR=<storage_name>              # Directory to store output
    starID=$(printf "%09d" $id)        # 9-digit star ID to include varsource

    # Export paths and secure 1 thread/cpu
    export TEMDIR=$VSC_SCRATCH_NODE
    export VARDIR=$VSC_SCRATCH/platosim/$PROJECT/varsource
    export OUTDIR=$VSC_SCRATCH/platosim/$PROJECT/$SIMDIR/$starID
    export PLATO=$VSC_DATA/plato
    export PLATO_WORKDIR=$VSC_DATA/$WORKDIR
    export PLATO_PROJECT_HOME=$VSC_DATA/PlatoSim3
    export PYTHONPATH=$PLATO:$PLATO_PROJECT_HOME/python
    export PLATONIUM=$PLATO_PROJECT_HOME/python/platosim/platonium/platonium.py
    export PATH=$VSC_DATA/miniconda3/bin:$PATH    
    export OMP_NUM_THREADS=1

    # Activate Conda environment
    source activate platonium

    # Run PLATOnium
    python $PLATONIUM $id $group $camera $quarter --project $PROJECT -o $TEMDIR -s $OUTDIR -v 0
    """

    # Save textfile for worker
    if ofile:
        with open(ofile, "w") as ofile:
            ofile.write(dedent(script).strip())

    # Print if simulation will run and if so, with what resources
    print(f'Suggested SLURM job script for {numSim} individual simulations: ' +
          f'nodes={nodes}, cores={cores}, mem={mem}Mb, walltime={walltime}')

    return script





def getParamFile(ids, groups, cameras=None, quarters=None, fcam=False, ofile=False):

    """Function to create a job script to be used on the VSC.

    Parameters
    ----------
    ids : range()
        Python range function with IDs (stars or CCDs)
    groups : range()
        Python range function with camera group IDs {1, 2, 3, 4}
    cameras : range()
        Python range function with camera IDs {1, 2, 3, 4, 5, 6}
    quarters : range()
        Python range function with mission quarters {1, 2, ...}
    fcam : bool
        Flag to produce a file for the two F-CAMs instead
    ofile : string
        Absolute path to the output ascii file saved

    Returns
    -------
    textfile : string
        A string containing the requested parameter space
    """

    # Check if F-CAM is requested -> Group 5
    if fcam:
        G = range(5,6)
        C = range(1,3)

    # Add rows in a loop

    if cameras is None and quarters is None:
        # Add rows in a loop
        textfile = "id,group\n"
        for idNo in ids:
            for groupNo in groups:
                textfile += f"{idNo},{groupNo}\n"
    
    elif quarters is None:
        # Add rows in a loop
        textfile = "id,group,camera\n"
        for idNo in ids:
            for groupNo in groups:
                for cameraNo in cameras:
                    textfile += f"{idNo},{groupNo},{cameraNo}\n"
    else:
        textfile = "id,group,camera,quarter\n"
        for idNo in ids:
            for groupNo in groups:
                for cameraNo in cameras:
                    for quarterNo in quarters:
                        textfile += f"{idNo},{groupNo},{cameraNo},{quarterNo}\n"
                    
    # Save textfile for worker
    if ofile:
        with open(ofile, "w") as ofile:
            ofile.write(dedent(textfile).strip())

    return textfile




def convertWorkerLog(workerLog):

    """Get a pandas data frame of the worker log.

    This function takes the worker log file in ascii format and convert
    it to a pandas data frame for easier data handling.
    """
    
    month_convert = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

    df = pd.read_csv(workerLog, sep='\s+|\t+|\s+\t+|\t+\s+', engine='python',
                     names=['state', 'a', 'core', 'b', 'c',
                            'month', 'date', 'time', 'year'])
    df = df.drop(['a', 'b', 'c'], axis=1)
    df = df.reset_index()
    df = df.rename(columns={'index':'sim'})
    N = len(df.sim)
    df['datetime'] = [datetime.datetime(df.year.iloc[i],
                                        month_convert[df.month.iloc[i]],
                                        df.date.iloc[i],
                                        int(df.time.iloc[i][0:2]),
                                        int(df.time.iloc[i][3:5]),
                                        int(df.time.iloc[i][6:8])) for i in range(N)]

    return df.drop(['month', 'date', 'time', 'year'], axis=1)




    
def workerOverview(workerLog, paramFile, ofile=False, plot=False, fullFrame=False):

    """Overview of the run time for worker using SLURM.
    """

    # Load data frame and manipulate worker log 

    df = convertWorkerLog(workerLog)        
    unique = df.sim.unique()
    umax = len(unique)
    imax = 96

    # If requested, plot an overview of start and end times
    
    if plot:
        if umax < imax: nmax = umax
        else: nmax = imax
        fig, ax = plt.subplots(1, 1, figsize=(8, int(3.0+0.1*nmax)))

    dex, walltime = [], []
    for i in tqdm(unique, bar_format=ut.tqdmBar()):
        df0 = df[df.sim == i]

        # Completed CPUs
        if df0.shape[0] == 2 and df0.state.to_numpy()[1] == 'completed':
            walltime.append(df0.datetime.iloc[1].timestamp() - df0.datetime.iloc[0].timestamp())
            if plot and i < imax:

                if fullFrame:
                    if i <= 24: c = 'blue'
                    elif i > 24 and i <= 48: c = 'turquoise'
                    elif i > 48 and i <= 72: c = 'orange'
                    else: c = 'deeppink'
                else:
                    c = 'b'
                
                ax.hlines(df0.sim.iloc[0], df0.datetime.iloc[0], df0.datetime.iloc[1], color=c, alpha=0.5)
                ax.plot(df0.datetime.iloc[0], df0.sim.iloc[0], c=c, marker='>', mec='k', ms=6)
                ax.plot(df0.datetime.iloc[1], df0.sim.iloc[1], 'lime', marker='s', mec='k', ms=6)

        # Failed CPUs
        elif (df0.shape[0] == 1 or
              df0.shape[0] == 2 and df0.state.to_numpy()[1] == 'failed'):
            dex.append(df0.sim.iloc[0]-1)
            print(f'Simulation {df0.sim.iloc[0]} did not finish!')
            if plot and i < imax:
                ax.plot(df0.datetime.iloc[0], df0.sim.iloc[0], 'r', marker='>', mec='k', ms=6)

    if plot:
        plt.title('Overview of worker')
        plt.xlabel('Date-time')
        plt.ylabel('Simulation no.')
        plt.xticks(rotation=30)
        plt.ylim(0, nmax+1)
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Check how many simulations that did not finish in time
    # Also save a parameterisation file in that case
        
    if dex:
        paramFile = Path(paramFile)
        df1 = pd.read_csv(paramFile)
        df1 = df1.loc[dex]
        errorcode('warning', f'{df1.shape[0]} simulations did not finish within the walltime!')
        if ofile:
            df1.to_csv(ofile)
    else:
        errorcode('message', 'All simulations finished successfully!')
        print(f'Average walltime per cpu  : {np.mean(walltime)/3600:.3f} h')
        print(f'Fastest walltime for cpus : {np.min(walltime)/3600:.3f} h')
        print(f'Slowest walltime for cpus : {np.max(walltime)/3600:.3f} h')
        
    return df, np.array(walltime)

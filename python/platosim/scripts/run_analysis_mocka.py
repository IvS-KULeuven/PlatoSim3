#!/usr/bin/env python3

# Built-in
import os
import warnings
import argparse
from pathlib import Path

# PlatoSim standard
import numpy as np
import pandas as pd

# PLATOnium standard
import star_shadow as ss

# PlatoSim functions
import platosim.utilities as ut
from platosim.lightcurve import LightCurve

# Ignore warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------#
#                        START OF SCRIPT                       #
#--------------------------------------------------------------#

# Parse arguments
parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

man_group = parser.add_argument_group('MANDATORY PARAMETERS')
man_group.add_argument('starID', type=int, help='Star ID')
man_group.add_argument('idir',   type=str, help='Input directory containing star folders')
man_group.add_argument('odir',   type=str, help='Output directory to save analysis')
man_group.add_argument('gdir',   type=str, help='Path to file with data gaps: "instrumentGAP.tab"')

opt_group = parser.add_argument_group('OPTIONAL PARAMETERS')
opt_group.add_argument('--bin_size',  metavar='FLOAT', type=float, help='Time bin size [sec]')
opt_group.add_argument('--snr_thres', metavar='FLOAT', type=float, help='Optimal SNR criterion')
opt_group.add_argument('-v', '--verbose', action='store_true', help='Flag print to bash')
opt_group.add_argument('-c', '--clean',   action='store_true', help='Flag to remove camera data')

args = parser.parse_args()

# User defined parameters
verbose   = args.verbose
bin_size  = args.bin_size / 3600  # [hours]
snr_thres = args.snr_thres

# File paths
star = f'{args.starID}'.zfill(9)
idir = Path(args.idir).resolve() / star
odir = Path(args.odir).resolve()
gdir = Path(args.gdir).resolve()
odir_final = odir / 'lightcurve'
odir_table = odir / 'table'
odir_modes = odir / 'modes'

# Create folders
odir_final.mkdir(parents=True, exist_ok=True)
odir_table.mkdir(parents=True, exist_ok=True)
os.system(f'chmod 755 {odir_final}')
os.system(f'chmod 755 {odir_table}')
if snr_thres:
    odir_modes.mkdir(parents=True, exist_ok=True)
    os.system(f'chmod 755 {odir_modes}')

# Define filenames
filename_final = f'lc_{star}'
filename_table = f'table_{star}'
filename_modes = f'modes_{star}'
filename_dat = odir_final / f'{filename_final}.dat'
filename_ftr = odir_final / f'{filename_final}.ftr'
filename_tab = odir_table / f'{filename_table}.ftr'
filename_mod = odir_modes / f'{filename_modes}.ftr'
filename_gap = gdir / 'instrumentGAP.tab'


# EXTRACT FINAL LIGHT CURVE

# Construct light curve object
lcs = LightCurve(idir, 'multi')

# Save simulation table
if args.verbose:
    print('Saving simulation table')
ds = lcs.stat_sim_table(filename_tab)

# Merge ligth curves
lc = lcs.merge(suffix='ftr',
               verbose=verbose,
               flux_group_mean=True,
               binsize=bin_size,
               clip=ds.mag.iloc[0],
               flux_offset=True,
               flux_err=True,
               ofile=filename_ftr)


# STAR SHADOW ANALYSIS

if snr_thres:

    # Introducing data gaps
    if args.verbose:
        print('Introducing data gaps')
    df = lc.gaps(filename_gap, replace=True)
    df = df.dropna()
    
    # Prepare light curve for starshadow
    df.time /= 86400.
    df.to_csv(filename_dat, sep=' ', index=False, header=False)

    # Perform prewhitening using STARSHADOW
    ss.analyse_lc_from_file(str(filename_dat), save_dir=str(odir_final), stage='freq',
                            overwrite=True, verbose=args.verbose, sn_thr=snr_thres)

    # Load file containing columns
    folder_hdf5   = odir_final  / f'{filename_final}_analysis'
    filename_hdf5 = folder_hdf5 / f'{filename_final}_analysis_2.hdf5' 
    result = ss.utility.read_parameters_hdf5(filename_hdf5, verbose=args.verbose)

    # Save data into feather file
    mean = result['sin_mean']
    err  = result['sin_err']
    snr  = result['sin_select']
    df = pd.DataFrame()
    df['freq']       = mean[2]      # [c/d]
    df['freq_err']   = err[2]       # [c/d]
    df['ampl']       = mean[3]*1e6  # [ppm]
    df['ampl_err']   = err[3]*1e6   # [ppm]
    df['phase']      = mean[4]      # [rad]
    df['phase_err']  = err[4]       # [rad]
    df['passed_snr'] = snr[1]       # [bool]
    df = df.sort_values('freq').reset_index(drop=True)

    # Save feather with modes
    df.to_feather(filename_mod)
    os.system(f'chmod 755 {filename_mod}')

    # Remove star shadow light curve
    filename_dat.unlink()
    os.system(f'rm -r {folder_hdf5}')

    
# Remove output and starshadow files
if args.clean:
    os.system(f'rm -r {str(idir)}')

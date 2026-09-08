#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import platosim.noise as ns
import platosim.utilities as ut

#==============================================================#
#               PARSING COMMAND-LINE ARGUMENTS                 #
#==============================================================#

parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

man_group = parser.add_argument_group('MANDATORY PARAMETERS')
man_group.add_argument('ID',   metavar='INT',  type=int, help='Star ID')
man_group.add_argument('idir', metavar='PATH', type=str, help='Input directory')
man_group.add_argument('odir', metavar='PATH', type=str, help='Output directory')
man_group.add_argument('power', metavar='FLOAT', type=float, help='Output directory')

args = parser.parse_args()
ID   = args.ID
idir = args.idir
odir = args.odir
power = args.power

# Time array of 2 years
duration = ut.year() * 2
time_sec = np.arange(0, duration, 25)
time_day = np.arange(0, duration/86400, 25/86400)

# Fetch simulation table
starID = f'{ID}'.zfill(9)

# Check if file exist
output_dir = Path(f'{odir}/{starID}')
ofile = output_dir / 'varsource_001.txt'
#if not ofile.is_file():

# Create varsource from pulsations
dx = pd.read_feather(f'{idir}/pulsations_{starID}_001.ftr')
dv = pd.DataFrame()
dv['time'] = time_sec
dv['dmag'] = ns.timeSeriesFromFourier(time_day, dx.freq, dx.ampl, dx.phase, power=power)

# Save light curve
output_dir.mkdir(parents=True, exist_ok=True)
data = np.transpose([dv.time, dv.dmag])
np.savetxt(str(ofile), data, fmt=['%.1f', '%.8f'])

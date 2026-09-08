#!/usr/bin/env python3

import os
import argparse

import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('filename', type=str, nargs='*', help='Filename of pandas table to show/create')
parser.add_argument('-o', '--outdir', metavar='PATH', type=str, help='Output directory to save')
parser.add_argument('--project',      metavar='NAME', type=str, help='Name of PLATOnium project')
args = parser.parse_args()

# Input files
files = args.filename

# Output directory
if args.outdir:
    odir = Path(args.outdir).resolve()
elif args.project:
    project = args.project
    odir = Path(os.getenv('PLATO_WORKDIR')) / project
else:
    odir = False

# If only one file, simply show the content
# Else concatanate all files given as input and save file

if len(files) == 1:
    print(pd.read_feather(files[0]))
else:
    df = pd.concat((pd.read_feather(f) for f in files), ignore_index=True)
    print(df)
    if odir:
        df.to_feather(f'{odir}/tableSims.ftr')

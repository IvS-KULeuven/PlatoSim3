import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('path', type=str, help='Directory of varsources')
parser.add_argument('star', type=int, help='Source ID')
args = parser.parse_args()

path = Path(args.path)
star = f'{args.star}'.zfill(9)

ifile = path / f'varsource_{star}.feather'
ofile = path / f'varsource_{star}.txt'

df = pd.read_feather(ifile)
data = np.transpose([df.time, df.delta_mag])
np.savetxt(ofile, data, fmt=['%.2f', '%.10f'])
os.system(ifile)

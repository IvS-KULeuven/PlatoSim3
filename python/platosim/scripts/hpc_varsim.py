#!/usr/bin/env python3
from platosim.hpc import HPC
hpc = HPC(project='mocka', cpus=6)
odir = '/lhome/nicholas/software/workdir/mocka/varsource'
hpc.run(script='varsim', odir=odir, sim_range=[1, 6000])

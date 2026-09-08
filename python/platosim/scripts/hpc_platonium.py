#!/usr/bin/env python3

from platosim.hpc import HPC

# Parse arguments
hpc = HPC('mocka', cpus=6)

# Run simulations
hpc.run(script='platonium', param_file='test.data',
       kwargs='--seed 12345 --detrend poly --clip')

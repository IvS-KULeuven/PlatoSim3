#!/usr/bin/env python3

import os
from pathlib import Path
from platosim.utilities import errorcode

#--------------------------------------------------------------#
#             TEST HARNESS FOR PLATONIUM TOOLKIT               #
#--------------------------------------------------------------#

errorcode('software', '\nTEST HARENESS FOR THE PLATONIUM TOOLKIT')

#os.system('source $PLATO_PROJECT_HOME/.venv/bin/activate')
project = 'validation'
path = Path(os.getenv('PLATO_PROJECT_HOME')) / 'python/platosim/platonium'
wdir = Path(os.getenv('PLATO_WORKDIR')) / project
idir = wdir / 'input'

def validation(name, checks):
    """Utility to validate command-line executions
    """
    script = path / f'{name}.py'
    uv_run = f'uv run --active {script}'
    n = len(checks)
    for i,check in zip(range(n), checks):
        cmd = os.system(f'{uv_run} {check}')
        if cmd == 0:
            errorcode('message', f'[Success]: Check {i}/{n-1} passes : {name} {check}')
        elif cmd == 2:
            exit()
        else:
            errorcode('warning', f'Check {i}/{n-1} failed : {name} {check}')

# VALIDATION TESTS FOR PICSIM

errorcode('module', '\nValidation tests for picsim')
checks = [
    f'-h > /dev/null',
    f'--pic 100 P1 LOPS2 --group 1 --ncams 24 --project {project} -v 0',
    f'--pic 100 P1 LOPS2 --spec G --lum V --mag 9.5-11.2 -o {idir} -v 0',
    f'--pic 100 P5 LOPN1 --release PIC200 --dist 30 --dmag 8 --project {project} -v 0',
    #f'--pic all all LOPS2 --incat {idir}/starcat**.ftr --project {project} -v 0',
    #f'--simbad Mizar --project {project} -v 0',
    #f'--vizier LOPN1 --project {project} -v 0',
]
validation('picsim', checks)

# VALIDATION TESTS FOR VARSIM

errorcode('module', '\nValidation tests for varsim')
checks =[
    f'-h > /dev/null',
    f'--star_params 1 1 5777 4.5 0.0 --planet_params 10 50 0 90 0 1 1 --tdur 10 -v 0',
    f'--star Sun --planet Earth --tdur 10 -o {idir}/varsource1.txt -v 0',
    f'--star bCep  --puls ToyModel     -o {idir}/varsource2.txt -v 0',
    f'--star SPB   --puls Pedersen2021 -o {idir}/varsource3.txt -v 0',
    f'--star dSct  --puls Bowman2018   -o {idir}/varsource4.txt -v 0',
    f'--star gDor  --puls Gang2020     -o {idir}/varsource5.txt -v 0',
    f'--star roAp  --puls ToyModel     -o {idir}/varsource6.txt -v 0',
    f'--star RRLyr --puls Planchy2021  -o {idir}/varsource7.txt -v 0',
    f'--star Ceph  --puls Planchy2021  -o {idir}/varsource8.txt -v 0',
]
validation('varsim', checks)

# VALIDATION TESTS FOR PAYLOAD

errorcode('module', '\nValidation tests for payload')
checks = [
    f'-h > /dev/null',
    f'100 LOPS2 --project {project} --ted_ampl 1.5 --quarter 1 -v 0',
]
validation('payload', checks)

# VALIDATION TESTS FOR PAYLOAD

errorcode('module', '\nValidation tests for platonium')
checks = [
    f'-h > /dev/null',
    f'1 1 1 1 --project {project} --sample P1 --nexp 1 --bexp 100 --seed 12345 --performance expected -v 0 -w',
    f'2 2 2 1 --project {project} --field LOPS2 --tdur 0.1 --bdur 5 --con_none --cadence 25 -v 0 -w',
    f'3 3 3 1 --project {project} --field LOPS2 --nexp 1 --mag 10 --con_dmag 5 --con_dist 45 -v 0 -w',
    f'5 1 5 1 --project {project} --field LOPN1 --nexp 1 --varfile {idir}/varsource1.txt -v 0 -w',
    f'6 2 6 1 --project {project} --field LOPN1 --nexp 1 --yaml inputfile.yaml --compress --saveplot -v 0 -w',
    f'4 4 4 1 --project {project} --field LOPS2 --tdur 5 --mask 1 --detrend poly --poly_deg 2 --clip -v 0 -w',
]
validation('platonium', checks)

# PROLOGUE

errorcode('module', '\nPrologue and cleaning up')
os.system(f'rm -r {wdir}')
errorcode('message', 'Finito! Thanks for being patient :)\n')

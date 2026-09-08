import os
import sys

varExists = True

if not 'projectDir' in globals():
    varExists = False
    print ("The global variable projectDir does not exist, set projectDir to the proper location in your environment.")

if not 'workDir' in globals():
    varExists = False
    print ("The global variable workDir does not exist, set workDir to the proper location in your environment.")

if varExists:
    os.environ['PLATO_PROJECT_HOME'] = projectDir
    os.environ['PLATO_WORK_DIR']     = workDir

    sys.path.append(projectDir + "/python")

del(varExists)

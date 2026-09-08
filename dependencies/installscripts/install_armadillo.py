#
# install.sh runs automatically this script. It can be done manually running (from the same folder than install.sh):
# $ python ./dependencies/installscripts/install_armadillo.py
#


import os,sys,shutil,subprocess, glob


# Specify the dependency package name

packageName = "armadillo-6.500.4"

# Specify build and install folders

currentWorkingDir = os.getcwd()
buildDir         = currentWorkingDir + "/dependencies/Downloads/"
parentInstallDir = currentWorkingDir + "/dependencies/Installs/"
installDir = currentWorkingDir + "/dependencies/Installs/" + packageName


# Check if /dependencies/Installs directory exists

if not os.path.isdir(parentInstallDir):
    os.mkdir(parentInstallDir)

# If there is an older version of the install dir, make sure to remove it first.

shutil.rmtree(installDir, ignore_errors=True)

# Print a banner

print("\n\n\n")
print("====================")
print("Installing ARMADILLO")
print("====================")
print("\n")

# Build and install package

installProcedure = "cd {build};                                     \
                    tar -xzvf {package}.tgz;                        \
                    cd {package};                                   \
                    mkdir build;                                    \
                    cd build;                                       \
                    cmake ..;                                       \
                    make -j {num_threads};                          \
                    make install DESTDIR={install}".format(build=buildDir, 
                                                           package=packageName, 
                                                           num_threads=os.environ.get("INSTALL_NUM_THREADS"), 
                                                           install=installDir)

process = subprocess.run(installProcedure, shell=True)
if not process.returncode == 0:
    exit(1)


# Armadillo installs the libraries and header files in a folder structure depending on the OS.
# This folder structure is different from what the other dependency packages use, so to harmonize
# structure, we copy the old folder structure into a new one.

# First try to copy the header files to the folder /include.
# Note: from Python 3.5 on it's possible to use "glob(installDir + "/**/include", recursive=True)" 

pathOfIncludeFolder =   glob.glob(installDir + "/include") + glob.glob(installDir + "/*/include") \
                      + glob.glob(installDir + "/*/*/include") + glob.glob(installDir + "/*/*/*/include")

if len(pathOfIncludeFolder) != 1:
    print("Path of include folder: {0}".format(pathOfIncludeFolder))
    print("Armadillo installation failed: please email the folder structure in " + installDir + "to Joris De Ridder")
    exit(1)
else:
    shutil.copytree(pathOfIncludeFolder[0], installDir+"/include")


# Then try to copy the library files to the folder /lib

pathOfLibFolder =   glob.glob(installDir + "/lib") + glob.glob(installDir + "/*/lib")         \
                  + glob.glob(installDir + "/*/*/lib") + glob.glob(installDir + "/*/*/*/lib") \
                  + glob.glob(installDir + "/*/lib64") + glob.glob(installDir + "/*/*/lib64")

if len(pathOfLibFolder) != 1:
    print("Path of lib folder: {0}".format(pathOfLibFolder))
    print("Armadillo installation failed: please email the folder structure in " + installDir + "to Joris De Ridder")
    exit(1)
else:
    shutil.copytree(pathOfLibFolder[0], installDir+"/lib")


# If we arrived here, the copying succeeded, and we need to remove the old folder

#shutil.rmtree(installDir+"/usr", ignore_errors=True)


# For Mac systems, we still need to correct the relative path in the armadillo library

if sys.platform == "darwin":

    correctionProcedure = "cd {install}/lib;                                                            \
                           install_name_tool -id {install}/lib/libarmadillo.dylib libarmadillo.dylib    \
                          ".format(install=installDir)

    subprocess.call(correctionProcedure, shell=True)



# After installation in the install folder, remove the decompressed package folder in 
# the Downloads dir so that only the .tgz file remains in the Downloads dir.

shutil.rmtree(buildDir+packageName, ignore_errors=True)



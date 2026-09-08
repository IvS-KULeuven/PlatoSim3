#
# install.sh runs automatically this script. It can be done manually running (from the same folder than install.sh):
# $ python ./dependencies/installscripts/install_fftw.py
#

import os,sys,shutil,subprocess


# Specify the dependency package name

packageName = "fftw-3.3.4"

# Specify build and install folders

currentWorkingDir = os.getcwd()
buildDir         = currentWorkingDir + "/dependencies/Downloads/"
parentInstallDir = currentWorkingDir + "/dependencies/Installs/"
installDir       = parentInstallDir + packageName 


# Check if /dependencies/Installs directory exists

if not os.path.isdir(parentInstallDir):
    os.mkdir(parentInstallDir)


# If there is an older version of the install dir, make sure to remove it first.

shutil.rmtree(installDir, ignore_errors=True)

# Print a banner

print("\n\n\n")
print("===============")
print("Installing FFTW")
print("===============")
print("\n")

# Build and install package

installProcedure = "cd {build};                                                                       \
                    tar -xvf {package}.tgz;                                                           \
                    cd {package};                                                                     \
                    ./configure --prefix={install} --enable-threads --enable-float --disable-fortran; \
                    make -j {num_threads};                                                            \
                    make install".format(build=buildDir, 
                                         package=packageName, 
                                         install=installDir,
                                         num_threads=os.environ.get("INSTALL_NUM_THREADS"))

process = subprocess.run(installProcedure, shell=True)
if not process.returncode == 0:
    exit(1)



# After installation in the install folder, remove the decompressed package folder in 
# the Downloads dir so that only the .tgz file remains in the Downloads dir.

shutil.rmtree(buildDir+packageName, ignore_errors=True)

#
# install.sh runs automatically this script. It can be done manually running (from the same folder than install.sh):
# $ python ./dependencies/installscripts/install_hdf5.py
#


import os,shutil,subprocess


# Specify the dependency package name

packageName = "hdf5-1.14.6"

# Specify build and install folders

currentWorkingDir = os.getcwd()
buildDir         = currentWorkingDir + "/dependencies/Downloads/"
parentInstallDir = currentWorkingDir + "/dependencies/Installs/"
installDir       = parentInstallDir + packageName 


# Check if /dependencies/Installs directory exists

if not os.path.isdir(parentInstallDir):
    os.mkdir(parentInstallDir)
    
# Remove a possible older version, and create a fresh one

shutil.rmtree(installDir, ignore_errors=True)
os.mkdir(installDir)

# Print a banner

print("\n\n\n")
print("===============")
print("Installing HDF5")
print("===============")
print("\n")

# Build and install package

installProcedure = "cd {build};                                     \
                    tar -xzvf {package}.tar.gz;                     \
                    cd {package};                                   \
                    ./configure --prefix={install} --enable-cxx;    \
                    make -j {num_threads};                          \
                    make install".format(build=buildDir, 
                                         package=packageName, 
                                         install=installDir,
                                         num_threads=os.environ.get("INSTALL_NUM_THREADS"))

process = subprocess.run(installProcedure, shell=True)
if not process.returncode == 0:
    exit(1)

# After installation in the install folder, remove the decompressed package folder in the build dir
# so that only the .tgz file remains in the Downloads folder.

shutil.rmtree(buildDir+packageName, ignore_errors=True)



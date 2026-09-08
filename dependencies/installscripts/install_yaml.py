#
# install.sh runs automatically this script. It can be done manually running (from the same folder than install.sh):
# $ python ./dependencies/installscripts/install_yaml.py
#


import os,shutil,subprocess


# Specify the dependency package name

packageName = "yaml-cpp"

# Specify build and install folders

currentWorkingDir = os.getcwd()
buildDir         = currentWorkingDir + "/dependencies/Downloads/"
parentInstallDir = currentWorkingDir + "/dependencies/Installs/"
installDir       = parentInstallDir + packageName 


# Check if /dependencies/Installs directory exists

if not os.path.isdir(parentInstallDir):
    os.mkdir(parentInstallDir)


# Print a banner

print("\n\n\n")
print("===================")
print("Installing YAML-CPP")
print("===================")
print("\n")

# Build and install package

installProcedure = "cd {build};                                     \
					tar -xzvf {package}.tgz;                        \
					cd {package};                                   \
                    mkdir build;                                    \
                    cd build;                                       \
                    cmake ..;                                       \
                    make -j {num_threads}".format(build=buildDir, 
                                                  package=packageName,
                                                  num_threads=os.environ.get("INSTALL_NUM_THREADS"))

process = subprocess.run(installProcedure, shell=True)
if not process.returncode == 0:
    exit(1)


# If there is an older version of the install dir, make sure to remove it first.

shutil.rmtree(installDir, ignore_errors=True)

# Copy the header files.

shutil.copytree(buildDir+packageName+"/include", installDir+"/include")

# Also copy the library

os.mkdir(installDir+"/lib/")
shutil.copy(buildDir+packageName+"/build/libyaml-cpp.a", installDir+"/lib/")

# After installation in the install folder, remove the decompressed package folder in 
# the Downloads dir so that only the .tgz file remains in the Downloads dir.

shutil.rmtree(buildDir+packageName, ignore_errors=True)



#
# install.sh runs automatically this script. It can be done manually running (from the same folder than install.sh):
# $ python ./dependencies/installscripts/install_faddeeva.py
#


import os,sys,shutil,subprocess, glob


# Specify the dependency package name

packageName1 = "zeromq-4.1.4"
packageName2 = "cppzmq-master"

# Specify build and install folders

currentWorkingDir = os.getcwd()
buildDir         = currentWorkingDir + "/dependencies/Downloads/"
parentInstallDir = currentWorkingDir + "/dependencies/Installs/"
installDir = currentWorkingDir + "/dependencies/Installs/" + packageName1


# Check if /dependencies/Installs directory exists

if not os.path.isdir(parentInstallDir):
    os.mkdir(parentInstallDir)


# If there is an older version of the install dir, make sure to remove it first.

shutil.rmtree(installDir, ignore_errors=True)


# Print a banner

print("\n\n\n")
print("===================")
print("Installing ZEROMQ")
print("===================")
print("\n")


# Build and install package

#installProcedure = 	"cd {build};					\
#			tar -xzvf {package}.tar.gz -C /{install}	\
#			".format(build=buildDir, package=packageName, install=installDir)



installProcedure = "cd {build};                                         \
                    tar -xzvf {package1}.tar.gz;                        \
                    mkdir {install};                                    \
                    cd {package1};                                      \
                    ./configure --prefix={install} --without-libsodium;	\
                    make -j {num_threads};                              \
                    make install;                                       \
                    cd {build};                                         \
                    tar -xzvf {package2}.tar.gz;                        \
                    cd {package2};                                      \
                    mv *.hpp {install}/include/.;                       \
                    ".format(build=buildDir, 
                             package1=packageName1, 
                             package2=packageName2, 
                             install=installDir,
                             num_threads=os.environ.get("INSTALL_NUM_THREADS"))

process = subprocess.run(installProcedure, shell=True)
if not process.returncode == 0:
    exit(1)


# After installation in the install folder, remove the decompressed package folder in 
# the Downloads dir so that only the .tgz file remains in the Downloads dir.

shutil.rmtree(buildDir+packageName1, ignore_errors=True)
shutil.rmtree(buildDir+packageName2, ignore_errors=True)




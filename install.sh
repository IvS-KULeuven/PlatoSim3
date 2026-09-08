#!/bin/bash
#
# Just run this script:
# $ ./install.sh


# Stop this script if we encounter an error with one of the packages

set -e

# Set number of threads
THREADS=4

while getopts "j:" opt; do
    case "$opt" in
        j)
            THREADS=$OPTARG

            # Validate: must be a non-negative integer
            case $THREADS in
                ''|*[!0-9]*)
                    printf '%s\n' "Error: -j requires a positive integer" >&2
                    exit 1
                    ;;
		0)
                    printf '%s\n' "Error: -j 0 not allowed. Edit the script if you know what you are doing." >&2
                    printf '%s\n' "       Please use a positive integer, e.g. -j 4." >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            printf 'Usage: %s -j <number>\n' "$0" >&2
            exit 1
            ;;
    esac
done

# Check if the "python" command exists, if not try "python3".

myPython=python
if ! [ -x "$(command -v python)" ]; then
  echo "Didn't find a python executable, trying python3." 
  myPython=python3
fi


# Install dependencies
# There is one python script under dependencies/installscripts/ folder for each 
# dependency package. Each script Unzip/Untar packages in dependencies/Downloads/ 
# and installs dependencies packages in dependencies/Installs/

INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_hdf5.py
INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_yaml.py
INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_armadillo.py
INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_fftw.py
INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_faddeeva.py
INSTALL_NUM_THREADS="$THREADS" "$myPython" ./dependencies/installscripts/install_zeromq.py


# Build the simulator

if [ ! -d ./build ]; then
    mkdir build
fi

cd build
cmake ..
make  -j "$THREADS"


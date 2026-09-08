
# Given a set of HDF5 output files of PlatoSim, extract the lightcurved saved
# in those files, and save them to .csv files
#
# Example: python3 extractLightcurves outputfile1.hdf5 outputfile2.hdf5 
#
#          would create the files outputfile1_lc.csv and outputfile2_lc.csv
#          containing the lightcurves

import sys
import os
import h5py
import numpy as np

#--- Check the number of input arguments

if len(sys.argv) < 2:
    print("Usage:   $ python3 extractLightcurves.py <outputfile1.hdf5> [<outputfile2.hdf5 ...]")
    exit(1)


#--- Loop over all Platosim output files

for path in sys.argv[1:]:
    if not os.path.exists(path):
        print("Couldn't find HDF5 file " + path + ": skipping")
        continue

    error = []
    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        error.append(e)

    if len(error) != 0:
        print("Couldn't open " + path + " to read: skipping")
        continue
        
    # Verify whether the HDF5 file has the proper structure

    if "Photometry" not in f.keys():
        print(path + " does not have a Photometry group: skipping")
        continue
    if "StarPositions" not in f.keys():
        print(path + " does not have a StarPositions group: skipping")
        continue

    # Get the time points

    time = np.array(f["StarPositions"]["Time"])

    # Get the estimed flux for all stars, in one big matrix, first column being the time points.

    starIDs = list(f["Photometry"]["Lightcurves"].keys())
    
    result = np.zeros((len(time), 1+len(starIDs)))
    result[:,0] = time

    for n in range(len(starIDs)):
        starID = starIDs[n]
        if "estimatedFlux" not in f["Photometry"]["Lightcurves"][starID].keys():
            print(path + " does not have an estimatedFlux array for " + starID + ": skipping " + starID)
            continue
        flux = np.array(f["Photometry"]["Lightcurves"][starID]["estimatedFlux"])
        result[:,n+1] = flux

    # Create the filename of the txt file to write the lightcurves


    fileName = os.path.basename(path)                  # E.g. /a/b/foo.hdf5 -> foo.hdf5
    baseName, extension = os.path.splitext(fileName)   # E.g. "foo" and "hdf5" 
    outputfileName  = baseName + "_lc.csv"

    # Save the lightcurves

    print("Writing lightcurve(s) to " + outputfileName)
    header = "time," + ",".join(starIDs)
    np.savetxt(outputfileName, result, header=header, delimiter=',')



* Changelog PlatoSim 3.7.1



** Improvements

*** Changed YAML with new wavelength depedent values for the N-CAM as reqired BOL calculated from MPDB frozen v.4.

*** Chaged `simulation.useFastCamera()` to properly update YAML entry `Fluxm0` for blue and red filter of F-CAMs.

*** Changed `simulation.useNormalCamera()` to properly update YAML entry `Fluxm0` for N-CAMs.

*** Changed `slurm.getParamerisationFile()` in order to return SLURM parameterisation csv file the two most commen use cases.

*** Changed `starquery.gaiaQueryCone()` to have similar functionality (i.e. returning same columns) as `starquery.gaiaQueryRegion()`.

*** Changed how the initial number of occupied traps is determined for the Short et al model.

*** Changed the `install.sh` file so that it breaks when a dependency fails to install.

*** Changed `lightcurve.star()` to `lightcurve.target()` to be consistent with source not only being a star.

*** Updated the `orbit.txt` file into `oribit_prime_10_years.txt` (See Issue #1163).

*** Updated the technical Jupyter notebooks.

*** Updated the BFE validation test more rebost for fainter targets.

*** Updated the `setup.sh` file to file so that it breaks when a dependency fails to install.

*** Variable cycleTime in Straylight can now be non-integer



** Bug fixes

*** Fixed issue #1087: Correction of the readout time not accounting for the parallel prescan (bias rows).

*** Fixed issue #1169: Bugfix when including bright (Yale) stars in `picsim.py`.

*** Fixed issue #1157: Bugfix of time determination in aberration.

*** Fixed issue #1150: Issue with F-CAM simulations due to overload of HDF5 RAM memory.

*** Bugfix in `varsim.py` for simulating only transits.

*** Bugfix in digital staturation test.

*** Bugfix in destructor of Detector.



** New features/functionality

*** Merged `sovt1` branch into `develop`. New features to setup the spacecraft in `platonium.py` according to latest performance tests.

*** Added bias rows to the number of columns to read out in `simulation.py`.

*** Added multithreaded compilation of dependencies.

*** Added PIC 2.1.0.1 (LOPS2) to the available `picsim` catalogues.

*** Added additional Limb Darkening (LD) models to varsim.py [linear, quadratic, square-root, power-2].

*** Added new arument `--field` to `platonium.py` in order to specify which stellar catalogue to use as input if multiple exists.

*** Added new unit conversions to `utilities.py`: [`cpd2muhz`, `muhz2cpd`, `ppt2mmag`, `mmag2ppt`].

*** Added new section about the output files of the PLATOnium-L1 pipeline setup, plus updated figures and text in PLATOnium tutorials.

*** Added PRNU from file feature

*** Added straylight feature

*** Added bad pixel map feature
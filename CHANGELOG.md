# Changelog for PlatoSim

Checklist to update for a future release:

* Check that Conda installation of PlatoSim works (`master` and `develop` branch)
* Check that test harness returns only `Success` statements
* Check Python package installation `poetry install --with docs`      in a fresh Conda environment
* Check Python package installation `poetry install --with platonium` in a fresh Conda environment
* Update the "version" in the file `pyproject.toml`
* Check all for a successful run of all Jupyter notebooks in `docs/tutorials` (runs with `poetry install` package management)

## Upcoming release

<!-- 3.7.1 -->
<!-- ***** -->

### Fixed

### Changed

### Added

### Removed


<!-- 3.7.1 -->
<!-- ***** -->

## 07/04/2026: 3.7.1

### Fixed

* Fixed issue #1087: Correction of the readout time not accounting for the parallel prescan (bias rows).
* Fixed issue #1169: Bugfix when including bright (Yale) stars in `picsim.py`.
* Fixed issue #1157: Bugfix of time determination in aberration.
* Fixed issue #1150: Issue with F-CAM simulations due to overload of HDF5 RAM memory.
* Bugfix in `varsim.py` for simulating only transits.
* Bugfix in digital staturation test.
* Bugfix in destructor of Detector.

### Changed

* Changed YAML with new wavelength depedent values for the N-CAM as reqired BOL calculated from MPDB frozen v.4.
* Chaged `simulation.useFastCamera()` to properly update YAML entry `Fluxm0` for blue and red filter of F-CAMs.
* Changed `simulation.useNormalCamera()` to properly update YAML entry `Fluxm0` for N-CAMs.
* Changed `slurm.getParamerisationFile()` in order to return SLURM parameterisation csv file the two most commen use cases.
* Changed `starquery.gaiaQueryCone()` to have similar functionality (i.e. returning same columns) as `starquery.gaiaQueryRegion()`.
* Changed how the initial number of occupied traps is determined for the Short et al model.
* Changed the `install.sh` file so that it breaks when a dependency fails to install.
* Changed `lightcurve.star()` to `lightcurve.target()` to be consistent with source not only being a star.
* Updated the `orbit.txt` file into `oribit_prime_10_years.txt` (See Issue #1163).
* Updated the technical Jupyter notebooks.
* Updated the BFE validation test more rebost for fainter targets.
* Updated the `setup.sh` file to file so that it breaks when a dependency fails to install.
* Variable cycleTime in Straylight can now be non-integer

### Added

* Merged `sovt1` branch into `develop`. New features to setup the spacecraft in `platonium.py` according to latest performance tests.
* Added bias rows to the number of columns to read out in `simulation.py`.
* Added multithreaded compilation of dependencies.
* Added PIC 2.1.0.1 (LOPS2) to the available `picsim` catalogues.
* Added additional Limb Darkening (LD) models to varsim.py [linear, quadratic, square-root, power-2].
* Added new arument `--field` to `platonium.py` in order to specify which stellar catalogue to use as input if multiple exists.
* Added new unit conversions to `utilities.py`: [`cpd2muhz`, `muhz2cpd`, `ppt2mmag`, `mmag2ppt`].
* Added new section about the output files of the PLATOnium-L1 pipeline setup, plus updated figures and text in PLATOnium tutorials.
* Added PRNU from file feature
* Added straylight feature
* Added bad pixel map feature

### Removed

* Removed `Dockerfile.txt` from base as this is redundant with the new LESIA L1 pipeline Docker installation.
* Removed `inputfile_FCAM_required.yaml` as file was outdated. New strategy is to use Python to setup a F-CAM simulations using: `simulation.useFastCamera()`.
* Removed old `orbit.txt` file.
* Removing `descartes` dependency.

<!-- 3.7.0 -->
<!-- ***** -->

## 27/03/2025: 3.7.0

### Fixed

* Bugfix in `varsource.py` for using wrong time column
* Bugfix in `lightcurve.py` of NSR functions
* Bugfix a conflict in `pyproject.toml`
* Fixed missing return value in `ClosedLoopDetectorWithAnalyticNonGaussianPSF`
* Bugfix in `simulation.py` when dealing with F-Camera
* Bugfix in `lightcurve.py` using 'single'
* Bugfix to varsim and varsource
* Bugfix to platonium using verbosity level 1
* Bugfix for fetching cadence in lightcurve.py
* Bugfix for `platonium.py` and `payload.py`
* Bugfix for model selection in detrending algorithm
* Bugfix of analysis script for PLATO-CS in `platonium.py`
* Bugfix of varsim's argparse string making it fail when consulting help
* Bugfix for parsing argument ted_ampl in payload.py
* Bugfix in Camera.cpp so that now star position has a value for every exposure
* Bugfix of using seed correctly for stellar flares
* Bugfix of `varsim` for mocka usage
* Bugfix of LPV and bCep model in `varsim.py`
* Bugfix in RR Lyrae and Cepheid model in `varsource.py`
* Bugfix in `simulation.py` `createStarCatalogFileFromPixelCoordinates()` when dealing with quaternions.
* Bugfix in `getFlux` in `simfile.py`, this output of this function was too low.
* Bugfix in platonium.py setting `--cadence` and using `sim.useNormalCamera()`

### Changed

* Added more tweaks for l1 pipeline.
* Refactored aberration in `Sky.cpp`.
* Changed names in technical notebook.
* Cosmetic changes in `Detector.cpp`.
* Modified `applyPhotometry()` in `DetectorWithAnalyticNonGaussianPSF`.
* `simfile.getApertureMask()` now also return SPR.
* Update the plot method used in `plot.py`.
* Update sigma-clipping lower bound threshold for `lightcurve.py`.
* Speedup of lowess stitching algorithm of `lightcurve.py`.
* Upgrade `picsim.py --vizier` feature.
* Lowess-Theil-Sen rebust detrending method added to `lightcurve.py`.
* Move auto sigma selection for clip into `platomium.py`.
* Update F-Camera YAML parameters in `simulation.py`.
* Bugfix for platonium animation calling `simfile.getApertureMask()`.
* Implemented a passband correction in `varsim.py`.
* Added parameters from the input yaml file to the HDF5 file.
* Removed the use of `normal` in `referenceFrames.skyToPixelCoordinates()`.
* Update of pulsation amplitude distribution in `varsource.py`.
* Changed how the number of traps for CTI are calculated.
* Interpolate aberration in between timesteps.
* Update the `playload.py` and `patonium.py`.
* Import `cumtrapz` or `cumulative_trapezoid` based on `scipy` version.
* Changed the comment for the gain nonlinearity in inputfile.
* Update color plotting for Aitoff galactic sky projections.
* `getInfoOfStarWithID()` in `Sky.cpp` now returns a tuple of doubles instead of tuple of ints.
* In `CMakeLists.txt` changed doc string from C++11 to C++14.
* Added option to disable O-C plot in detrend and clip plots.
* Added function to query a target star based on its Gaia DR3 ID in `starquery.py`.
* Update version of `ligo.skymap` to 2.0.0.
* Cosmetic changes in `referenceFrames.py`.
* Update starquery.py to include Gaia DR3 proper motion.
* Redundant time column in `ACS`, `StellarPositions`, and `GhostPositions` should be been removed and added as an individual time column (see addition below). This is breaking backward compatible change only for users that do not use the `SimFile.py` class.

### Added

* Added straylight implementation for moon.
* Added validation test for straylight.
* Added magnitude dependent sigma-clipping value to platonium post-processing.
* Added notebook for MOCKA.
* Added validation test for SPR.
* Solar flare model implemented in `varsim.py`.
* Added script for running `platonium.py` and `varsim.py` in parallel.
* Added new analysis script to merge PLATOnium light curves.
* Added `lightcurve.gaps` method.
* Added new Lesia L1 script to `platonium.py`.
* Added script to generate a time series from pulsation modes of `varsim.py`


<!-- 3.6.1 -->
<!-- ***** -->

## 26/01/2024: 3.6.1

### Fixed

* Bugfix in DatectorWithAnalyticNonGaussian::applyPhotometry() module where not the right mask was used. (Issue #913)
* Bugfix in generation of distortionmap with mapped PSF (Issue #811)
* Simple CTI is no longer written to HDF5 file (Issue #863)
* Fixed sign error in the telescope tilt rotation matrix in both Telescope.cpp and referenceFrames.py (Issue #857)

### Changed

* Redundant time column in `ACS`, `StellarPositions`, and `GhostPositions` have been removed and added as an individual time column (see addition below). This is breaking backward compatible change only for users that do not use the `SimFile.py` class.
* Small bugfixes to code
* Small errors in documentation
* Reduce time it takes to run the validation tests. (Issue #869)
* Removed declared but unused variables in Sky.cpp
* Updates to the online documentation (mainly typos).
* PLATOnium/varsim: Stellar spot modulations has a factor of two too low amplitude. Now the model is much more representative for the expected level of stellar activity.
* PLATOnium/varsim: This method has changed its structure slightly regarding the parsed arguments.
* The verbosity level for all methods in the PLATOnium toolkit (picsim, varsim, payload, and platonium) has changed to be inline with the usage of PlatoSim (except for the `verbose=0` which is designed for running simulations on a computing cluster). 

### Added

* New time column is saved to the HDF5 file by default.
* The PLATOnium toolkit can now simulate the F-CAMs.
* Added AppleClang (difference from Clang) as a possible compiler.
* Added non-linear gain.
* Generally a lot of small bugfixes for the PLATOnium toolkit has been made (Issue #776, #883, #884, #885, #886, #887, #888, #889, #898, #908, #914 #916).
* Jupyter tutorial notebooks did not work for user without a functional LaTeX installation (configured through `.matplotlibrc`). Now the module `import matplotlibrc` checks if the user has a valid LaTeX installation, and if so, activates the LaTeX rendering, and if not, fall back to normal text rendering.
* PLATOnium/picsim: now has the option to create a stellar catalogue for a given target in the CDS/Simbad database.
* PLATOnium/picsim: now has the option to create a stellar catalogue for an entire PLATO pointing field. This can e.g. be used as input to generate full-frame CCD images.
* PLATOnium/platonium: can now simulate full-frame CCD images.
* PLATOnium/platonium: can now simulate the F-CAMs.
* PLATOnium/platonium: the user can now specify if they want to use random and instrumental systematics "as required" or "as expected" ("as build" will be availble closer to launch).
* PLATOnium/platonium: can now perform a simple detrending and stitching to the on-board light curves produced by PlatoSim. 
* Added complete documentation on how to use Poetry to install PLATOnium (for users and developers).
* Added ocumentation about how to setup PLATOnium to run the L1 proto-pipeline. 


<!-- 3.6.0 -->
<!-- ***** -->

## 16/05/2023: 3.6.0

### Fixed

* Changed Mapped distortion and Mapped inverse distortion routines so that they now are each others inverse. Also changed the corresponding python scripts.
* Fixed that `simfile.showImage()` was not scaling the correct flux in ADU.
* Fixed log color scaling bug in `simfile.showImage()`.
* Fixed bug in `mappedGaussianPSF` validation test.
* Fixed bug where generating throughputmap, automatically assumed distortion should be used. For analytic PSF without distortion this gave a error.

### Changed

* Change of HDF5 structure for `Cosmics`, now using consistent upper case letter for `Exposure` (Issue #765)
* Change of YAML inputfile `Telescope` block named `UseDriftFromFile` to `DriftSource` (Issue #766)
* Changed input structure of input files:
  * `ObservingParameters/RApointing` -> `Platform/Orientation/Angles/RAPointing`
  * `ObservingParameters/DecPointing` -> `latform/Orientation/Angles/DecPointing`
* Changed validation tests to deal with new input structure.
* Changed the naming convention SC (spacecraft) into PLM (Payload Module) in python code.
* Made absolute aberation test easier to read.
* Removed ``doc/Validation`` directory. The content of this directory is mostly containd the ``tests/validationTests`` directory.
* Interpolated the ``skyBackground`` map so that the entire sky is filled.
* When the ``WriteBackgroundMap`` option is true, we will either save a time series of the background value (if we use a constant background) or the background map at the beginning of the simulation (if we use a variable background map).
* Changed `simfile.py` functions to remove and cleanup many of the repeating routines used.
* Changed default jitter timescale to 250s instead of 3600s.
* Updated the old Doxygen documentation to a more user-friendly Sphinx documentation.
* Updated the old Jupyter notebook tutorials. Each notebook follows a cronological order and is inline with the changes made to the YAML file.
* Updated all Python function docstrings (to a appropiate Pythonic version).
* Changed directory structure of `/python/platosim`. All command line scripts are now placed within the folder `script`.

### Added

* Added option to save high resolution, analytic non-Gaussian PSF to HDF5.
* Added variable background (Issue #729)
* Added the option to use Quaternions in the input file. (Issue #709)
* Added option to change the output format of the HDF5 output file. This speeds up long simulations.
* Added option to use differently formatted HDF5-PSF files. (Issue #811)
* Added PLATOnium toolkit (see the new documentation page).
* Added Poetry installation for developers (see the new documentation).


<!-- 3.5.5 -->
<!-- ***** -->

## 29/11/2022: 3.5.5

### Fixed

* Fixed bug where mapped distortion was not used when calculating the vignetting. (Issue #716)
* Bugfix in pathlib in `picsim` file
* Fixed colorbar for autoscaling in `simfile` for `showImage` method.

### Changed

* Changed implementation of mapped distortion

### Added

* Added image slider to imageShow() function in `simFile.py`
* Added =Photometry= and =LightCurve= class in `photometry.py` and `lightcurve.py` file
* Added option to change structure of output HDF5 file
* Added documentation on various places


<!-- 3.5.4 -->
<!-- ***** -->

## 09/08/2022: 3.5.4

### Fixed

* Fixed bug where incorrect gain was applied to the right CCD.
* Fixed a bug when applying open shutter smearing when we include relative transmissivity. 
* Fixed issue with CTI (Short2013 model) where we did not use the correct dwell time.

### Changed

* Changed the test for the gain, to correctly test that the left/right CCD have different gains.
* Changed the start time for the orbit.txt file
* changed brighterFatterEffect.py validation test to be able to deal with the different gain values (on different CCDs)

### Added

* Added a sensible estimate of the number of occupied traps when applying "Short2013" CTI model
* Added comparison between new and old Cosmic ray model to the validation tests
* New plot module to plot a star in the CCD focal plane
* Added the platonium package into the PlatoSim python files


<!-- 3.5.3 -->
<!-- ***** -->

## 15/03/2022: 3.5.3

### Fixed

* When cosmics are added for F-Cams, we make sure they can not fall into covered part
* Corrected bug 'calculateSubfieldAroundCoordinates' in 'referenceFrames.py' where middel pixel of subfield was rounded up/down by 1.
* orbit.txt file is now only read if we include aberration in the simulation

### Changed

* Improved stability of validations tests for drift/jitter from file
* Renamed previous starcatalog
* F-Camera can now be simulated with custom CCD

### Added

* Added custom inputfile for F-Camera
* Added new starcatalog
* Added CTI in SmearingMaps for "Short2013" CTI model


<!-- 3.5.2 -->
<!-- ***** -->

## 01/02/2022: 3.5.2

### Fixed

* Corrected bug in validationtest for Cosmics.
* Corrected bug in python mapped distortion functions in python/platosim/referenceFrames.py. (GitHub #659)
* Fixed bug in Camera.cpp. Previously distortion for mapped PSF would only be taken into account when 
includeFieldDistoritions was set to True. Mapped distoriton should always happen, independent of that value. 

### Changed

* Changed field distortion for mapped PSF to deal with more general mapped psf files.
* Changed field distoriton for analytic PSF from Radial model to Wang model. (GitHub #652)
* Changed 'distortedToUndistortedFocalPlaneCoordinates' and 'undistortedToDistortedFocalPlaneCoordinates' in 
python/platosim/referenceFrames.py
* Orientation angle received via network is propagated correctly to the detector (Github #660)
* Changed deprecated 'append' method for pandas dataframe in MappedGaussianPSF validation test into 'concat' method
* Changed pixelToSkyCoordinates function in referenceFrames.py to work better with F-cameras

### Added

* Added inhomogenous trap density (GitHub #639)
* Added validationtest for Short2013 CTI
* Added validationtest for Short2013fromfile CTI
* Added 'distortioncoefficients.txt' and 'distortioninversecoefficients.txt'
* Added a first version of the PlatoSim license
* Added metallic shield around CCD for F-Cameras
* Added validation test for metallic shield around CCDs for F-Cameras


<!-- 3.5.1 -->
<!-- ***** -->

## 29/10/2021: 3.5.1

### Fixed

* The diffused PSF that was saved to the output HDF5 is now rotated with respect to the CCD it falls on. (GitHub #627)
* Fixed bug where the star coordinates where written to the output HDF5 file without taking field distortion into account. (GitHub #631)

### Changed

* The dependencies python install files now check that the `Installs` directory exists and creates this directory if it doesn't. 
* Made the log files for `Camera::makeStarCatalogSelection` clearer. 
* Changed the python functions in `referenceFrames.py`, `plot.py` and `simulation.py` to deal with mapped distortion.
* Mapped distortion now uses a continuous approximation insead of the previous (crude) method of one-to-one fitting of closed point.
* Changed cosmics intensity from uniform to skew-normal. (GitHub #638)
* Renamed `getCosmicsCoordinates()` to `getCosmicsAffectedPixels()` in `simfile.py`

### Added

* Added an option to individually switch on/off extended or pointlike ghosts. 
* Added `getCosmicsInfo()` method to extract the entry position, entry angle and the trail length of all cosmics


<!-- 3.5.0 -->
<!-- ***** -->

## 13/09/2021: 3.5.0

### Fixed

* Timeshift is not applied when reading out CCDs for the F-CAMs (GitHub #540)
* In `hdf5ToFits.py` typecheck before converting to `string` (GitHub #600)
* Removed fortran dependencies in fftw  install script

### Changed

* Update `showSim.py` to include biasMapsRight and biasMapsLeft
* New analytic PSF model and set of parameters for N6000K
* Include more accurate PSF files for mapped PSF model. The new files can be downloaded from the `Prerequisites` section of the PlatoSim website. 
* Updated mail.cpp to allow a new log level: 0: only shows errors and no warning. 
* Update `Simulation::writeInputParametersToHDF5` function.
* Implemented a new mapped distortion method for the mapped PSF model. The distortion table is included in psf files. 
* Updated website 

### Added

* Added a more accurate aberration model. Instead of assuming a circular orbit with constant speed around the sun, we can now include the path of the spacecraft in an orbit file to simulate any time-dependent velocity. An accurate orbit file is included in the `inputfiles` directory. 


<!-- 3.4.1 -->
<!-- ***** -->

## 02/06/2021: 3.4.1

### Fixed

* Corrected the implementation of the Jitter (GitHub issue #508)
* Corrected the implementation of the Drift
* Fixed the conda build in Jenkins

### Changed

* Added option to the method to include time in output of `getYawPitchRoll`
* Apply the BFE after full-well saturation (GitHub issue #584)
* Added the option to have conda install for python 3.6, 3.7, 3.8 and 3.9

### Added

* Test that checks the Jitter on different CCD's
* Added method `getYawPitchRollFromDrift` in the simfile.py file
* Option to save diffused PSF in output HDF5 file (GitHub issue #564)
* Added method `setSubfieldAroundPixelRows` to simulation.py
* Added option to (not) include in output file:
    - High resolution PSF (if PSF is not Analytic Gaussian)
    - Star Catalog
    - Platoform Yaw, Pitch, Roll
    - Transmission Efficiency


<!-- 3.4.0 -->
<!-- ***** -->

## 31/03/2021: 3.4.0

### Fixed

* HighResMap matrix is now initialized with zeros (GitHub issue #543)
* Photon noise applied after CTI (GitHub issue #431)
* CTI by Short et al. has no effect on the maximum value (GitHub issue #403)
* CTI Short2013 model has no effect (GitHub issue #446)
* Incorrect FEE temperature, from file (GitHub issue #413)
* BFE seems to increase total flux (GitHub issue #397)
* Simulation with too short external jitter file (GitHub issue #327)
* Input file of the photometry tutorial needs to be updated (GitHub issue #368)
* Edge effects introduced by convolution with non-analytical PSF (GitHub issue #357)
* Full-image smearing strange value for column 2254 (GitHub issue #450)
* isnan ambiguous in Detector.cpp (GitHub issue #492)
* Some trouble with the routine createStarCatalogFileFromPixelCoordinates (GitHub issue #506)
* Analytical Non Gaussian PSF: strange orientation of the PSF on the CCD (GitHub issue #511)
* Use of a MappedFromFileAsymmetrical PSF with sub-pixel resolution is 1/64 generates an error (GitHub issue #530)

### Changed

* Provide package acces to python scipts (GitHub feature request #548)
* Updated HDF5 library from v10.2 to v12.0 	
* Replaced natural & mechanical vignetting with overall relative transmissivity (GitHub issue #478)
* Read BFE coefficients from file instead of calculating them (GitHub issue #434)
* CCD ID new standard (GitHub issue #416)
* Implemented time-dependent CTI (GitHub issue #476)
* Unique ID needed in ClosedLoopUtility and Log-File (GitHub issue #488)
* Implemented time-dependent CTI (GitHub issue #476)
	
### Added

* Validation & verification notebooks
* Validation & verification scripts
* Added a method to create a single point source
* Record cosmic particle hits in output HDF5 file (GitHub feature request #553)
* Added method to obtain cosmics into SimFile class
* 6.25s time-shift between CCDs (GitHub issue #401)
* Using non-rotationally symmetrical PSF from file (GitHub issue #407)
* Access methods for datasets at top level in HDF5 file (GitHub issue #437)
* Conversion between CCD coordinates & field angles (GitHub issue #460)
* Mechanical vignetting < edge of FOV (GitHub issue #435)
* Transmission variation as a function of FOV (GitHub issue #415)
* Group all contributions to the overall relative transmissivity (GitHub issue #478)
* Implement ghosts (GitHub issue #515)
* Add linear irradiance ratio decrease of point-like ghosts (GitHub issue #526)
* Added getPointLikeGhostCoordinates() and getExtendedGhostCoordinates() methods to SimFile class

	
<!-- 3.3.7 -->
<!-- ***** -->

## 25/02/2020: 3.3.7

### Fixed

* Input file of the photometry tutorial needs to be updated (GitHub issue #368)
* Order of the effects (GitHub issue #394)

### Changed

* Input parameter update (GitHub issue #377):
  + wavelength range
  + irradiance over the PLATO wavelength range (PIS only; nothing changes for PlatoSim)
  + added: orientation angle of the solar panels
  + transmission efficiency
  + throughput bandwidth
  + FEE readout noise
  + gain + stability of the CCD gain
  + QE
  + digital saturation for fast cameras (this is now the same as for the normal cameras)
* Dynamic frame transfer times (GitHub issue #369)

### Added

* FEE over/undershoot (GitHub issue #376)
* Dump Analytical PSF within hdf5 output file (GitHub issue #379)
* Jitter from network


<!-- 3.3.6 -->
<!-- ***** -->

## 10/04/2019: 3.3.6

### Fixed
* Bias register map expressed as additional columns (GitHub issue #290)
* Implement partial readout (GitHub issue #285)
* Backward compatibility to the configuration files? (GitHub issue #292)
* Improve documentation on supplementary input files of PlatoSim (GitHub issue #308)
* Migtool error (GitHub issue #307)
* Update HDF5 dependency to 1.10.2 (GitHub issue #322)
* Remove python 3.5 dependency (GitHub issue #318)
* Improved error trapping for CCD/ReadoutMode/ReadoutMode (GitHub issue #302)
* Incorrect sky background level (GitHub issue #325)
* Incorrect flatfield level (GitHub issue #326)
* Segfault on running test harness (GitHub issue #329)
* Open-shutter smearing not accounted for (GitHub issue #339)
* Implement mechanical vignetting (GitHub issue #334)
* Partial-readout parameters not read out correctly (GitHub issue #346)
* Conserving disc space by writing the images as int matrices into the .hdf5 files (GitHub issue #348)
* Bug fix in the createStarCatalogFileFromPixelCoordinates() method.

### Added

* Temperature dependency of the dark current (space environment)


<!-- 3.3.5 -->
<!-- ***** -->

## 26/10/2018: 3.3.5

### Fixed

* Inconsistency in star position output (GitHub issue #294)


<!-- 3.3.4 -->
<!-- ***** -->

## 21/09/2018: 3.3.4

### Fixed

* bug when using demo_fgs.py with field distortion (GitHub issue #280)
* Wrong number of cosmics for small images (GitHub issue #283)

### Changed

* Cosmics can be enabled/disabled per area (image area / bias map / smearing map)
* Random seeds = -1 => use computer time instead (no longer fast-forward random distributions)

### Added

* Configurable log level
* Updated documentation


<!-- 3.3.3 -->
<!-- ***** -->

## 17/05/2018: 3.3.3

### Fixed

* Dark edge seen at bottom of the sub-field due to CTI by Short et al. (GitHub issue #263)
* User-given sky background not multiplied with the transmissivity of the optics (GitHub issue #265)
* Photon flux of stars should be floored instead of rounded (in the jitter steps) (GitHub issue #267)
* Open-shutter smearing outside sub-field should not take numRowsBiasMap into account (GitHub issue #269)


<!-- 3.3.2 -->
<!-- ***** -->

## 27/04/2018: 3.3.2

### Fixed

* Images datasets output dataype and specification (GitHub issue # 167)

### Added

* Documented h5ls and h5get
* Documented output control parameters
* Documented how to install via conda when no pop-up window would appear, asking for the credentials

### Changed

* Parameter values after release of v1.4 of the data package


<!-- 3.3.1 -->
<!-- ***** -->

## 17/04/2018: 3.3.1

### Fixed

* Path to executable set to /build in simulation.py (GitHub issue #257)


<!-- 3.3.0 -->
<!-- ***** -->

## 9/04/2018: 3.3.0

### Added

* Charge diffusion + jitter smoothing
* Documented installation via conda
* Use an external defined star ID in input star catalogues (GitHub issue #229)
* Dark current
* Brighter-fatter effect (BFE)
* Possibility of header lines and custom star IDs in the star catalogue ASCII files
* Stellar variability
* Created an <code>Examples</code> folder (in the <code>python</code> directory) where demo scripts show how to use the simulator from Python
* Safety checks in <code>Parameter.h</code> to ensure that the time series from a file has time points in strictly increasing order
* Flag to limit size of HDF5 output files
* Group in the input files (<code>ControlHDF5Content</code>) to control the content of the HDF5 output file
    - <code>WriteSubPixelImages</code> (moved from the <code>CCD</code> group: Boolean flag for writing the sub-pixel images to the HDF5 file [default=no]
    - <code>WriteStarPositions</code> (new): Boolean flag for writing the star positions to the HDF5 file [default=yes]
* Cosmics
* <code>Sky</code> section in the input file with the configuration parameters for the sky background and cosmics
* Added scripts for comparison with PIS
* Time dependency for:
	- PSF sigma (analytic non-Gaussian PSF)
	- focal length 
	- throughput maps

### Fixed

* Bug in jitter from file that caused negative heartbeat intervals in some specific cases
* Kernel dimension restrictions (GitHub issue #211)
* Number of cosmics too high (GitHub issue #206)
* Bug in <code>DetectorWithAnalyticNonGaussianPSF:addFlux()</code> that caused the CCD orientation angle to be ignored when the analytic non-Gaussian PSF was chosen
* Bug in <code>createStarCatalogFileFromPixelCoordinates()</code> so that it now also works when in the input yaml file the telescope group ID and/or CCD position is not "Custom"
* ThroughputMap reset? (GitHub issue # 202)

### Changed

* Project number in the documentation (3.2 -> 3.3)
* Removed relative paths in tests (required for automatic testing in Jenkins)
* Reading in exposure time as double (instead of integer)
* <code>JitterFromFile</code> and <code>ThermoElasticDriftFromFile</code> now only read in the relevant parts of the files
* Improved comments in the default input file
* If <code>UseJitter == no</code>, then the jitter file is no longer read, even when <code>UseJitterFromFile == yes</code> (idem for drift)
* Parameter review -> update of configuration parameter values
* Update of the field distortion polynomial
* Update of the documentation pages:
	- re-structuring
	- added information about new configuration parameters and (optional) input files
	- improved the description of configuration parameters and procedures
* Updated <code>CMakeLists.txt</code> to use C++14 rather than C++11
* Incorporated <code>StarCatalog</code> in <code>Sky</code> so that the former becomes obsolete
* Extended <code>Parameter<T,N></code>  such that it can accommodate arrays of scalars (such as the distortion coefficients)

### Removed

* Unnecessary log messages


<!-- 3.2.1 -->
<!-- ***** -->

## 06/04/2018: 3.2.1

### Added

* Clarification of the <code>SubPixel</code> parameter in the configuration file (GitHub issue #175)
* Added parameters for camera groups and pre-defined CCD positions

### Fixed

* Corrected relative paths in tests (for automatic builds + testing)
* Correction of the application of the FEE and CCD gain + values in the configuration files + documentation
* Throughput made dependent of CCD position (GitHub issue # 193)
* Correction of the angle dependency of QE and polarisation
* Corrected orientation angles for CCD 2 and 4

### Changed

* Updated version number in the documentation
* Using new gain and readout noise in the calculation of the photometry
* CCD code A, B, C, and D were replaced by 3, 2, 4, and 1


<!-- 3.2.0 -->
<!-- ***** -->

## 05/07/2017: 3.2.0

### Fixed

* Flux values calculated in Camera (expressed in photons) are now rounded instead of floored

### Changed

* Flatfield map only generated if <code>IncludeFlatfield == "yes"</code>


<!-- 3.2.0 RC2 -->
<!-- ********* -->

## 16/06/2017: 3.2.0-RC2

### Changed

* Set the <code>CCD/Position</code> to <code>Custom</code> such that the default settings are used for backward compatibility with the previous release of PlatoSim3
* Updated documentation with respect to
	* Reference frames
	* Description of Camera groups and pre-defined CCDs
* Updates to tutorials to bring them in-line with the changes in this release


<!-- 3.2.0 RC1 -->
<!-- ********* -->

## 15/05/2017: 3.2.0-RC1

### Added

* More detailed throughput specifications (following PLATO-DLR-PL-RP-001): in addition to vignetting:
	- particulate and molecular contamination,
	- angle-dependent quantum efficiency,
	- and angle-dependent polarisation
* Readout noise: contribution of the FEE and CCD added in quadrature
* Gain: contribution from the CCD and the FEE (different for both detector halves and both ADCs)
* Temperature dependency:
	- Implemented for FEE and CCD gain, and for electronic offset
	- Either fixed at the nominal operating temperature of the component or read from a file; similar to jitter and drift
* Quantisation (i.e. combined effect of (1) gain, (2) electronic offset, (3) rounding pixel values, and (4) digital saturation) can be switched on/off
* Performance optimisation:
	- Split time series into chunks
	- Distribution over nodes using Slurm
	- Angle-dependent analytic Gaussian PSF
	- Angle-dependent analytic non-Gaussian PSF
* Kinematic aberration:
	- Differential & absolute
	- Baseline: circular Earth orbit
* Ageing: linear degradation implemented for the transmission efficiency
* New dependency: Faddeeva (used by the analytical PSFs) 
* Visualisation of the output with <code>h5ls</code> and <code>h5get</code> (Python functions)
* Easy selection of requested camera group from the input file, use <code>Telescope/GroupID</code> = [1,2,3,4,Fast,Custom]
* Easy selection of requested CCD position, use <code>CCD/Position</code> = [1,2,3,4,Custom]

### Changed

* Updated reference frames, following PLATO-DLR-PL-TN-016 and PLATO-OHB-PL-LI-009 (see PLATO-KUL-PL-TN-001), in particular: sunshield pointing towards the Sun
* Electronic offset: moved from CCD to FEE
* Updated input files, incorporating the changes in configuration parameters
* Updated documentation, describing the new configuration parameters
* Updated tutorials, incorporating the changes in configuration parameters

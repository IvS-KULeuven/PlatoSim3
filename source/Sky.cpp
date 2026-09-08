
#include "Sky.h"




/**
 * \brief Constructor
 *
 * \param configParams    Configuration parameters as read from the (e.g. yaml) inputfile
 *
 */

Sky::Sky(ConfigurationParameters &configParams)
{
    // Configure this Sky

    configure(configParams);

    // Open and read the file containing the position and magnitude of all stars
    // The path of the starInputfile should have been set in configure().

    ifstream myfile(starInputfile);
    if (myfile.is_open())
    {
        string line;
        unsigned int n = 0;
        while (getline(myfile, line))
        {
            // Skip empty lines

            if (line.size() == 0) continue;

            // Skip lines that only contain white space

            const string whitespace = " \t\r\n";
            if (line.find_first_not_of(whitespace) == string::npos) continue;

            // Skip header lines starting with '#'.

            if (line[0] == '#') continue;

            istringstream buffer(line);
            vector<double> numbers((istream_iterator<double>(buffer)), istream_iterator<double>());

            // If the line contains 4 numbers then the last one is the star ID. If not, then
            // use the line number (starting from 0) as star ID.

            unsigned int starID;
            if (numbers.size() == 3)
            {
                starID = n;
            }
            if (numbers.size() == 4)
            {
                starID = static_cast<unsigned int>(numbers[3]);
            }

            starDB.emplace(starID, make_tuple(numbers[0] / Angle::degrees, numbers[1] / Angle::degrees, numbers[2]));    // (starID, (RA[rad], DEC[rad], Vmag)
            n++;
        }

        myfile.close();

        Log.info("Sky: found " + to_string(starDB.size()) + " stars in file " + starInputfile);
    }
    else
    {
        Log.error("Sky: Cannot open star catalog file " + starInputfile);
        exit(1);
    }

    // Open and read the file containing the position and velocity of Plato if aberration is included
    // The path of the file  should have been set in configure().
    if (includeAberrationCorrection)
    {
        // time0  is the begin time of the exposure in [s]. It is measured w.r.t. the zero-point that is mentioned
        // the spacecraft orbit file, which is 2014-06-30T12:00:00.000. The start time of exposure # 0 w.r.t. this zero-point
        // is stated in the yaml inputfile in "Camera/AberrationCorrection/StartTime", expressed in [s].
      orbitStartTime = configParams.getDouble("Camera/AberrationCorrection/StartTime");
      double time0 = orbitStartTime
	+ configParams.getDouble("ObservingParameters/BeginExposureNr") * configParams.getDouble("ObservingParameters/CycleTime");
      double endTime   = time0 + configParams.getDouble("ObservingParameters/NumExposures") * configParams.getDouble("ObservingParameters/CycleTime");

        ifstream orbitFile(orbitPlatoFile);
        if (orbitFile.is_open())
        {
            string line;
            while (getline(orbitFile, line))
            {
                // Skip empty lines

                if (line.size() == 0) continue;

                // Skip lines that only contain white space

                const string whitespace = " /t/r/n";
                if (line.find_first_not_of(whitespace) == string::npos) continue;

                // Skip header line starting with '#'.

                if (line[0] == '#') continue;

                istringstream buffer(line);
                vector<double> numbers((istream_iterator<double>(buffer)), istream_iterator<double>());
                if (time0 <= numbers[1])
                {
                    valarray<double> v = {numbers[5], numbers[6], numbers[7]};
                    orbitDB.push_back(make_tuple(numbers[1], v, numbers[8]));    // (time, [v1, v2, v3], |v|)
                            if (numbers[1] > endTime){break;}
                }
            }

            if (orbitDB.size() == 0.)
            {
                Log.error("Sky: no values between start and end time in orbitPlatoFile");
                exit(EXIT_FAILURE);
            }
            myfile.close();

        } else {
          Log.error("Sky: Cannot open orbit file " + orbitPlatoFile);
          exit(1);
        }
    }
}








/**
 * \brief Destructor
 */

Sky::~Sky()
{

}







/**
 * \brief Configure the Sky class with the user given input parameters
 */

void Sky::configure(ConfigurationParameters &configParams)
{
    // Store the path of the general database of stars

    starInputfile = configParams.getAbsoluteFilename("ObservingParameters/StarCatalogFile");

    // Store the path of the file with the Plato orbit
    includeAberrationCorrection = configParams.getBoolean("Camera/IncludeAberrationCorrection");
    orbitPlatoFile = configParams.getAbsoluteFilename("Camera/AberrationCorrection/OrbitFile");


    // If there variable stars, get their time series files

    bool includeVariableSources = configParams.getBoolean("Sky/IncludeVariableSources");
    if (includeVariableSources)
    {
        // the VariableSourceListFile contains two columns:
        // Col 1: star ID   (unsigned integer)
        // Col 2: path to the file with the time series of that variable star.
        //        This time series file should contain:
        //        Col 1: time [d]
        //        Col 2: delta-magnitude

        string variableSourceListFile = configParams.getAbsoluteFilename("Sky/VariableSourceList");


        // Open and read the file containing the list of variable stars

        ifstream myfile(variableSourceListFile);
        if (myfile.is_open())
        {
            unsigned int starID;
            string timeSeriesPath;

            while (myfile >> starID >> timeSeriesPath)
            {
                // Parameter<double> requires an absolute path, so make sure the path specified in
                // timeSeriesFile is absolute.

                if (FileUtilities::isRelative(timeSeriesPath))
                {
                    string projectLocation = configParams.getString("General/ProjectLocation");
                    projectLocation = StringUtilities::replaceEnvironmentVariable(projectLocation);
                    timeSeriesPath = projectLocation + "/" + timeSeriesPath;
                }
               
                // Store the user specified time series of delta Magnitude for this star in a map<>.

                deltaMagnitude.emplace(starID, make_unique<Parameter<double>>(timeSeriesPath, 1));
            }

            myfile.close();

            Log.info("Sky: found " + to_string(deltaMagnitude.size()) + " variable stars in file " + variableSourceListFile);
        }
        else
        {
            Log.error("Sky: Cannot read the variable star source list file " + variableSourceListFile);
            exit(1);
        }
    }


}










/**
 * \brief Update the time dependent parameters of Sky (e.g. stellar variability) to their
 *        value at the given time point
 *
 * \param time: current time
 *
 * \return
 */

void Sky::updateParameters(double time)
{
    // Update the delta-magnitude of each variable source and add the value to the magnitude of the star

    for (unsigned int n = 0; n < selectedVariableStars.size(); n++)
    {
        const unsigned int index = selectedVariableStars[n];
        const unsigned int starID = selectedStarID[index];

        auto& deltaMag = deltaMagnitude[starID];      // deltaMag is a Parameter<double> pointing to a time series file
        deltaMag->updateValue(time);

        const double Vmag0 = get<2>(starDB[starID]);  // Tuple elements [0], [1], [2] contain RA, dec, Vmag
        selectedVmag[index] = Vmag0 + (*deltaMag)();
    }
}












/**
 *  \brief Compute the equatorial sky coordinates of the Sun given the julian date.
 *
 *  \param julianDate         Julian date (floating point). The distinction between JD and BJD is neglected.
 *  \param outputAngleUnit    Angle::radians if output angles should be in radians, Angle::degrees if in degrees
 *  \return (RA, DEC)         Pair containing the equatorial sky coordinates of the Sun
 *
 *  \note Source: "Computing the solar vector", Blanco-Muriel et al., (2001), Solar Energy, Vol 70, pp 431-441
 */

pair<double, double> getSunCoordinates(double julianDate, Unit outputAngleUnit = Angle::degrees)
{
    // Compute the (fractional) number of days since 1 Jan 2000 at 12:00 noon.

    const double elapsedJulianDays = julianDate - 2451545.0;

    // In the following we assume that the solar ecliptic latitude is always exactly 0.0.

    const double Omega = 2.1429 - 0.0010394594 * elapsedJulianDays;
    const double meanLongitude = 4.8950630 + 0.017202791698 * elapsedJulianDays;
    const double meanAnomaly = 6.2400600 + 0.0172019699 * elapsedJulianDays;

    const double eclipticLongitude = meanLongitude + 0.03341607 * sin(meanAnomaly) + 0.00034894 * sin(2*meanAnomaly) - 0.0001134 - 0.0000203 * sin(Omega);
    const double eclipticObliquity = 0.4090928 - 6.2140e-9 * elapsedJulianDays + 0.00003963 * cos(Omega);

    // Compute the RA, DEC of the Sun. Ensure that the RA is positive.

    double rightAscensionSun = atan2(cos(eclipticObliquity) * sin(eclipticLongitude), cos(eclipticLongitude));
    double declinationSun = asin(sin(eclipticObliquity) * sin(eclipticLongitude));

    if (rightAscensionSun < 0.0) rightAscensionSun += 2.0 * 3.141592653589793;

    // Convert the unit to radians or degrees, whatever the user requested

    rightAscensionSun *= outputAngleUnit;
    declinationSun *= outputAngleUnit;

    // That's it

    return make_pair(rightAscensionSun, declinationSun);
}










/**
 * \brief Given a circle on the sky, select all stars from the database within that circle.
 *
 * \note The stars right on the circle are also included in the catalogue (in the output file).
 *
 * \param RA0 Right ascencsion of the centre point of the circle on the sky [degrees or radians (see further)].
 * \param dec0 Declination of the centre point of the circle on the sky [degrees or radians (see further)].
 * \param radius Radius of the circle on the sky [degrees or radians (see further)].
 * \param angleUnit If the input angles are in degrees: Angle:degrees, if in radians: Angle::radians.
 *
 * \return Total number of selected stars.
 */
unsigned long Sky::selectStarsWithinRadiusFrom(double RA0, double dec0, double radius, Unit angleUnit)
{
    // All computations are done in radians, so if RA0, dec0, and radius are expressed in degrees,
    // divide the degree unit away into radians.

    double RACircleCenter  = RA0    / angleUnit;      // [radians]
    double decCircleCenter = dec0   / angleUnit;      // [radians]
    double radiusCircle    = radius / angleUnit;      // [radians]

    // Reset possible previous selections

    selectedStarID.clear();
    selectedRA.clear();
    selectedDec.clear();
    selectedVmag.clear();
    selectedVariableStars.clear();

    // Copy the star ID, RA, Dec, and Vmag of the selected stars.
    // It's not sufficient to simply keep the starIDs of the selected stars, because the coordinates
    // of the selected stars may change due to aberration, or the magnitude may change due to variability.
    // We don't want to apply such changes to the original database of stars.

    for (auto const& star: starDB)
    {
        unsigned int starID = star.first;
        double RA, dec, Vmag;
        tie(RA, dec, Vmag) = star.second;
        double angularDistances = angularDistanceBetween(RACircleCenter, decCircleCenter, RA, dec, Angle::radians);  // [rad]

        if (angularDistances <= radiusCircle)
        {
            selectedStarID.push_back(starID);
            selectedRA.push_back(RA);
            selectedDec.push_back(dec);
            selectedVmag.push_back(Vmag);

            // Also keep track of which selected stars are variable. Saves us many search loops afterwards.
            // selectedVariableStars contains the _indices_ (of selected*) of those stars that are variable.

            if (deltaMagnitude.find(starID) != deltaMagnitude.end())
            {
                selectedVariableStars.push_back(selectedStarID.size()-1);
            }
        }
    }

    return selectedStarID.size();
}


/**
 * \brief Given a circle on the sky, select all stars from the database within that circle.
 *
 * \note The stars right on the circle are also included in the catalogue (in the output file).
 *
 * \param RA0 Right ascencsion of the centre point of the circle on the sky [degrees or radians (see further)].
 * \param dec0 Declination of the centre point of the circle on the sky [degrees or radians (see further)].
 * \param radius Radius of the circle on the sky [degrees or radians (see further)].
 * \param angleUnit If the input angles are in degrees: Angle:degrees, if in radians: Angle::radians.
 *
 * \return Total number of selected stars.
 */
unsigned long Sky::selectGhostOrigsWithinRadiusFrom(double RA0, double dec0, double radius, Unit angleUnit)
{
    // All computations are done in radians, so if RA0, dec0, and radius are expressed in degrees,
    // divide the degree unit away into radians.

    double RACircleCenter  = RA0    / angleUnit;      // [radians]
    double decCircleCenter = dec0   / angleUnit;      // [radians]
    double radiusCircle    = radius / angleUnit;      // [radians]

    // Reset possible previous selections

    selectedGhostOrigID.clear();
    selectedGhostOrigRA.clear();
    selectedGhostOrigDec.clear();
    selectedGhostOrigVmag.clear();
    selectedVariableGhostOrigs.clear();

    // Copy the star ID, RA, Dec, and Vmag of the selected ghost originators.
    // It's not sufficient to simply keep the starIDs of the selected stars, because the coordinates
    // of the selected stars may change due to aberration, or the magnitude may change due to variability.
    // We don't want to apply such changes to the original database of stars.

    for (auto const& star: starDB)
    {
        unsigned int starID = star.first;
        double RA, dec, Vmag;
        tie(RA, dec, Vmag) = star.second;
        double angularDistances = angularDistanceBetween(RACircleCenter, decCircleCenter, RA, dec, Angle::radians);  // [radians]

        if (angularDistances <= radiusCircle)
        {
            selectedGhostOrigID.push_back(starID);
            selectedGhostOrigRA.push_back(RA);
            selectedGhostOrigDec.push_back(dec);
            selectedGhostOrigVmag.push_back(Vmag);

            // Also keep track of which selected stars are variable. Saves us many search loops afterwards.
            // selectedVariableStars contains the _indices_ (of selected*) of those stars that are variable.

            if (deltaMagnitude.find(starID) != deltaMagnitude.end())
            {
                selectedVariableGhostOrigs.push_back(selectedGhostOrigID.size()-1);
            }
        }
    }

    return selectedGhostOrigID.size();
}













/**
 * \brief Calculate the apparent positions of the previously selected stars
 * based on the current platform pointing coordinates.
 *
 * \detail Important: First call selectStarsWithinRadiusFrom() to get a
selection of stars, so that the
 *                    aberration does not need to be done on the entire
database.
 *
 * This calculation is an approximation based on the position around the sun as
 * given by the input  orbit file. We also do calculate the differential
 * aberration which takes into account the aberration correction done for the
 * spacecraft pointing.
 * The position of the spacecraft is obtained from an orbit file and these
 * positions are stored in the orbitDB variable. This is a vector that
 * contains the time, velocity direction and speed of the spacecraft for every
 * timepoint needed in this simulation.
 *
 *
 * \param platform     Current platform from which the position of the Sun
 *                     and the pointing coordinates are requested,
 * \param aberrationCorrectionType   Type of the aberattion that we are
 *                     moddeling. This can be either differential or absolute.
 * \param startTime    Start time of the exposure [s]
 *                     (exposure number 0 is at startTime = 0)
 * 
 * \return Star catalogue with all the aberration corrected stars.
 */
void Sky::aberrateSelectedStarPositions(Platform &platform, string aberrationCorrectionType, double startTime)
{
    aberrateSelectedPositions(platform, selectedStarID, selectedRA, selectedDec, aberrationCorrectionType, startTime);
}










/**
 * \brief Calculate the apparent positions of the selected "items" (stars or
ghost originators)  based on the current platform
 *        pointing coordinates.
 *
 * \detail Important: First call selectGhostOrigsWithinRadiusFrom() or
selectStarsWithinRadiusFrom() to get a
 * selection of stars, so that the aberration does not need to be done on the
entire database.
 *
 * This calculation is an approximation based on the position around the sun as
 * given by the input  orbit file. We also do calculate the differential
 * aberration which takes into account the aberration correction done for the
 * spacecraft pointing.
 * The position of the spacecraft is obtained from an orbit file and these
 * positions are stored in the orbitDB variable. This is a vector that
 * contains the time, velocity direction and speed of the spacecraft for every
 * timepoint needed in this simulation.
 *
 * \param platform     Current platform from which the position of the Sun and
the
 *                     pointing coordinates are requested,
 * \param aberrationCorrectionType   Type of the aberattion that we are
 *                     moddeling. This can be either differential or absolute.
 * \param startTime    Start time of the exposure [s],
 *                     (exposure number 0 is at startTime = 0)
 * \param selectedID   ID of the star/ghosts that we want to aberrate
 * \param RA           RA of the star/ghosts that we want to aberrate
 * \param Dec          Dec of the star/ghosts that we want to aberrate
 *
 * \return Star catalogue with all the aberration corrected stars.
 */
void Sky::aberrateSelectedPositions(Platform &platform, vector<unsigned int> &selectedID, vector<double> &RA, vector<double> &Dec, string aberrationCorrectionType, double startTime)
{

    using StringUtilities::dtos;

    valarray<double> v = std::get<1>(orbitDB.at(0));
    double speed = std::get<2>(orbitDB.at(0));

    for (unsigned int i = 0; i < orbitDB.size(); i++)
    {
        if (std::get<0>(orbitDB.at(i)) <= orbitStartTime + startTime)
        {
            speed = std::get<2>(orbitDB.at(i));
            v = std::get<1>(orbitDB.at(i));
        }
    }

    //rotation matrix to compensate the aberration of light for the pointing direction, needed to calculate the differential aberration

    valarray<double> rot0 = {1., 0., 0.};
    valarray<double> rot1 = {0., 1., 0.};
    valarray<double> rot2 = {0., 0., 1.};

    //ratio of the velocity of PLATO to the speed of light

    double beta = speed / 300000.;

    if (aberrationCorrectionType == "differential")
    {
        Log.info("StarCatalog::aberrate: applying differential aberration correction");

        // Request the current platform pointing coordinates (i.e. pointing of the Fast Camera's)

        double raPlatform, decPlatform;
        tie(raPlatform, decPlatform) = platform.getCurrentPointingCoordinates();

        double lambdaPlatform, betaPlatform;
        equatorial2ecliptic(raPlatform, decPlatform, lambdaPlatform, betaPlatform);

        //direction of the pointing

        valarray<double> p = {cos(lambdaPlatform) * cos(betaPlatform), sin(lambdaPlatform) * cos(betaPlatform), sin(betaPlatform)};

        //angle between velocity direction and pointing

        double pangle = acos((v * p).sum());

        //relativistically aberrated angle between velocity direction and pointing

        double oangle = atan2(sqrt(1. - beta * beta) * sin(pangle), cos(pangle) + beta);

        //rotation axis between velocity direction and pointing

        valarray<double> r = {p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]};
        r /= sqrt((r * r).sum());

        //rotation matrix for rotation axis r with angle difference after aberration, this reverses the aberration effect for the pointing direction

        double c = cos(oangle - pangle);
        double s = sin(oangle - pangle);
        double x = r[0], y = r[1], z = r[2];
        rot0 = {c + x * x * (1. - c), x * y * (1. - c) - z * s, x * z * (1. - c) + y * s};
        rot1 = {y * x * (1. - c) + z * s, c + y * y * (1. - c), y * z * (1. - c) - x * s};
        rot2 = {z * x * (1. - c) - y * s, z * y * (1. - c) + x * s, c + z * z * (1. - c)};

    }
    else
    {
        Log.info("StarCatalog::aberrate: applying absolute aberration correction");
    }

    for (unsigned int n = 0; n < selectedID.size(); ++n)
    {
        double raStar, decStar, Vmag;
        tie(raStar, decStar, Vmag) = starDB[selectedID[n]];       // ra & dec in [rad]

        double lambdaStar, betaStar;
        equatorial2ecliptic(raStar, decStar, lambdaStar, betaStar);

        //direction of the star

        valarray<double> s = {cos(lambdaStar) * cos(betaStar), sin(lambdaStar) * cos(betaStar), sin(betaStar)};

        //angle between velocity direction and star direction

        double sangle = acos((v * s).sum());

        //relativistically aberrated angle between velocity direction and star direction

        double oangle = atan2(sqrt(1. - beta * beta) * sin(sangle), cos(sangle) + beta);

        //relativistically aberrated star direction

        valarray<double> a = s - v * cos(sangle);
        a = v * cos(oangle) + a / sqrt((a * a).sum()) * sin(oangle);

        //rotate aberrated star direction to compensate for aberrated pointing to get the differential aberrated star direction

        a = {(rot0 * a).sum(), (rot1 * a).sum(), (rot2 * a).sum()};

        //calculate ecliptic coordinates of aberrated star direction

        betaStar = atan(a[2] / sqrt(a[0] * a[0] + a[1] * a[1]));
        lambdaStar = atan2(a[1], a[0]);

        double raStarAberrated, decStarAberrated;
        ecliptic2equatorial(lambdaStar, betaStar, raStarAberrated, decStarAberrated);

        RA[n]  = raStarAberrated;
        Dec[n] = decStarAberrated;

        // Write debugging info on the first star only

        if (n == 0)
        {
            Log.debug("StarCatalog::aberrate: ra[0], dec[0] = " + dtos(raStarAberrated, false, 8) + ", " + dtos(decStarAberrated, false, 8));
        }
    }
  
}












/**
 * \brief Calculate the apparent positions of the previously selected ghost
originators based on the current platform
 *        pointing coordinates.
 *
 * \detail Important: First call selectGhostOrigsWithinRadiusFrom() to get a
selection of stars, so that the
 *                    aberration does not need to be done on the entire
database.
 *
 * This calculation is an approximation based on the position around the sun as
 * given by the input  orbit file. We also do calculate the differential
 * aberration which takes into account the aberration correction done for the
 * spacecraft pointing.
 * The position of the spacecraft is obtained from an orbit file and these
 * positions are stored in the orbitDB variable. This is a vector that
 * contains the time, velocity direction and speed of the spacecraft for every
 * timepoint needed in this simulation.
 *
 * \param platform     Current platform from which the position of the Sun and
the
 *                     pointing coordinates are requested,
 * \param aberrationCorrectionType   Type of the aberattion that we are
 *                     moddeling. This can be either differential or absolute.
 * \param startTime    Start time of the exposure [s],
 *                     (exposure number 0 is at startTime = 0)
 *
 * \return Star catalogue with all the aberration corrected stars.
 */
void Sky::aberrateSelectedGhostOrigPositions(Platform &platform, string aberrationCorrectionType, double startTime)
{
    aberrateSelectedPositions(platform, selectedGhostOrigID, selectedGhostOrigRA, selectedGhostOrigDec, aberrationCorrectionType, startTime);
}









/**
 * \brief Return the star ID, RA, Dec, and Vmag of selected star #n
 *
 * \detail Important: first call selectStarsWithinRadiusFrom() to get a proper selection of stars.
 *
 * \param n: 0 <= n < number of selected stars
 *
 * \return starID: Identification number of the selected star.
 *         RA:     Right ascension of the star. Aberrated if aberrateSelectedStarPositions() was called before.
 *         Dec:    Declination of the star. Aberrated if aberrateSelectedStarPositions() was called before.
 *         Vmag:   Johnson V magnitude. Possibly variable if updatedParameters() was called before.
 */
tuple<unsigned int, double, double, double> Sky::getSelectedStar(unsigned int n)
{
    if (n > selectedStarID.size()-1)
    {
        throw IllegalArgumentException("Sky::getSelectedStar(): star number is larger than " + to_string(selectedStarID.size()-1));
    }
    else
    {
        return make_tuple(selectedStarID[n], selectedRA[n], selectedDec[n], selectedVmag[n]);
    }
}

/**
 * \brief Return the star ID, RA, Dec, and Vmag of selected ghost originator #n
 *
 * \detail Important: first call selectStarsWithinRadiusFrom() to get a proper selection of stars.
 *
 * \param n: 0 <= n < number of selected stars
 *
 * \return starID: Identification number of the selected star.
 *         RA:     Right ascension of the star. Aberrated if aberrateSelectedStarPositions() was called before.
 *         Dec:    Declination of the star. Aberrated if aberrateSelectedStarPositions() was called before.
 *         Vmag:   Johnson V magnitude. Possibly variable if updatedParameters() was called before.
 */
tuple<unsigned int, double, double, double> Sky::getSelectedGhostOrig(unsigned int n)
{
    if (n > selectedGhostOrigID.size()-1)
    {
        throw IllegalArgumentException("Sky::getSelectedGhostOrig(): ghost originator number is larger than " + to_string(selectedGhostOrigID.size()-1));
    }
    else
    {
        return make_tuple(selectedGhostOrigID[n], selectedGhostOrigRA[n], selectedGhostOrigDec[n], selectedGhostOrigVmag[n]);
    }
}








/**
 * \brief Return the RA [rad], dec [rad], and Vmag of the star with the given starID.
 *
 * \detail If the starID is unknown, an IllegalArgumentException will be thrown.
 *
 * \return RA, dec, Vmag
 *
 */

tuple<double, double, double> Sky::getInfoOfStarWithID(unsigned int starID)
{
    if (starDB.count(starID) == 0)
    {
        if (starID == 0)
        {
            return {0.0, 0.0, 0.0};
        }
        else
        {
            throw IllegalArgumentException(
                "Sky::GetInfoOfSelectedStarWithID(): starID " +
                to_string(starID) + " unknown");
        }
    }
    else
    {
        return starDB[starID];
    }
}









/**
 * \brief Return the solar radiant flux at the given wavelength, measured above the atmosphere of the earth.
 *
 * \details This function uses the Wehrli (1985) solar irradiance table
 *          plus linear interpolation. Note that the units of the tabulated
 *          data radiant fluxes are \f$ J s^{-1} m^{-2} (nm)^{-1} \f$, where nm
 *          is nanometer (unit of wavelength) while the units of the radiant
 *          flux that this function returns is SI: \f$ J s^{-1} m^{-2} m^{-1} \f$
 *
 * \param lambda  Wavelength [m], should be in [199.5 nm, 100075 nm]
 *
 * \return solar radiant flux at air mass zero [\f$J s^{-1} m^{-2} m^{-1}\f$]
 */

double Sky::solarRadiantFlux(double lambda)
{
    int i;         // Index of the first point, defining the linear relation
    double result; // solar radiant flux in SI units.

    // The tabulated wavelengths are in nm, so we have to convert from [m] to [nm]

    double lam = lambda * 1.0e+9;

    // Find the location of the wavelength point lam in the wavelength
    // array skydata::sunwavel[]

    locate(lam, skydata::sunwavel, 920, i);

    // Check if the location was found

    if (i == -1)
    {
        Log.error("Sky::solarRadiantFlux()): no data for the given wavelength");
        exit(1);
    }

    // Do the linear interpolation

    result = skydata::sunflux[i]
                + (skydata::sunflux[i+1] - skydata::sunflux[i])
                / (skydata::sunwavel[i+1] - skydata::sunwavel[i])
                * (lam - skydata::sunwavel[i]);

    // The tabulated flux data are per nm (unit of wavelength), so we should
    // convert to flux per m.

    return result * 1.0e9;             // conversion: (nm)^{-1} -> m^{-1}
}














/**
 * \brief Computes the solar radiant flux between the wavelengths lambda1 and lambda2.
 *
 * \details This function uses the Wehrli (1985) solar irradiance table
 *          plus the SolarRadiantFlux(lambda) function.
 *          . lambda1 and lambda2 should be between 199.5 nm and 10075 nm.
 *          . This function is overloaded.
 *
 * \param lambda1  Lower wavelength [m] of the interval, should be in [199.5 nm, 100075 nm]
 * \param lambda2  Upper wavelength [m] of the interval, should be in [199.5 nm, 100075 nm]
 *
 * \return Integrated solar radiant flux [\f$J s^{-1} m^{-2}\f$]
 */

double Sky::solarRadiantFlux(double lambda1, double lambda2)
{
    double lam1;
    double lam2;

    // Check if the wavelengths are within the table boundaries.

    if ((lambda1 >= 199.5e-9) && (lambda1 <= 10075.0e-9) && (lambda2 >= 199.5e-9) && (lambda2 <= 10075.0e-9))
    {
        // If the integral boundaries are equal, the integral is zero

        if (lambda1 == lambda2)
        {
            return (0.0);
        }

        // Check if the first wavelength is indeed greater than the second

        if (lambda1 < lambda2)
        {
            lam1 = lambda1;
            lam2 = lambda2;
        }
        else
        {
            lam1 = lambda2;
            lam2 = lambda1;
        }

        // Integrate with the trapezium method (Numerical Recipes, pg. 137)

        const int JMAX = 30;
        const double EPS = 1.0e-5;
        double x, tnm, sum, del, olds;
        double s = 0.0;
        int it, j, k;

        olds = -1.0e30;

        for (j = 1; j <= JMAX; j++)
        {
            if (j == 1)
            {
                s = 0.5 * (lam2 - lam1) * (solarRadiantFlux(lam1) + solarRadiantFlux(lam2));
            }
            else
            {
                for (it = 1, k = 1; k < j - 1; k++) it <<= 1;
                tnm = it;
                del = (lam2 - lam1) / tnm;
                x = lam1 + 0.5 * del;
                for (sum = 0.0, k = 1; k <= it; k++, x += del)
                {
                    sum += solarRadiantFlux(x);
                }
                s = 0.5 * (s + (lam2 - lam1) * sum / tnm);
            }

            if (j > 5)
            {
                if (fabs(s - olds) < EPS * fabs(olds) || (s == 0.0 && olds == 0.0))
                return s;
            }

            olds = s;
        }

        Log.error("Sky::solarRadiantFlux(): Integration not converged");
        exit (1);
    }
    else
    {
        // This is the case that the given wavelengths were not between the table boundaries.

        Log.error("Sky::solarRadiantFlux(): wavelength must be in [199.5e-9, 10075.0e-9]");
        exit (1);
    }
}











/**
 * \brief Compute the Solar radiant flux in the given passband
 *
 * \details This function uses the Wehrli (1985) solar irradiance table
 *          plus the SolarRadiantFlux(lambda) function.
 *
 * \param lambda     Ordered wavelengths of the passband [m], should be in [199.5 nm, 100075 nm]
 * \param throughput Throughput of the passband
 *
 * \return Solar radiant flux  [\f$J s^{-1} m^{-2}\f$]
 */

double Sky::solarRadiantFlux(vector<double> &lambda, vector<double> &throughput)
{
    const double lambda1 = lambda[0];
    const double lambda2 = lambda[lambda.size()-1];

    // Check if the passband wavelengths are within [199.5, 10075] nm.

    if ((lambda1 < 199.5e-9) || (lambda1 > 10075.0e-9) || (lambda2 < 199.5e-9) || (lambda2 > 10075.0e-9))
    {
        Log.error("Sky::solarRadiantFlux(): Passband wavelengths not in [199.5, 10075] nm.");
        exit(1);
    }

  // Build up the function you want to integrate

   integrand.clear();
   integrand.resize(lambda.size());

   for (unsigned int i = 0; i < lambda.size(); i++)
   {
      integrand[i] = solarRadiantFlux(lambda[i]) * throughput[i];
   }

   tabfunction.init(lambda, integrand, lambda.size());
   tabfunction.setInterpolationMethod(Linear_Interpolation);

   // Integrate over the throughput band

   const double integral = tabfunction.integrate(lambda[0], lambda[lambda.size()-1]);

   return integral;

}










/**
 * \brief Compute the zodiacal background flux in the wavelength interval [lambda1, lambda2], for a given position in the sky.
 *
 * \note Some parts of the sky cannot be sampled!
 *
 * \param alpha    Right ascension coordinate  [rad]
 * \param delta    Declination coordinate      [rad]
 * \param lambda1  Lower wavelength of the interval [m]
 * \param lambda2  Upper wavelength of the interval [m]
 *
 * \return Zodiacal flux [\f$J s^{-1} m^{-2} sr^{-1}\f$]
 */

double Sky::zodiacalFlux(double alpha, double delta, double lambda1, double lambda2)
{
    double lam, beta;
    double flux500;
    int lam_index, beta_index;

    // Convert the equatorial coordinates to geocentric ecliptic coordinates.
    // All coordinates are in radians.

    auto skyPoint = SkyCoordinates(alpha, delta, Angle::radians);
    tie(lam, beta) = skyPoint.toEcliptic(Angle::radians);

    // The zodiacal light is approximately symmetric with respect to the
    // ecliptic, and with respect to the helioecliptic meridian
    // (= sun-ecliptic poles-antisolar point).

    beta = fabs (beta);
    if (lam > Constants::PI) lam = 2.0 * Constants::PI - lam;

    // Convert from radians to degrees

    beta = rad2deg(beta);
    lam  = rad2deg(lam);

    // Locate the coordinates lam & beta in the coordinate arrays
    // Check if the coordinates are out of boundary.

    locate(lam, skydata::zodlong, 19, lam_index);
    locate(beta,skydata::zodlat,  11, beta_index);

    if ((lam_index == -1) || (beta_index == -1))
    {
        string position = "(" + to_string(rad2deg(alpha)) + ", " + to_string(rad2deg(delta)) + ")";
        Log.warning("Sky::zodiacalFlux(): No data for (alpha, delta) = " + position);
        return 0.0;
    }

    // Do a bilinear interpolation

    double dx = (skydata::zodlat[beta_index+1] - beta) / (skydata::zodlat[beta_index+1] - skydata::zodlat[beta_index]);
    double dy = (skydata::zodlong[lam_index+1] - lam) / (skydata::zodlong[lam_index+1] - skydata::zodlong[lam_index]);
    double P1 = skydata::zod[lam_index][beta_index];
    double P2 = skydata::zod[lam_index][beta_index+1];
    double P3 = skydata::zod[lam_index+1][beta_index];
    double P4 = skydata::zod[lam_index+1][beta_index+1];

    flux500 =  dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;

    // The tabulated fluxes are given in:
    //    10^{-8} J s^{-1} m^{-2} sr^{-1} (micrometer)^{-1}
    // so we need to convert to:
    //    J s^{-1} m^{-2} sr^{-1} m^{-1}

    flux500 *= 0.01;

    // Now we have the zodiacal light at 500 nm, from which we need to
    // derive the zodiacal light flux in the interval [lambda1, lambda2].
    // For this we use the fact that the zodiacal flux has a solar
    // wavelength dependence.
    return (flux500 * solarRadiantFlux(lambda1, lambda2) / solarRadiantFlux(500e-9));
}













/**
 * \brief Compute the zodiacal background flux in the given passband, for a given position in the sky.
 *
 * \note Some parts of the sky cannot be sampled!
 *
 * \param alpha       Right ascension coordinate [rad]
 * \param delta       Declination coordinate [rad]
 * \param lambda      Wavelengths of the passband [m]
 * \param throughput  Throughput of the passband
 *
 * \return  Zodiacal flux [\f$J s^{-1} m^{-2} sr^{-1}\f$]
 */

double Sky::zodiacalFlux(double alpha, double delta, vector<double> &lambda, vector<double> &throughput)
{
    double lam, beta;
    double flux500;
    int lam_index, beta_index;


    // Convert the equatorial coordinates to geocentric ecliptic coordinates.
    // All coordinates are in radians.

    auto skyPoint = SkyCoordinates(alpha, delta, Angle::radians);
    tie(lam, beta) = skyPoint.toEcliptic(Angle::radians);

    // The zodiacal light is approximately symmetric with respect to the
    // ecliptic, and with respect to the helioecliptic meridian (= sun-ecliptic poles-antisolar point).

    beta = fabs(beta);
    if (lam > Constants::PI) lam = 2.0 * Constants::PI - lam;

    // Convert from radians to degrees

    beta = rad2deg(beta);
    lam  = rad2deg(lam);

    // Locate the coordinates lam & beta in the coordinate arrays
    // Check if the coordinates are out of boundary.

    locate(lam,  skydata::zodlong, 19, lam_index);
    locate(beta, skydata::zodlat,  11, beta_index);


    if ((lam_index == -1) || (beta_index == -1))
    {
        string position = "(" + to_string(rad2deg(alpha)) + ", " + to_string(rad2deg(delta)) + ")";
        Log.warning("Sky::zodiacalFlux(): No data for (alpha, delta) = " + position);
        return 0.0;
    }

    // Do a bilinear interpolation

    double dx = (skydata::zodlat[beta_index+1] - beta) / (skydata::zodlat[beta_index+1] - skydata::zodlat[beta_index]);
    double dy = (skydata::zodlong[lam_index+1] - lam) / (skydata::zodlong[lam_index+1] - skydata::zodlong[lam_index]);
    double P1 = skydata::zod[lam_index][beta_index];
    double P2 = skydata::zod[lam_index][beta_index+1];
    double P3 = skydata::zod[lam_index+1][beta_index];
    double P4 = skydata::zod[lam_index+1][beta_index+1];

    flux500 =  dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;

    // The tabulated fluxes are given in:
    //    10^{-8} J s^{-1} m^{-2} sr^{-1} (micrometer)^{-1}
    // so we need to convert to:
    //    J s^{-1} m^{-2} sr^{-1} m^{-1}

    flux500 *= 0.01;

    // Now we have the zodiacal light at 500 nm, from which we need to
    // derive the zodiacal light flux in the passband 'throughput'.
    // For this we use the fact that the zodiacal flux has a solar
    // wavelength dependence.

    return (flux500 * solarRadiantFlux(lambda, throughput) / solarRadiantFlux(500e-9));

}












/**
 * \brief Return the Stellar background (unresolved stars + diffuse galactic background + extragalactic background)
 *        for the given equatorial coordinates, in the wavelength interval [lambda1, lambda2].
 *
 * \details Tabulated values of the Pioneer 10 blue passband (spectral range: [395, 495] nm),
 *          and the Pioneer 10 red passband (spectral range: [590, 690] nm) are used. The value
 *          in the given interval will be computer by (crude!) linear inter- and extrapolation which.
 *          Care is taken that in the Pioneer 10 passbands, the interpolation returns exactly the
 *          tabulated values.
 *
 * \note Some parts of the sky cannot be sampled!
 *
 * \param RA       Equatorial coordinate: right ascension [rad]
 * \param dec      Equatorial coordinate: declination [rad]
 * \param lambda1  Begin wavelength of the interval [m]
 * \param lambda2  End   wavelength of the interval [m]
 *
 * \return Stellar background flux in the Pioneer 10 blue/red passband [\f$J s^{-1} m^{-2} sr^{-1}\f$]
 */

double Sky::stellarBackgroundFlux(double RA, double dec, double lambda1, double lambda2)
{
    double alpha, delta;
    int alpha_index, delta_index;
    double blueflux, redflux;
    double a, b;

    // Convert from radians to degrees

    alpha = rad2deg(RA);
    delta = rad2deg(dec);

    // Locate the coordinates alpha & delta in the coordinate arrays
    // Check if the coordinates are out of boundary.

    locate(alpha, skydata::skyRA, 37, alpha_index);
    locate(delta, skydata::skydec, 27, delta_index);

    if ((alpha_index == -1) || (delta_index == -1))
    {
        string skyPosition = "(" + to_string(alpha) + ", " + to_string(delta) + ") deg";
        Log.warning("Sky::stellarBgFlux(): No data for (alpha, delta) = " + skyPosition);
        return 0.0;
    }


    // Do a bilinear interpolation for both the blue and the red passband

    double dx = (skydata::skydec[delta_index+1] - delta) / (skydata::skydec[delta_index+1] - skydata::skydec[delta_index]);
    double dy = (skydata::skyRA[alpha_index+1] - alpha) / (skydata::skyRA[alpha_index+1] - skydata::skyRA[alpha_index]);

    double P1 = skydata::skyblue[alpha_index][delta_index];
    double P2 = skydata::skyblue[alpha_index][delta_index+1];
    double P3 = skydata::skyblue[alpha_index+1][delta_index];
    double P4 = skydata::skyblue[alpha_index+1][delta_index+1];

    blueflux =   dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;

    P1 = skydata::skyred[alpha_index][delta_index];
    P2 = skydata::skyred[alpha_index][delta_index+1];
    P3 = skydata::skyred[alpha_index+1][delta_index];
    P4 = skydata::skyred[alpha_index+1][delta_index+1];

    redflux =   dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;


    // Convert the result from S10sun units (brightness of 10th magnitude
    // solar type stars per degree square) to SI units:
    //   J s^{-1} m^{-2} sr^{-1}

    blueflux *= 1.2084e-9;
    redflux  *= 1.0757e-9;


    // Do an extra/interpolation to the user given wavelength band
    // Note: Our 'a' is in the formulae 'a/2'.

    const double R1 = 590.0e-9;   // Lower wavelength of Pioneer 10 Red Passb.
    const double R2 = 690.0e-9;   // Upper wavelength of Pioneer 10 Red Passb.
    const double B1 = 395.0e-9;   // Lower wavelength of Pioneer 10 Blue Passb.
    const double B2 = 495.0e-9;   // Upper wavelength of Pioneer 10 Blue Passb.

    a =  (redflux * (B2 - B1) - blueflux * (R2 - R1))
            / ((R2 * R2 - R1 * R1) * (B2 - B1) - (B2 * B2 - B1 * B1) * (R2 - R1));

    b =  (blueflux * (R2 * R2 - R1 * R1) - redflux * (B2 * B2 - B1 * B1))
            / ((R2 * R2 - R1 * R1) * (B2 - B1) - (B2 * B2 - B1 * B1) * (R2 - R1));


    return (a * (lambda2*lambda2 - lambda1*lambda1) + b * (lambda2 - lambda1));
}














/**
 * \brief Approximate the stellar background (unresolved stars + diffuse galactic background
 *       + extragalactic background) for the given equatorial coordinates, in the given passband.
 * 
 * \details Tabulated values for the Pioneer 10 blue passband (spectral range: [395, 495] nm),
 *          and for the Pioneer 10 red passband (spectral range: [590, 690] nm) are used.
 *          For lambda <= 690 nm, first the monochromatic flux function is approximated as a
 *          linear function (increasing with wavelength because there is more red than blue
 *          background light). Care is taken that the approximated flux is never negative.
 *          For lambda >= 690 nm, the monochromatic flux is approximated as a constant function
 *          with the value of the flux at lambda = 690 nm. Care is taken that in the Pioneer 10
 *          passbands, the interpolation returns exactly the tabulated values.
 * 
 * \note Some parts of the sky cannot be sampled!
 *
 * \param RA          Equatorial coordinate: right ascension [radians]
 * \param dec         Equatorial coordinate: declination [radians]
 * \param lambda      Wavelength values of the passband  [m]
 * \param throughput  Throughput of the passband
 *
 * \return  Stellar background flux [\f$J s^{-1} m^{-2} sr^{-1}\f$]
 */

double Sky::stellarBackgroundFlux (double RA, double dec, vector<double> &lambda, vector<double> &throughput)
{
    int alpha_index, delta_index;
    double blueflux, redflux;
    double a, b;

    // Convert from radians to degrees

    double alpha = rad2deg(RA);
    double delta = rad2deg(dec);

    // Locate the coordinates alpha & delta in the coordinate arrays
    // Check if the coordinates are out of boundary.

    locate(alpha, skydata::skyRA, 37, alpha_index);
    locate(delta, skydata::skydec, 27, delta_index);

    if ((alpha_index == -1) || (delta_index == -1))
    {
        string skyPosition = "(" + to_string(alpha) + ", " + to_string(delta) + ") deg";
        Log.warning("Sky::stellarBgFlux(): No data for (alpha, delta) = " + skyPosition);
        return 0.0;
    }


    // Do a bilinear interpolation for both the blue and the red passband

    double dx = (skydata::skydec[delta_index+1] - delta) / (skydata::skydec[delta_index+1] - skydata::skydec[delta_index]);
    double dy = (skydata::skyRA[alpha_index+1] - alpha) / (skydata::skyRA[alpha_index+1] - skydata::skyRA[alpha_index]);

    double P1 = skydata::skyblue[alpha_index][delta_index];
    double P2 = skydata::skyblue[alpha_index][delta_index+1];
    double P3 = skydata::skyblue[alpha_index+1][delta_index];
    double P4 = skydata::skyblue[alpha_index+1][delta_index+1];

    blueflux =   dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;

    P1 = skydata::skyred[alpha_index][delta_index];
    P2 = skydata::skyred[alpha_index][delta_index+1];
    P3 = skydata::skyred[alpha_index+1][delta_index];
    P4 = skydata::skyred[alpha_index+1][delta_index+1];

    redflux =   dx * dy * P1 + (1.0 - dx) * dy * P2 + dx * (1.0 - dy) * P3  + (1.0 - dx) * (1.0 - dy) * P4;


    // Convert the result from S10sun units (brightness of 10th magnitude
    // solar type stars per degree square) to SI units:
    //   J s^{-1} m^{-2} sr^{-1}

    blueflux *= 1.2084e-9;
    redflux  *= 1.0757e-9;

    // Compute the linear relation f(lambda) = a * lambda + b
    // to be used for lambda <= 690 nm.

    const double R1 = 590.0e-9;   // Lower wavelength of Pioneer 10 Red Passb.
    const double R2 = 690.0e-9;   // Upper wavelength of Pioneer 10 Red Passb.
    const double B1 = 395.0e-9;   // Lower wavelength of Pioneer 10 Blue Passb.
    const double B2 = 495.0e-9;   // Upper wavelength of Pioneer 10 Blue Passb.

    a =  2.0 * (redflux * (B2 - B1) - blueflux * (R2 - R1))
        / ((R2 * R2 - R1 * R1) * (B2 - B1) - (B2 * B2 - B1 * B1) * (R2 - R1));

    b =  (blueflux * (R2 * R2 - R1 * R1) - redflux * (B2 * B2 - B1 * B1))
        / ((R2 * R2 - R1 * R1) * (B2 - B1) - (B2 * B2 - B1 * B1) * (R2 - R1));


    // Check if the monochromatic flux is indeed increasing for lambda <= 690 nm.

    if (a <= 0.0)
    {
        Log.warning("Sky::stellarBgFlux(): Bad behaviour of monochromatic background flux for lambda <= 690 nm.");
    }

    // Compute where this linear function is zero. Useful to know, because
    // we don't want negative sky background fluxes.

    double root = - b / a;

    // Set up the integrand

    integrand.clear();
    integrand.resize (lambda.size());

    for (unsigned int i = 0; i < lambda.size(); i++)
    {
        if (lambda[i] <= root)
        {
            integrand[i] = 0.0;
        }

        if ((lambda[i] > root) && (lambda[i] <= R2))
        {
            integrand[i] = (a * lambda[i] + b) * throughput[i];
        }

        if (lambda[i] > R2)
        {
            integrand[i] = (a * R2 + b) * throughput[i];
        }
    }

    // Return the integral

    tabfunction.init(lambda, integrand, lambda.size());
    tabfunction.setInterpolationMethod(Linear_Interpolation);

    return (tabfunction.integrate(lambda[0], lambda[lambda.size()-1]));
}













/**
 * \brief Given an array[0..N-1] of strictly ascending values, and a value x,
 *        return an index so that array[index] <= x <= array[index+1]
 *        If the value x is out of the array boundaries, index will be set to -1.
 *
 * \note NO error trapping!
 *
 * \param x
 * \param array
 * \param N
 * \param index
 */

void Sky::locate(double x, const double *array, int N, int &index)
{
    int index1, index2;

    // Check if x is out of the tabulated range

    if ((x < array[0]) || (x > array[N-1]))
    {
        index = -1;
        return;
    }

    // Check if x happens to be the first element of the array

    if (x == array[0])
    {
        index = 0;
    }

    // Check if x happens to be the last element of the array

    if (x == array[N-1])
    {
        index = N - 2;
    }

    // Find the location with bisection

    index1 = 0;                       // We already checked the lower and upper
    index2 = N - 1;                   // borders.

    unsigned int middle;               // Middle point

    while (index2 - index1 > 1)
    {
        middle = (index2 + index1) >> 1;

        if (x >= array[middle])
        {
            index1 = middle;
        }
        else
        {
            index2 = middle;
        }
    }

    index = index1;
}



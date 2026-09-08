#include "StrayLight.h"
#include "ConfigurationParameters.h"
#include "Constants.h"
#include "armadillo"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


double operator-(Time t1, Time t2);
double operator-(Time t1, Time t2) { return difftime(t1.t, t2.t); }


std::ostream& operator<<(std::ostream& os, const Time &obj)
{
    os << obj.totalSeconds();
    return os;
}




/**
 * \brief: Constructor for StrayLight class.
 *
 * \param: configParam   Configuration parameters for the detector.
 * \param: hdf5file      HDF5 file to write to.
 * \param: camera        Camera on which the straylight will fall.
 *
 */
StrayLight::StrayLight(ConfigurationParameters &configParam, HDF5File &hdf5File,
                       Camera &camera)
  : HDF5Writer(hdf5File), camera(camera)
{
    // Parse the parameters from the configuration file.

    configure(configParam);
}





/**
 * \brief: Configure the StrayLight object using the configuration parameters.
 *
 * \param: configParam      Configuration parameters.
 *
 */
void StrayLight::configure(ConfigurationParameters &configParam)
{
    numExposure         = configParam.getInteger("ObservingParameters/NumExposures");
    beginExposures      =
        configParam.getInteger("ObservingParameters/BeginExposureNr");
    cycleTime           = configParam.getDouble("ObservingParameters/CycleTime");
    pixelSize           = configParam.getDouble("CCD/PixelSize") * 1e-6;  // [m]

    // Get the coordinates of the telescope reference frame

    arma::vec nx = {1, 0, 0};
    arma::vec ny = {0, 1, 0};
    arma::vec nz = {0, 0, 1};

    telescopeAxis.col(0) = camera.telescopeToSkyCoordinates(nx);
    telescopeAxis.col(1) = camera.telescopeToSkyCoordinates(ny);
    telescopeAxis.col(2) = camera.telescopeToSkyCoordinates(nz);

    // Get the time in the orbit file that corresponds to exposure number 0.

    time0 = configParam.getString("Sky/StrayLight/Time0");

        // Read in the positions of the file and save them into the vectors.

    std::string orbitPath =
        configParam.getAbsoluteFilename("Sky/StrayLight/FilePath");
    readInFile(orbitPath, sc_positions, moon_positions, sun_positions);

    // Read in the PSTRadiance file and save it into vectors.

    std::string pstRadiancePath = configParam.getAbsoluteFilename("Sky/StrayLight/PstRadiancePath");
    getPSTRadiance(pstRadiancePath);

    // TODO: We should read this in from the input file once we add the option
    // to get straylight from earth.
    double radiusMoon = 1.7381e6; // [m]
    double moon_reflectivity = 0.11;

    // Create the two celestial objects
    moon.radius = radiusMoon;
    moon.reflectivity = moon_reflectivity;
}





/*
 * \brief: Read in the orbit path and extract the positions of the celestial
objects that we will need during the simulation.
 *
 * \param: orbit path: filename of the orbit path that we will read in
 * \param: sc_position: vector that we will use to store the position of the
 *                      spacecraft.
 * \param: moon_position: vector that we will use to store the position of the
 *                        spacecraft.
 * \param: sun_position: vector that we will use to store the position of the
 *                       sun.
 */
void StrayLight::readInFile(std::string orbitPath, std::vector<arma::vec> &sc_position,
                            std::vector<arma::vec> &moon_position,
                            std::vector<arma::vec> &sun_position)
{
    // begin time from which we want to start extracting
    Time t0 = Time(time0);

    // get the lower and upper time between which we want to extract data [starting from t0]
    double lower_bound = cycleTime * beginExposures;
    double upper_bound = cycleTime * numExposure + lower_bound;

    std::ifstream orbitFile(orbitPath);
    if (orbitFile.is_open())
    {
        std::string line;
        while (getline(orbitFile, line))
        {
            // Skip empty lines

            if (line.size() == 0)
                continue;

            // Skip lines that only contain white space

            const std::string whitespace = " /t/r/n";
            if (line.find_first_not_of(whitespace) == std::string::npos)
                continue;

            // Skip header line starting with '#'.

            if (line[0] == '#')
                continue;

            std::istringstream buffer(line);
            std::vector<std::string> value_of_line = splitLine(line);

            Time t = Time(value_of_line[0]);

            // Skip points that are not part of the simulation.
            double time = t - t0;
            if (time < lower_bound)
                continue;

            // If this is a point that we do consider we extract the data
            // and add it to the vectors.

            arma::vec sc_row(3);
            sc_row[0] = std::stod(value_of_line[3]);
            sc_row[1] = std::stod(value_of_line[4]);
            sc_row[2] = std::stod(value_of_line[5]);

            arma::vec sun_row(3);
            sun_row[0] = std::stod(value_of_line[7]);
            sun_row[1] = std::stod(value_of_line[8]);
            sun_row[2] = std::stod(value_of_line[9]);

            arma::vec moon_row(3);
            moon_row[0] = std::stod(value_of_line[11]);
            moon_row[1] = std::stod(value_of_line[12]);
            moon_row[2] = std::stod(value_of_line[13]);

	        // We add these values and convert them from [km] to [m] units
            sc_position.push_back(sc_row * 1000);
            sun_position.push_back(sun_row * 1000);
            moon_position.push_back(moon_row * 1000);
            times.push_back(time);

            // We add one line outside our upper_bound and then we stop
            if (time > upper_bound)
                break;
        }
    }
}





/**
 * \brief: Reads in a line from CSV file and returns a vector with elements the
 *         comma seperated variables.
 *
 * \param: line   The line from the CSV file. 
 */
std::vector<std::string> StrayLight::splitLine(std::string &line)
{
    int i = 0;
    bool isString = false;
    std::vector<std::string> values;

    // Temporary string used to split string

    std::string s;
    while (line[i] != '\0')
    {
        if (isString)
        {
            if (line[i] == '\"')
            {
                isString = false;
            }
            else
            {
                s += line[i];
            }
        }
        else
        {
            if (line[i] == '\"')
            {
                isString = true;
            }
            else if (line[i] != ',')
            {
                // Append the char to the temp string
                s += line[i];
            }
            else
            {
                values.push_back(s);
                s.clear();
            }
        }
        i++;
    }
    values.push_back(s);
    return values;
}





/**
 *
 * \brief: reads in and interpolates the PST Radiance data as defined in the
 *         pstRadiancePath.
 *
 * \param: pstRadiancePath: path where we can find the pstRadiance
 *
 * \note: the term pstRadiance is here used to describe the wavelength
 *        integrated product of the pst with the spectral radiance of a
 *        BB radiator, devided by the photon energy. Since the pst is dependent
 *        on two incident angles (rho, az), so is this property.
 *
 * \note: We interpolate the pst over rho using a 1-D monotonic cubic interpolation.
 */
void StrayLight::getPSTRadiance(std::string pstRadiancePath)
{
    // Save file in datastructure
    if (!FileUtilities::fileExists(pstRadiancePath))
    {
        throw FileException("Straylight: trying to load the PST HDF5 file (" + pstRadiancePath + "), but file doesn't exist.");
    }
    try
    {
        pstFile.open(pstRadiancePath);
    }
    catch (H5::FileIException ex)
    {
        Log.error("H5::FileIException: " + string(ex.getCDetailMsg()));
        throw H5FileException("Straylight: Could not open HDF5 file: " + pstRadiancePath);
    }

    std::array<std::vector<double>, 5> pstRadianceValues;

    int idx = 0;
    for (int az : azs)
    {
        std::string group = "/az_" + std::to_string(az);
        vector<double> pst;
        vector<double> rho;

        if (pstFile.hasGroup(group))
        {

          // Read in the pst values 

          if (pstFile.hasDataset(group, "pst")) {pstFile.readArray(group, "pst", pst);}
          else
          {
            throw H5FileException("Straylight: PST file does have the dataset pst in group:  " + group);
          }

          // Read in the corresponding rho values

          if (pstFile.hasDataset(group, "rho")) {pstFile.readArray(group, "rho", rho);}
          else
          {
            throw H5FileException("Straylight: PST file does have the dataset rho in group:  " + group);
          }
        }
        pstRadianceValues[idx] = pst;
        rhoValues[idx] = rho;
        idx++;
    }

    // interpolate the PSTRadiance values between rhoValues.

      interpolatePSTRadiance(rhoValues, pstRadianceValues);
}





/**
 *
 * \brief: Get a 1-D monotonic cubic interpolation of the pstRadiance.
 *
 * \param: rhoValues: rho values where we know the pstRadiance values.
 * \param: pstRadianceValues: pstRadiance values that are known.
 *
 * \note: This function stores the 4 parameters that describe the cubic polynoom
 *        between rhoValues[i] and rhoValues[i+1] in parameters[i].
 */
void StrayLight::interpolatePSTRadiance(std::array<std::vector<double>, 5> rhoValues,
                                  std::array<std::vector<double>, 5> pstRadianceValues)
{

    for (int i = 0; i < 5; i++)
    {
        std::vector<double> PSTRadiance = pstRadianceValues[i];
        std::vector<double> rho = rhoValues[i];

        std::vector<std::array<double, 4>> cubicParameters;

        // Do a PCHIP 1-D monotonic cubic interpolation and add the 4
        // parameters (for cubic polynomial) into cubicParameters vector

        double d0 = 0; // Boundary condition of the derivative
                       // at start of interpolation

        for (int angle_idx = 0; angle_idx < rho.size() - 1; angle_idx++)
        {
            double alpha =
                double(
                       (rho[angle_idx + 1] - rho[angle_idx]) +
                       2 * (rho[angle_idx + 2] - rho[angle_idx + 1])) /
              (3 * (rho[angle_idx + 2] - rho[angle_idx]));

            // Boundary condition of the derivative at rho[angle_idx+1]
            double d1;

            // The boundary condition of the derivative at the end of the interpolation.
            if (angle_idx == rho.size() -1) {d1 = 0;}

            else if ((PSTRadiance[angle_idx] == PSTRadiance[angle_idx + 1]) &&
                     (PSTRadiance[angle_idx] == PSTRadiance[angle_idx + 2]))
            {
                d1 = 0;
            }
            else{
                d1 =
                    (PSTRadiance[angle_idx + 1] -
                     PSTRadiance[angle_idx]) *
                    (PSTRadiance[angle_idx + 2] -
                     PSTRadiance[angle_idx + 1]) /
                    (alpha *
                         (PSTRadiance[angle_idx + 2] -
                          PSTRadiance[angle_idx + 1]) *
                         (rho[angle_idx + 1] - rho[angle_idx]) +
                     (1 - alpha) *
                         (PSTRadiance[angle_idx + 1] -
                          PSTRadiance[angle_idx]) *
                         (rho[angle_idx + 2] - rho[angle_idx + 1]));
            }

            // Get parameters in the region rho_AZ[angle_idx]

            std::array<double, 4> param = getCubicParameters(
                    double(rho[angle_idx]), double(rho[angle_idx + 1]),
                                            PSTRadiance[angle_idx], PSTRadiance[angle_idx + 1], d0, d1);

            cubicParameters.push_back(param);

            d0 = d1;
        }

        parameters[i] = cubicParameters;
    }
}





/**
 * \brief: Returns the 4 parameters that fit the 1-D monotonic cubic
 *         interpolation *in the region x_0 -> x_1.
 *
 * \details: See scipy PchipInterpolator (https://t.ly/8cqwu)
 *
 * \return: Array (parameters) with four parameters that define a cubic
 *          polynomial so that:
 *          f(x) = parameters[0]*x^3 + parameters[1]*x^2 + parameters[2]*x +
 *          parameters[3], and
 *          f(x_0) = y_0; f(x_1) = y_1; f'(x_0) = d0; f'(x_1) = d1
 *
 * \param: x_0 x_1   Min/Max x-values for which the cubic function f(x) is
 *                   defined. (x in the interval [x_0, x_1])
 * \param: y_0, y_1  Values at the Min/Max x-values. (y_i = f(x_i))
 * \param: d0,  d1   Values of the derivative function y'(x) at the Min/Max x-values. (y'(x_i) = di)
 */
std::array<double, 4> StrayLight::getCubicParameters(double x_0, double x_1,
                                                     double y_0, double y_1,
                                                     double d0, double d1)
{

    double a =
        (2 * (y_1 - y_0) - (x_1 - x_0) * (d1 + d0)) / (std::pow(x_0 - x_1, 3));

    double b = ((d1 - d0) + 3 * a * (std::pow(x_0, 2) - std::pow(x_1, 2))) /
               (2 * (x_1 - x_0));
    double c = d0 - 3 * a * std::pow(x_0, 2) - 2 * b * x_0;
    double d = y_1 - a * std::pow(x_1, 3) - b * std::pow(x_1, 2) - c * x_1;

    std::array<double, 4> parameters = {a, b, c, d};

    return parameters;
}





/**
 *
 * \brief: Returns the average straylight of the moon (in #electrons) over the subfield
 *
 * \param: time: Time at which we want to find the straylight
 *
 * \note: this function extract the needed parameters to call a more general
 *       function getStrayLightMoon(). 
   
 */
double StrayLight::getStrayLightMoon(double time)
{

    // Get the index so that we get the position at the correct time
    int idx = 0;
    while (times[idx] < time){idx++;}

    // We get the positions of the spacecraft, moon and sun
    arma::vec sun_pos = sun_positions[idx];
    arma::vec moon_pos = moon_positions[idx];
    arma::vec sc_pos = sc_positions[idx];

    Log.debug("Straylight: sun positions: (" + to_string(sun_pos[0]) +
              " " + to_string(sun_pos[1]) + " " + to_string(sun_pos[2]) + ")");
    Log.debug("Straylight: moon positions: (" + to_string(moon_pos[0]) + " " +
              to_string(moon_pos[1]) + " " + to_string(moon_pos[2]) + ")");
    Log.debug( "Straylight: spacecraft positions: (" + to_string(sc_pos[0]) + " " +
              to_string(sc_pos[1]) + " " + to_string(sc_pos[2]) + ")");

    // Transform these into a reference frame where the moon lies in the origin
    // and sun lies on the z-axis.

    // Translation that sets the moon in the origin

    sun_pos  = sun_pos - moon_pos;
    sc_pos   = sc_pos- moon_pos;
    moon_pos = moon_pos - moon_pos;
    
    double A = std::sqrt(sun_pos[0]*sun_pos[0] + sun_pos[1]*sun_pos[1]);
    double N = std::sqrt(sun_pos[0]*sun_pos[0] + sun_pos[1]*sun_pos[1] + sun_pos[2]*sun_pos[2]);

    // Rotate so that the sun lies on the z-axis

    double x = sun_pos[0];
    double y = sun_pos[1];
    double z = sun_pos[2];
    
    arma::Mat<double> rotation = { { y*N, -x*N, 0},
                                   { x*z,  y*z, -A*A},
                                   { x*A,  y*A,  z*A},};

    rotation = rotation/(A*N);

    sun_pos  = rotation * sun_pos;
    moon_pos = rotation * moon_pos;
    sc_pos = rotation * sc_pos;

    arma::Mat<double> rotatedAxis = rotation * telescopeAxis;
    return getStrayLightObject(moon, sun_pos, moon_pos, sc_pos,
                                                        rotatedAxis, 1000);
}





/**
 *
 * \brief: Returns the straylight of the object (in #electrons) in the subfield pixel
 *        (row, column).
 *
 * \param: object        The object from which we want to get the straylight
 * \param: sun_pos       Position of the sun
 * \param: object_pos    Position of the object
 * \param: sc_pos        Position of the spacecraft
 * \param: gridPoints    Amount of gridpoints used to model the object
 * 
 */
double StrayLight::getStrayLightObject(CelestialObject object, arma::vec sun_pos, arma::vec object_pos,
                                       arma::vec sc_pos, arma::Mat<double> telescopeAxis,
                                       unsigned int nGrid)
{

    // This models the total straylight from an object by reflecting light from (nGrid * nGrid) grid points on its surface.
    // Each grid point contributes to the straylight based on three key angles and its surface area.
    // The contribution is calculated as follows:

    // Contribution = Surface Area × cos(gpIrradiance) × cos(gpRadiance) × cos(scIrradiance) [m^2]

    // Where:
    //    -> gpIrradiance: The angle at which sunlight reaches the grid point relative to the normal at that point.
    //    -> gpRadiance: The angle at which the spacecraft is positioned relative to the grid point's normal.
    //    -> scIrradiance: The angle at which reflected light reaches the camera relative to the optical axis.

    // It is important to note that we only consider contributions where the cosine of all angles is positive. 
    // This condition ensures that light can effectively reach the camera, as negative values would indicate
    // angles where light does not contribute to straylight detection.

    double gridDependencies = 0;    // Total contribution of the grid dependencies
    for (unsigned int i=1; i<=nGrid; i++)        
    {
        for (unsigned int j=0; j<=nGrid; j++)
        {
            double theta = i * Constants::PI / (2 * nGrid); // Remark that we don't include the values pi/2 -> pi,
                                                            // since cos[theta] < 0 for these values.
            double phi   = 2*j*Constants::PI / nGrid;

            arma::vec n_object = {sin(theta)*cos(phi),
                          sin(theta)*sin(phi),
                          cos(theta)};
            double dA = std::pow(object.radius, 2)*sin(theta)
                        *(Constants::PI / (2*nGrid))* (2*Constants::PI / nGrid);

            double cos_gpIrradiance =
                arma::dot(sun_pos - object.radius * n_object, n_object) /
                arma::norm(sun_pos - object.radius * n_object);

            double cos_gpRadiance = arma::dot(sc_pos - object.radius*n_object, n_object) / arma::norm(sc_pos - object.radius*n_object);
            double cos_scIrradiance =
                arma::dot(object.radius * n_object - sc_pos,
                          telescopeAxis.col(2)) /
                arma::norm(object.radius * n_object - sc_pos);

            if (cos_gpRadiance > 0 && cos_scIrradiance > 0)
            {
                gridDependencies +=
                    cos_gpIrradiance * cos_gpRadiance * cos_scIrradiance * dA;

            }
        }
    }

    // We now determines the contributions of straylight that are wavelength-dependent. These consist of:
    // - Planck's equation for a black body radiator (B), which models the sunlight that will be reflected. 
    // - The Point Source Transmittance (PST) function, which requires the declination angle (the angle between the optical axis and the incoming ray)
    //   and the azimuth angle (the angle of the projected ray on the camera relative to the x-axis of the telescope). 
    // - Energy per photon, calculated using E = h * c * lambda, where E is energy, h is Planck’s constant, c is the speed of light and lambda is the wavelength.
    
    // The explicit wavelength dependence has already been integrated out (int_lambda B * PST / E). We simply load in the combined effects 
    // from an HDF5 file at the correct declination and azimuth angles. 

    // get the declination
    double declination = acos(arma::dot(sc_pos, telescopeAxis.col(2)) / arma::norm(sc_pos));
    declination = rad2deg(declination);

    // get the azimuth
    arma::vec projected = sc_pos - arma::dot(sc_pos, telescopeAxis.col(2))*telescopeAxis.col(2);
    double azimuth = acos(arma::dot(projected, telescopeAxis.col(0)) / arma::norm(projected));
    azimuth = rad2deg(azimuth);


    double pstRadiance = getPSTRadianceValue(declination, azimuth);
    
    // Lastely we have some constant values that need to be set in order to get
    // the straylight these are:
    // k * pi * pixelSize * exposureTime * Rsun^2 / (d_object_sun^2 *
    // d_object_camera^2), where
    // -> k is the reflexivity of the object
    // -> pixelSize is the area of a pixel (in m)
    // -> exposureTime is the time for one exposure (in s)
    // -> Rsun is the radius of the sun
    // -> d_object_sun is the distance from the object to the sun (in m)
    // -> d_object_camera is the discance from the object to the camera (in m)

    auto object_sun = (sun_pos - object_pos);
    double d_object_sun_sq = arma::dot(object_sun, object_sun);

    auto object_camera = (sc_pos - object_pos);
    double d_object_camera_sq = arma::dot(object_camera, object_camera);

    double const_term = Constants::PI * object.reflectivity * std::pow(pixelSize, 2) *
                        std::pow(Constants::SOLARRADIUS, 2) * cycleTime /
                        (d_object_sun_sq * d_object_camera_sq);

    return pstRadiance*gridDependencies*const_term;
}





/**
 *
 * \brief: Returns the "PST radiance value" at the given azimuth and declination level.
 *
 *
 * \param: declination, azimuth: angles for which we need to "PST value".
[degrees]
 *
 * \note: In this case the "PST radiance value" means contributions of the PST, BB
 *        radiator and photon energy integrated ever the wavelength.
 * \note: The PST value we get from the extrapolated PST might be negative, in
 *        that case we assume PST = 0.
 */
double StrayLight::getPSTRadianceValue(double declination, double azimuth)
{

    arma::vec pst_azs(4); 
    for (int idx_az=0; idx_az < 5; idx_az++)
    {
        int rhoLength = rhoValues[idx_az].size();

        // If our declanation is larger then the largest rho value in the pst
        // file, it's value is zero.
        if (rhoValues[idx_az][rhoLength - 1] < declination)
        {
            pst_azs[idx_az] = 0;
        }
        else
        {
            for (int idx_rho=0; idx_rho < rhoLength; idx_rho++)
            {
                // We want to find the closest index for which rhoValues[idx] <= declination.
                double rho = (rhoValues[idx_az])[idx_rho];
                if (rho >= declination)
                {
                
                    auto param = (parameters[idx_az])[idx_rho-1];
                    auto f = [param](double a)
                    {
                        return param[0] * std::pow(a, 3) + param[1] * std::pow(a, 2) +
                               param[2] * a + param[3];
                    };

                    pst_azs[idx_az] = f(declination);
                    break;
                }
            }
        }
    }

    if (azimuth < 0)
    {
        azimuth += 180;
    }
    else if (azimuth > 180)
    {
        azimuth -= 180;
    }

    if (azimuth == 0)
    {
        return pst_azs[0];
    }
    unsigned int idx = 0;
    while (azimuth > azs[idx])
    {
        idx++;
    }

    double pst = ((azimuth - azs[idx - 1]) * pst_azs[idx] +
                  (azs[idx] - azimuth) * pst_azs[idx - 1]) /
                 (azs[idx] - azs[idx - 1]);
    if (pst < 0)
    {
        pst = 0;
    }
    return pst;
}





/**
 * \brief: constructor for the Time class
 *
 * \note: this class is defined to deal well with
 *        the format of the time in the orbit file.
 */
Time::Time(std::string datetime)
{

    seconds = stoi(datetime.substr(13, 2));
    minutes = stoi(datetime.substr(11, 2));
    hours = stoi(datetime.substr(9, 2));
    days = stoi(datetime.substr(6, 2));
    months = stoi(datetime.substr(4, 2));
    years = stoi(datetime.substr(0, 4));

    struct tm tm;
    time_t rawtime;
    time(&rawtime);
    tm = *localtime(&rawtime);
    tm.tm_year = years - 1900;
    tm.tm_mon = months - 1;
    tm.tm_mday = days;
    tm.tm_hour = hours;
    tm.tm_min = minutes;
    tm.tm_sec = seconds;

    t = mktime(&tm);
}

int Time::totalSeconds() const
{
    return seconds + 60* (minutes + 60 * (hours + 24*days) );
}




#ifndef STRAYLIGHT_H
#define STRAYLIGHT_H

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "Camera.h"
#include "ConfigurationParameters.h"
#include "HDF5Writer.h"
#include "Logger.h"

// #include "HDF5File.h"
#include "armadillo"


class Camera;


struct GridPoint
{
    arma::vec point;
    double size;
};

struct CelestialObject
{
    double radius;
    double reflectivity;
};


class StrayLight : public HDF5Writer
{
  public:
    StrayLight(ConfigurationParameters &configParam, HDF5File &hdf5File,
                                                     Camera &camera);
    // ~StrayLight();
    void configure(ConfigurationParameters &configParam);
    double getStrayLightMoon(double time);


protected:
    CelestialObject moon;
    CelestialObject earth;

    void readInFile(std::string orbitPath, std::vector<arma::vec> &sc_pos,
                    std::vector<arma::vec> &moon_pos, std::vector<arma::vec> &sun_pos);
    std::vector<std::string> splitLine(std::string &line);
    void getPSTRadiance(std::string pstRadiancePath);
    void interpolatePSTRadiance(std::array<std::vector<double>, 5> rho, std::array<std::vector<double>, 5> pstRadiance);
    std::array<double, 4> getCubicParameters(double x_0, double x_1,
                                             double y_0, double y_1,
                                             double d0, double d1);

  double getStrayLightObject(CelestialObject object, arma::vec sun_pos, arma::vec object_pos, arma::vec sc_pos, arma::Mat<double> telescopeAxis, unsigned int nGridPoints);
    double getPSTRadianceValue(double declination, double azimuth);


    std::array<std::vector<std::array<double, 4>>,5> parameters;
    std::array<std::vector<double>, 5> rhoValues;
    std::vector<arma::vec> moon_positions;
    std::vector<arma::vec> sc_positions;
    std::vector<arma::vec> sun_positions;
    std::vector<double> times;
    arma::Mat<double> telescopeAxis{arma::Mat<double>(3,3)};

    std::string time0;

    std::array<int, 5> azs = {0, 45, 90, 135, 180};

    HDF5File pstFile;

    double pixelSize;
    double numExposure;
    double beginExposures;
    double cycleTime;

    int AZs[5];
    Camera &camera;
};



class Time
{
  public:
    Time(std::string datetime);
    Time();

    time_t t;

    int totalSeconds() const;

  private:
    int seconds;
    int minutes;
    int hours;
    int days;
    int months;
    int years;
};
#endif

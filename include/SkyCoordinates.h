
#ifndef SKYCOORDINATES_H
#define SKYCOORDINATES_H

#include <cmath>

#include "Units.h"
#include "Logger.h"


using namespace std;




class SkyCoordinates
{
    public:

        SkyCoordinates(double RA, double decl, Unit angleUnit = Angle::degrees);
        ~SkyCoordinates();

        pair<double, double> toGalactic(Unit angleUnit = Angle::degrees);
        pair<double, double> toEcliptic(Unit angleUnit = Angle::degrees);


        friend double angularDistanceBetween(SkyCoordinates &skyCoordinates1, SkyCoordinates &skyCoordinates2, Unit outputAngleUnit);

    protected:

        double RA;                    // Equatorial right ascension [rad]
        double decl;                  // Equatorial declination     [rad]

    private:

};



double angularDistanceBetween(const double RA1, const double dec1, const double RA2, const double dec2, Unit outputAngleUnit);
vector<double> angularDistanceBetween(const double RA0, const double dec0, const vector<double> &RA, const vector<double> &dec, Unit angleUnit);
void ecliptic2equatorial(const double lambda, const double beta, double &alpha, double &delta);
void equatorial2ecliptic(const double alpha, const double delta, double &lambda, double &beta);
void equatorial2galactic(const double alpha, const double delta, double &l, double &b);
double angularDistanceBetween(SkyCoordinates &skyCoordinates1, SkyCoordinates &skyCoordinates2, Unit outputAngleUnit);
vector<double> angularDistanceBetween(const double RA0, const double dec0, const vector<double> &RA, const vector<double> &dec, Unit outputAngleUnit);

#endif


#ifndef HDF5FILE_H
#define HDF5FILE_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <array>

#include "H5Cpp.h"
#include "memory.h"
#include "armadillo"

#include "Logger.h"
#include "version.h"
#include "HDF5Exceptions.h"
#include "Exceptions.h"

using namespace std;

struct RaggedArray
{
    map<string, H5::DataSet> uintDatasets;  // datasets, keyed by field name of type unsigned int
    map<string, H5::DataSet> doubleDatasets; // datasets, keyed by field name of type float
    H5::DataSet offsetsDataset;

    std::vector<uint64_t> offsets;
    hsize_t runningTotal = 0;
};

class HDF5File
{
    public:

        HDF5File();
        HDF5File(string filename, bool readonly=true);
        HDF5File(bool noFile);
        virtual ~HDF5File();

        void open(string filename, bool readonly=true);
        void close();

        bool hasGroup(string groupName);
        void createGroup(string groupName);
        RaggedArray createGroupForCosmics(string groupName, hsize_t numberExposures);
        void createGroup(string groupName, string arrayName, hsize_t dims[3],
                          const H5::PredType &type);

        bool hasDataset(string groupName, string datasetName);

        virtual void writeAttribute(string groupName, string attributeName, string attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, int attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, long attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, double attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, bool attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, vector<double> attributeValue);
        virtual void writeAttribute(string groupName, string attributeName, vector<int> attributeValue);

        virtual void writeArray(string groupName, string arrayName, int*          array, int size);
        virtual void writeArray(string groupName, string arrayName, unsigned int* array, int size);
        virtual void writeArray(string groupName, string arrayName, float*        array, int size);
        virtual void writeArray(string groupName, string arrayName, double*       array, int size);
        template <typename T>
        void writeArray(string groupName, string arrayName, const arma::Mat<T> &A);
        template <typename T>
        void writeArray(string groupName, string arrayName, int timeStep,
			const arma::Mat<T> &data);
        template<typename T>
        static H5::PredType getPredType(const arma::Mat<T>& A);

        virtual void writeArray(string groupName, string arrayName, const arma::Mat<float>& A);
        virtual void writeArray(string groupName, string arrayName, const arma::Mat<uint16_t>& A);

        double readDoubleGroupAttribute(string groupName, string attributeName);
        int readIntegerGroupAttribute(string groupName, string attributeName);
        void readArrayDatasetAttribute(string groupName, string dataset, string attributeName, double *outputArray);
        double readDoubleDatasetAttribute(string groupName, string datasetName, string attributeName);
        string readStringDatasetAttribute(string groupName, string datasetName, string attributeName);

        void readArray(string groupName, string arrayName, arma::Mat<float>& A);
        void readArray(string groupName, string arrayName, vector<double> &vec);
        void readArray(string groupName, string arrayName, vector<unsigned int> &vec);

        void writeVersionInformation();
        void writeTransmissionEfficiencyValues(double* array, int size);
        void writeSmearingMap(arma::Mat<float>& smearingMap, bool includeQuantisation, int exposureNr);
        void writeStraylight(vector<double>& );
        void writeTelescopeACS(vector<double>&, vector<double>&, vector<double>&, vector<double>&,
                                vector<double>&, vector<double>&);
        void writeStarPositionByExposure(map<double, map<unsigned int, array<double, 6>>>& detectedStarInfo,
                                int beginExposureNr);
        void writeStarPositionByStarID(map<double, map<unsigned int, array<double, 6>>>& detectedStarInfo,
                                vector<unsigned int> starIDs);
        void writePointlikeGhostByExposure(map<double, map<unsigned int, array<double, 6>>>&
                                 detectedPointLikeGhostInfo, int beginExposureNr);
        void writePointlikeGhostByStarID(std::map<double, std::map<unsigned int, std::array<double, 6>>>&
                                 detectedPointLikeGhostInfo);
        void writeExtendedGhostByExposure(map<double, map<unsigned int, array<double, 7>>>&
                                 detectedExtendedGhostInfo, int beginExposureNr);
        void writeExtendedGhostByStarID(map<double, map<unsigned int, array<double, 7>>>& detectedExtendedGhostInfo);
        void writeCosmics(RaggedArray& array, int exposureNr, vector<unsigned int> &entryRows,
                          vector<unsigned int> &entryColumns, vector<double> &trailLengths,
                          vector<double> &entryAngles, vector<double> &intensities, vector<unsigned int> &rows,
			  vector<unsigned int> &cols, vector<double> &flux);

    protected:

        H5::H5File *file;
        bool fileIsOpen;
        bool noFile;

    private:
        template<typename T>
	void addDataToRaggedArray(map<string, vector<T>>& data, map<string, H5::DataSet>& datasets, const hsize_t offsetVal, const hsize_t n);

};



bool fileExists(string filename);
template <typename T> struct Hdf5PredType;

template <>
struct Hdf5PredType<unsigned int>
{
  static const H5::PredType &get() { return H5::PredType::NATIVE_UINT; }
};

template <>
struct Hdf5PredType<double>
{
  static const H5::PredType &get() { return H5::PredType::NATIVE_DOUBLE; }
};


#endif

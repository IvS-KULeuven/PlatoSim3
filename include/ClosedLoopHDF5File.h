
#ifndef CLOSEDLOOPHDF5FILE_H
#define CLOSEDLOOPHDF5FILE_H

#include "HDF5File.h"


class ClosedLoopHDF5File : public HDF5File
{
    public:

        ClosedLoopHDF5File() {};

        ~ClosedLoopHDF5File() {if(fileIsOpen){close();} delete file;};

        void writeAttribute(string groupName, string attributeName, string         attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, int            attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, long           attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, double         attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, bool           attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, vector<double> attributeValue) override {};
        void writeAttribute(string groupName, string attributeName, vector<int>    attributeValue) override {};
        
        void writeArray(string groupName, string arrayName, int*          array, int size) override {};
        void writeArray(string groupName, string arrayName, unsigned int* array, int size) override {};
        void writeArray(string groupName, string arrayName, float*        array, int size) override {};
        void writeArray(string groupName, string arrayName, double*       array, int size) override {};

        void writeArray(string groupName, string arrayName, const arma::Mat<float>& A) override {};
        void writeArray(string groupName, string arrayName, const arma::Mat<uint16_t>& A) override {};

    protected:


    private:

};



#endif

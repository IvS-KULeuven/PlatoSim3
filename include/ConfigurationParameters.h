#ifndef CONFIGURATION_PARAMETERS_H_
#define CONFIGURATION_PARAMETERS_H_

#include <vector>
#include <string>
#include <regex>
#include <list>
#include <stack>
#include <fstream>

#include "yaml-cpp/yaml.h"
#include "yaml-cpp/exceptions.h"

#include "FileUtilities.h"
#include "StringUtilities.h"
#include "Exceptions.h"
#include "Logger.h"


using namespace std;


class ConfigurationParameters
{
    public:
        ConfigurationParameters();
        ConfigurationParameters(const string &);
        ~ConfigurationParameters();
    

        vector<string> getKeys(const string nodeName);
        bool nodeExists(const string nodeName);
        bool getBoolean(const string &);
        int getInteger(const string &);
        unsigned int getUnsignedInteger(const string &);
        long getLong(const string &);
        double getDouble(const string &);
        string getString(const string &);
        string getAbsoluteFilename(const string &);
    
        vector <double> getDoubleVector(const string &key);
        vector <int> getIntegerVector(const string &key);
    
        double getDoubleAt(const string &key, int idx);
        int getIntegerAt(const string &key, int idx);

        void setParameter(const string &, const string &);
        bool hasParameter(const string & key);
    
    private:
        string filename;
        YAML::Node config;
    
        YAML::Node getNode(const string & key);

};


#endif /* CONFIGURATION_PARAMETERS_H_ */

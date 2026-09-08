/**
 * \class HDF5File
 *
 * \brief      Provides an application level interface to the HDF5 C++ API
 *
 * \details
 *
 * This class provides a convenient application level interface to the HDF5 C++ wrapper to
 * the HDF C library that is developed by the HDFGroup.
 *
 * Use this class to access (read/write) HDF5 files from your code. Do not use the HDF5 C++
 * wrapper.
 */

#include "HDF5File.h"



/**
 * \brief      Default Constructor
 */

HDF5File::HDF5File()
: file(NULL), fileIsOpen(false)
{

}







/**
 * \brief  Open file constructor
 *
 * \param  filename   Absolute path of the file
 * \param  readonly   True if an existing file should only be read and not written.
 *                    False otherwise. Ignored if the file does not exist yet.
 */

HDF5File::HDF5File(string filename, bool readonly)
: file(NULL), fileIsOpen(false)
{
    H5::Exception::dontPrint();
    open(filename, readonly);
}







/**
 * \brief      Destructor
 */

HDF5File::~HDF5File()
{
    if (fileIsOpen)
    {
        close();
    }
    delete file;
}








/**
 * \brief Open an HDF5 File. If it doesn't already exist, create it.
 *
 *
 * \param filename   Absolute path of the file.
 * \param readonly   True if an existing file should only be read and not written.
 *                   False otherwise. Ignored if the file does not exist yet.
 */

void HDF5File::open(string filename, bool readonly)
{
    // If there was already a file opened, complain and exit

    if (fileIsOpen)
    {
        close();
        throw H5FileException("HDF5File::open(): file " + filename + " is already open.");
    }

    // Check if the file already exists.
    // If so, use append mode, if not

    if (fileExists(filename))
    {
        if (readonly)
        {
            // Open existing HDF5 file to read.

            file = new H5::H5File(filename.c_str(), H5F_ACC_RDONLY);

            Log.info("HDF5File: opened existing HDF5 file " + filename + " to read");

        }
        else
        {
            // Open an existing HDF5 file to both read and write

            file = new H5::H5File(filename.c_str(), H5F_ACC_RDWR);

            Log.info("HDF5File: opened existing HDF5 file " + filename + " to read/write");
        }
    }
    else
    {
        // The file doesn't exist yet. Create a new one.
        // New files are always opened for read/write.

        file = new H5::H5File(filename.c_str(), H5F_ACC_TRUNC);

        Log.info("HDF5File: opened new HDF5 file " + filename + " to read/write");
    }

    fileIsOpen = true;
    return;
}









/**
 * \brief Close the HDF5 file in case it was open. It's safe to close a file that wasn't even open.
 */
void HDF5File::close()
{
    if (fileIsOpen)
    {
        file->close();
        delete file;
        file = nullptr;
        fileIsOpen = false;
    }
}











/**
 * \brief      Check if the HDF5 file has a group with the given name
 *
 * \param[in]  groupName    the full name of the group
 *
 * \return     true if the group exists in this file, false otherwise
 */

bool HDF5File::hasGroup(string groupName)
{
    try
    {
        H5::Group parentGroup = file->openGroup(groupName.c_str());
        return true;
    }
    catch (H5::Exception ex) {}

    return false;
}






/**
 * \brief      Check if the given group has a dataset with the given name
 *
 * \param[in]  groupName    the full name of the group that contains the dataset
 * \param[in]  datasetName  the name of the dataset
 *
 * \return     true if the dataset exists in this group, false otherwise
 *
 * \exception  H5GroupException thrown when the group is unknown to the HDF5 file
 */
bool HDF5File::hasDataset(string groupName, string datasetName)
{
    if (hasGroup(groupName))
    {
        H5::Group group = file->openGroup(groupName.c_str());

        try
        {
            H5::DataSet dataset = group.openDataSet(datasetName.c_str());
            return true;
        }
        catch (H5::Exception ex) {}

        return false;
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    return false;
}




/**
 *
 * TODO: We should set up the groups for the cosmics
 *
 */
RaggedArray HDF5File::createGroupForCosmics(string subGroupName, hsize_t numberOfExposures)
{
    // Make sure the path of the group starts with a "/" (i.e. the root folder)

    if (!subGroupName.compare(0, 1, "/"))
    {
        subGroupName.erase(0, 1);
    }

    // Open the parent group

    H5::Group parentGroup = file->openGroup("/Cosmics");

    // Create the new subgroup inside the parent group.

    H5::Group group = parentGroup.createGroup(subGroupName.c_str());

    // We need to create data that will persiset over multiple exposures

    RaggedArray data;

    // The first persistend data is the offsets table
    // for which the size is known (number of exposures + 1)

    hsize_t offDims[1] = { numberOfExposures+1 };
    H5::DataSpace offSpace(1, offDims);
    data.offsetsDataset = group.createDataSet("/Cosmics/" + subGroupName + "/offsets", H5::PredType::NATIVE_UINT64, offSpace);

    data.offsets.resize(numberOfExposures + 1);
    data.offsets[0] = 0;

    // Documentation for the hdf5 file, so it is available from the hdf5 file
    // viewer.
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace attrSpace(H5S_SCALAR);
    
    const char* text = "Ragged array since not every exposure has the same amount of cosmics.\n "
                       "For exposure i, each field's values for that exposures are:\n\n "
                       "field[offsets[i]:offsets[i+1]].\n\nRemark that: "
                       "offsets[0] == 0 and an exposure with no entries has offsets[i] == offsets[i+1].";

    H5::Attribute attr = group.createAttribute("description", strType, attrSpace);
    attr.write(strType, &text);


    // Secondly we initialize the data the will form the rows of our ragged
    // array. We don't know the initial size of these array, but we will add to
    // the these array during the simulation. The type of data that is saved is
    // specified by the key values
    

    hsize_t initDims[1] = {0};
    hsize_t maxDims[1] = {H5S_UNLIMITED};
    H5::DataSpace space(1, initDims, maxDims);

    hsize_t chunkDims[1] = {4096};  
    H5::DSetCreatPropList plist;
    plist.setChunk(1, chunkDims);
    plist.setDeflate(4);

    vector<string> uintFieldNames = {"entryRows", "entryColumns", "rows", "columns"};
    vector<string> doubleFieldNames = {"trailLengths", "entryAngles", "intensities", "flux"};

    for (const auto& name : uintFieldNames)
    {
      data.uintDatasets[name] = group.createDataSet("/Cosmics/" + subGroupName + "/" + name,
						      H5::PredType::NATIVE_UINT, space, plist);
    }

    for (const auto& name : doubleFieldNames)
    {
      data.doubleDatasets[name] = group.createDataSet("/Cosmics/" + subGroupName + "/" + name,
                            H5::PredType::NATIVE_DOUBLE, space, plist);
    }

    return data;
}



/**
 * \brief Create a group in an HDF5 file.
 *
 * \param groupName: Full path of the group. Should always start with "/".
 *
 * \example createGroup("/my/path/to/subgroup1") will create "subgroup1"
 *          in the parent group "/my/path/to".
 *          Note that the parent group is assumed to already exist.
 *          If not, an error will be given. So, to create "/my/path/to/subgroup1"
 *          from scratch, you should call:
 *             createGroup("/my");
 *             createGroup("/my/path");
 *             createGroup("/my/path/to");
 *             createGroup("/my/path/to/subgroup1");
 */
void HDF5File::createGroup(string groupName)
{
    // Make sure the path of the group starts with a "/" (i.e. the root folder)

    if (groupName.compare(0, 1, "/"))
    {
        groupName.insert(0, "/");
    }

    // Find the parent group.
    // E.g. if the groupName is "/my/path/to/group1" then the parent group is
    //      "/my/path/to" and the subgroup is "group1".

    auto position = groupName.find_last_of("/");
    string parentGroupName = groupName.substr(0, position);
    string subGroupName = groupName.substr(position+1);

    // If the parent group name is empty, it means that the group should be
    // placed in the root group. In that case, make sure that the parentGroupName
    // is "/".

    if (parentGroupName.empty())
    {
        parentGroupName = "/";
    }

    // Open the parent group

    H5::Group parentGroup = file->openGroup(parentGroupName.c_str());

    // Create the new subgroup inside the parent group.

    H5::Group subGroup = parentGroup.createGroup(groupName.c_str());
}












 /**
 * \brief Add a string attribute to the HDF5 file.  The attribute is stored in
 *        "<GroupName>/<attributeName>".
 *
 * \param groupName: String containing the full path of an existing group. Starts with "/".
 * \param attributeName: String containing the name of the attribute
 * \param attributeValue: String containing the attribute value
 *
 * \example: writeAttribute("/InputParameters/ObservingParameters", "CCDPredefinedPosition", "User")
 */
void HDF5File::writeAttribute(string groupName, string attributeName, string attributeValue)
{
   // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file->getFileName() + " is not open.");
    }

    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group
    // H5T_VARIABLE refers to a variable-length string
    // H5S_SCALAR refers to a string scalar attribute

    H5::StrType variableLengthStringType(0, H5T_VARIABLE);
    H5::DataSpace attributeSpace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attributeName.c_str(), variableLengthStringType, attributeSpace);
    attribute.write(variableLengthStringType, H5std_string(attributeValue.c_str()));

    // That's it

    attribute.close();
    group.close();

    return;
}











// HDF5File::writeAttribute() for integer attributes
//
// PURPOSE: add an integer attribute to the HDF5 file.
//          The attribute is stored in "<GroupName>/<attributeName>".
//
// INPUT: groupName:      string containing the full path of an existing group. Starts with "/".
//        attributeName:  string containing the name of the attribute
//        attributeValue: integer containing the attribute value
//
// OUTPUT: None
//
// EXAMPLE: writeAttribute("/InputParameters/CCD", "CCDSizeX", 4510)
//
void HDF5File::writeAttribute(string groupName, string attributeName, int attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file->getFileName() + " is not open.");
    }


    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group

    H5::IntType integerType(H5::PredType::NATIVE_INT);
    H5::DataSpace attributeSpace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attributeName.c_str(), integerType, attributeSpace);
    attribute.write(integerType, &attributeValue);
    attribute.close();

    // That's it

    attribute.close();
    group.close();

    return;
}













// HDF5File::writeAttribute()  for long integer attributes
//
// PURPOSE: add a long integer attribute to the HDF5 file.
//          The attribute is stored in "<GroupName>/<attributeName>".
//
// INPUT: groupName:      string containing the full path of an existing group. Starts with "/".
//        attributeName:  string containing the name of the attribute
//        attributeValue: long integer containing the attribute value
//
// OUTPUT: None
//
// EXAMPLE: writeAttribute("/InputParameters/SeedParameters", "PhotonNoise", 1433237514)
//

void HDF5File::writeAttribute(string groupName, string attributeName, long attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file-> getFileName() + " is not open.");
    }


    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group

    H5::IntType integerType(H5::PredType::NATIVE_LONG);
    H5::DataSpace attributeSpace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attributeName.c_str(), integerType, attributeSpace);
    attribute.write(integerType, &attributeValue);
    attribute.close();

    // That's it

    attribute.close();
    group.close();

    return;
}












/**
 * \brief  Add a double-valued attribute to the HDF5 file.
 *         The attribute is stored in "<GroupName>/<attributeName>".
 *
 * \param groupName: String containing the full path of an existing group. Starts with "/".
 * \param attributeName: String containing the name of the attribute
 * \param attributeValue: Double containing the attribute value
 *
 * \example writeAttribute("/InputParameters/JitterParameters", "JitterYawRms", 0.01)
*/
void HDF5File::writeAttribute(string groupName, string attributeName, double attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file->getFileName() + " is not open.");
    }

    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group

    H5::FloatType floatType(H5::PredType::NATIVE_DOUBLE);
    H5::DataSpace attributeSpace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attributeName.c_str(), floatType, attributeSpace);
    attribute.write(floatType, &attributeValue);
    attribute.close();

    // That's it

    attribute.close();
    group.close();

    return;
}










/**
 * \brief Add a boolean attribute to the HDF5 file.
 *        The attribute is stored in "<GroupName>/<attributeName>".
 *
 * \param groupName: String containing the full path of an existing group. Starts with "/".
 * \param attributeName: String containing the name of the attribute
 * \param attributeValue: Integer containing the attribute value
 *
 * \example  writeAttribute("/InputParameters/Platform", "UseJitter", true)
 *
 * \exception  H5FileException: If the HDF5 file has not been opened
 * \exception  H5FileException: If the attribute already existed in the HDF5 file
 */
void HDF5File::writeAttribute(string groupName, string attributeName, bool attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        string errorMessage = "HDF5File::writeAttribute(): file is not open.";
        Log.error(errorMessage);
        throw H5FileException(errorMessage);
    }


    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        string errorMessage = "HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName;
        Log.error(errorMessage);
        throw H5FileException(errorMessage);
    }


    // Create and write the attribute to the group
    // First convert from boolean (true/false) to integer (1/0)

    int value;
    if (attributeValue)
    {
        value = 1;
    }
    else
    {
        value = 0;
    }

    H5::IntType integerType(H5::PredType::NATIVE_INT);
    H5::DataSpace attributeSpace(H5S_SCALAR);
    H5::Attribute attribute = group.createAttribute(attributeName.c_str(), integerType, attributeSpace);
    attribute.write(integerType, &value);
    attribute.close();

    // That's it

    attribute.close();
    group.close();

    return;
}













/**
 * \brief Add an attribute of type vector <double> to the HDF5 file.
 *        The attribute is stored in "<GroupName>/<attributeName>".
 *
 * \param groupName: String containing the full path of an existing group. Starts with "/".
 * \param attributeName: String containing the name of the attribute
 * \param attributeValue: Vector <double> containing the attribute values
 *
 * \example writeAttribute("/InputParameters/Camera/FieldDistortion", "Coefficients", values)
 */
void HDF5File::writeAttribute(string groupName, string attributeName, vector<double> attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file->getFileName() + " is not open.");
    }


    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group. The attribute is a vector <double> so we need
    // to specify the rank and the dimension of the vector. Since a vector is guaranteed to store
    // their elements contiguously, we can just pass the pointer to the first element when writing
    // the attribute.

    try
    {
        const int rank = 1;
        hsize_t dimension[1];
        dimension[0] = attributeValue.size();
        H5::FloatType floatType(H5::PredType::NATIVE_DOUBLE);
        H5::DataSpace attributeSpace(rank, dimension);
        H5::Attribute attribute = group.createAttribute(attributeName.c_str(), floatType, attributeSpace);
        attribute.write(floatType, &attributeValue[0]);
        attribute.close();
    }
    catch(H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " write error " + error.getCDetailMsg());
    }

    // That's it

    group.close();

    return;
}














/**
 * \brief Add an attribute of type vector <int> to the HDF5 file.
 *        The attribute is stored in "<GroupName>/<attributeName>".
 *
 * \param groupName: String containing the full path of an existing group. Starts with "/".
 * \param attributeName: String containing the name of the attribute
 * \param attributeValue: Vector <int> containing the attribute values
 *
 * \example writeAttribute("/InputParameters/CCDPositions/", "NumRows", values)
 */
void HDF5File::writeAttribute(string groupName, string attributeName, vector<int> attributeValue)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File::writeAttribute(): file " + file->getFileName() + " is not open.");
    }


    // Open the proper group where the input parameter belongs

    H5::Group group = file->openGroup(groupName.c_str());

    // Check whether the attribute is not already in the group. If so, complain.
    // The only way to do so, seems to try to access it and catch the exception
    // if it does not yet exist.

    bool attributeIsAlreadyInGroup = true;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        H5::Attribute attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        attributeIsAlreadyInGroup = false;
    }

    if (attributeIsAlreadyInGroup)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " already in group " + groupName);
    }


    // Create and write the attribute to the group. The attribute is a vector <int> so we need
    // to specify the rank and the dimension of the vector. Since a vector is guaranteed to store
    // their elements contiguously, we can just pass the pointer to the first element when writing
    // the attribute.

    try
    {
        const int rank = 1;
        hsize_t dimension[1];
        dimension[0] = attributeValue.size();
        H5::IntType intType(H5::PredType::NATIVE_INT);
        H5::DataSpace attributeSpace(rank, dimension);
        H5::Attribute attribute = group.createAttribute(attributeName.c_str(), intType, attributeSpace);
        attribute.write(intType, &attributeValue[0]);
        attribute.close();
    }
    catch(H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File::writeAttribute(): attribute " + attributeName + " write error " + error.getCDetailMsg());
    }

    // That's it

    group.close();

    return;
}














/**
 * \brief      read a double-valued attribute that is associated with groupName
 *
 * \param[in]  groupName      string containing the full path of an existing group. Starts with "/".
 * \param[in]  attributeName  string containing the name of the attribute
 *
 * \return     the value of the attribute
 *
 * \exception  H5FileException      if the HDF5 file has not been opened
 * \exception  H5GroupException     if the group is unknown to the HDF5 file
 * \exception  H5AttributeException if there is no attribute with the given name
 */
double HDF5File::readDoubleGroupAttribute(string groupName, string attributeName)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File: The file (" + file->getFileName() + ") has not been opened.");
    }

    // Open the proper group where the attribute is associated

    H5::Group group;
    if (hasGroup(groupName))
    {
        group = file->openGroup(groupName.c_str());
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    // Check whether the attribute is in the group by trying to read it.
    // If not, raise an exception.

    H5::Attribute attr;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File: Unknown Attribute (" + attributeName + ") in the group " + groupName + " for HDF5 file " + file->getFileName());
    }

    double value = 0.0;

    H5::DataType type = attr.getDataType();
    attr.read(type, &value);

    return value;
}










void HDF5File::readArrayDatasetAttribute(string groupName, string datasetName, string attributeName, double *outputArray)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File: The file (" + file->getFileName() + ") has not been opened.");
    }

    // Open the proper group where the attribute is associated

    H5::Group group;
    if (hasGroup(groupName))
    {
        group = file->openGroup(groupName.c_str());
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    H5::DataSet dataset;
    if (hasDataset(groupName, datasetName))
    {
        dataset = group.openDataSet(datasetName);
    }
    else
    {
        throw H5DatasetException("HDF5File: Unknown dataset (" + datasetName + ") in group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    // Check whether the attribute is in the group by trying to read it.
    // If not, raise an exception.

    H5::Attribute attr;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        attr = dataset.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File: Unknown Attribute (" + attributeName + ") in the group " + groupName + " for HDF5 file " + file->getFileName());
    }


    H5::DataType type = attr.getDataType();
    attr.read(type, outputArray);
}











/**
 * \brief      read a integer-valued attribute that is associated with groupName
 *
 * \param[in]  groupName      string containing the full path of an existing group. Starts with "/".
 * \param[in]  attributeName  string containing the name of the attribute
 *
 * \return     the value of the attribute
 *
 * \exception  H5FileException      if the HDF5 file has not been opened
 * \exception  H5GroupException     if the group is unknown to the HDF5 file
 * \exception  H5AttributeException if there is no attribute with the given name
 */

int HDF5File::readIntegerGroupAttribute(string groupName, string attributeName)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File: The file (" + file->getFileName() + ") has not been opened.");
    }

    // Open the proper group where the attribute is associated

    H5::Group group;
    if (hasGroup(groupName))
    {
        group = file->openGroup(groupName.c_str());
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    // Check whether the attribute is in the group by trying to read it.
    // If not, raise an exception.

    H5::Attribute attr;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        attr = group.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File: Unknown Attribute (" + attributeName + ") in the group " + groupName + " for HDF5 file " + file->getFileName());
    }

    int value = 0.0;

    H5::DataType type = attr.getDataType();
    attr.read(type, &value);

    return value;
}














/**
 * \brief      read a double-valued attribute that is associated with dataset
 *
 * \param[in]  groupName      string containing the full path of an existing group. Starts with "/".
 * \param[in]  datasetName    string containing the name of the dataSet where the attribute is attached
 * \param[in]  attributeName  string containing the name of the attribute
 *
 * \return     the value of the attribute
 *
 * \exception  H5FileException      if the HDF5 file has not been opened
 * \exception  H5GroupException     if the group is unknown to the HDF5 file
 * \exception  H5DatasetException   if the dataset is not known to the group
 * \exception  H5AttributeException if there is no attribute with the given name
 */

double HDF5File::readDoubleDatasetAttribute(string groupName, string datasetName, string attributeName)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File: The file (" + file->getFileName() + ") has not been opened.");
    }

    // Open the proper group where the attribute is associated

    H5::Group group;
    if (hasGroup(groupName))
    {
        group = file->openGroup(groupName.c_str());
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    H5::DataSet dataset;
    if (hasDataset(groupName, datasetName))
    {
        dataset = group.openDataSet(datasetName);
    }
    else
    {
        throw H5DatasetException("HDF5File: Unknown dataset (" + datasetName + ") in group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    // Check whether the attribute is in the group by trying to read it.
    // If not, raise an exception.

    H5::Attribute attr;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        attr = dataset.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File: Unknown Attribute (" + attributeName + ") in the group " + groupName + " for HDF5 file " + file->getFileName());
    }

    double value = 0.0;

    H5::DataType type = attr.getDataType();
    attr.read(type, &value);

    return value;
}












/**
 * \brief      read a string-valued attribute that is associated with dataset
 *
 * \param[in]  groupName      string containing the full path of an existing group. Starts with "/".
 * \param[in]  datasetName    string containing the name of the dataSet where the attribute is attached
 * \param[in]  attributeName  string containing the name of the attribute
 *
 * \return     the value of the attribute
 *
 * \exception  H5FileException      if the HDF5 file has not been opened
 * \exception  H5GroupException     if the group is unknown to the HDF5 file
 * \exception  H5DatasetException   if the dataset is not known to the group
 * \exception  H5AttributeException if there is no attribute with the given name
 */

string HDF5File::readStringDatasetAttribute(string groupName, string datasetName, string attributeName)
{
    // Complain if the file was not first opened

    if (!fileIsOpen)
    {
        throw H5FileException("HDF5File: The file (" + file->getFileName() + ") has not been opened.");
    }

    // Open the proper group where the attribute is associated

    H5::Group group;
    if (hasGroup(groupName))
    {
        group = file->openGroup(groupName.c_str());
    }
    else
    {
        throw H5GroupException("HDF5File: Unknown group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    H5::DataSet dataset;
    if (hasDataset(groupName, datasetName))
    {
        dataset = group.openDataSet(datasetName);
    }
    else 
    {
        throw H5DatasetException("HDF5File: Unknown dataset (" + datasetName + ") in group (" + groupName + ") in HDF5 file " + file->getFileName());
    }

    // Check whether the attribute is in the group by trying to read it.
    // If not, raise an exception.

    H5::Attribute attr;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the attribute

        attr = dataset.openAttribute(attributeName.c_str());
    }
    catch (H5::AttributeIException error)
    {
        throw H5AttributeException("HDF5File: Unknown Attribute (" + attributeName + ") in the group " + groupName + " for HDF5 file " + file->getFileName());
    }


    H5::StrType stringType = attr.getStrType();

    string attributeValue;
    attr.read(stringType, attributeValue);

    return attributeValue;
}












// HDF5File::writeArray()  for 1D integer arrays
//
// PURPOSE: write a 1D array to a specified group in the HDF5 file.
//
// INPUT: groupName: full path of an existing HDF5 Group in the file. Starts with "/".
//        arrayName: unique name of the array in the group, e.g. "starIDs000001"
//        array:     1D integer native array
//        size:      number of elements in the array
//
// OUTPUT: None

void HDF5File::writeArray(string groupName, string arrayName, int* array, int size)
{
    // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[1] = {size};
    H5::DataSpace arraySpace(1, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }


    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_INT, arraySpace);

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(array, H5::PredType::NATIVE_INT);

    // That's it

    return;
}













/**
 * \brief Write a 1D unsigned integer array to a specified group in the HDF5 file
 * 
 * \param groupName  Full path of an existing HDF5 Group in the file. Starts with "/". 
 * \param arrayName  Unique name of the array in the group 
 * \param array      1d unsigned integer native array
 * \param size       Number of elements in the array
 */

void HDF5File::writeArray(string groupName, string arrayName, unsigned int* array, int size)
{
    // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[1] = {size};
    H5::DataSpace arraySpace(1, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        string errorMessage = "HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.";
        Log.error(errorMessage);
        throw H5FileException(errorMessage);
    }


    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_UINT, arraySpace);

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(array, H5::PredType::NATIVE_UINT);

    // That's it

    return;
}














// HDF5File::writeArray()  for 1D float arrays
//
// PURPOSE: write a 1D array to a specified group in the HDF5 file.
//
// INPUT: array: 1D float native array
//        groupName: name of an existing HDF5 Group in the file. Starts with "/".
//        arrayName: unique name of the array in the group, e.g. "starXcoordinates000001"
//
// OUTPUT: None

void HDF5File::writeArray(string groupName, string arrayName, float* array, int size)
{
     // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[1] = {size};
    H5::DataSpace arraySpace(1, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }


    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_FLOAT, arraySpace);

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(array, H5::PredType::NATIVE_FLOAT);

    // That's it

    return;
}











// HDF5File::writeArray()  for 1D double-valued arrays
//
// PURPOSE: write a 1D array to a specified group in the HDF5 file.
//
// INPUT: array: 1D double native array
//        groupName: name of an existing HDF5 Group in the file. Starts with "/".
//        arrayName: unique name of the array in the group, e.g. "DecOpticalAxis"
//
// OUTPUT: None

void HDF5File::writeArray(string groupName, string arrayName, double* array, int size)
{
    // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[1] = {size};
    H5::DataSpace arraySpace(1, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }


    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_DOUBLE, arraySpace);

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(array, H5::PredType::NATIVE_DOUBLE);

    // That's it

    return;
}











// HDF5File::writeArray()  for 2D float armadillo arrays
//
// PURPOSE: write a 2D armadillo array to a specified group in the HDF5 file.
//
// INPUT: array: 2D armadillo array
//        groupName: name of an existing HDF5 Group in the file. Starts with "/".
//        arrayName: unique name of the array in the group, e.g. "image000001"
//
// OUTPUT: None
template<typename T>
void HDF5File::writeArray(string groupName, string arrayName, const arma::Mat<T>& A)
{
    // Sanity check on the shape of the array

    if ((A.n_rows == 0) && (A.n_cols == 0))
    {
        throw H5FileException("HDF5File::writeArray(): encountered array with shape (0,0)");
    }

    H5::PredType predType = getPredType(A);

    // Create a DataSpace defining the shape and type of the data 

    unsigned int Ndimensions = 2;
    unsigned long long shape[Ndimensions];
    shape[0] = A.n_rows;
    shape[1] = A.n_cols;
    H5::DataSpace arraySpace(Ndimensions, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }

    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), predType, arraySpace);
    
    // Copy the Armadillo array to a vector, because the internally Armadillo stores the data column-major
    // while HDF5 assumes data to be stored row-major

    vector<T> temp(A.n_rows * A.n_cols);
    for (int n = 0; n < A.n_rows; n++)
    {
        for(int k = 0; k < A.n_cols; k++)
        {
            const int nk = n * A.n_cols + k;
            temp[nk] = A(n,k);
        }
    }

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(temp.data(), predType);

    // That's it

    return;
}




/**
 * \brief Check which PredType to use two write the given matrix to an HDF5 file.
 * 
 * \param A: 2D armadillo array
 * 
 * \return PredType to use to write the given matrix to an HDF5file
 */
template <class T>
H5::PredType HDF5File::getPredType(const arma::Mat<T>& A)
{
    if(typeid(T) == typeid(float))
        return H5::PredType::NATIVE_FLOAT;
    if(typeid(T) == typeid(uint16_t))
        return H5::PredType::NATIVE_UINT16;

   throw H5FileException("HDF5File::getPredType(): supported datatypes are float and uint16_t");
}




void HDF5File::writeArray(string groupName, string arrayName, const arma::Mat<uint16_t>& A)
{
    // Sanity check on the shape of the array

    if ((A.n_rows == 0) && (A.n_cols == 0))
    {
        throw H5FileException("HDF5File::writeArray(): encountered array with shape (0,0)");
    }

    // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[2] = {A.n_rows, A.n_cols};
    H5::DataSpace arraySpace(2, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }

    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_UINT16, arraySpace);
    
    // Copy the Armadillo array to a vector, because the internally Armadillo stores the data column-major
    // while HDF5 assumes data to be stored row-major

    vector<uint16_t> temp(A.n_rows * A.n_cols);
    for (int n = 0; n < A.n_rows; n++)
    {
        for(int k = 0; k < A.n_cols; k++)
        {
            const int nk = n * A.n_cols + k;
            temp[nk] = A(n,k);
        }
    }

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(temp.data(), H5::PredType::NATIVE_UINT16);

    // That's it

    return;
}



void HDF5File::writeArray(string groupName, string arrayName, const arma::Mat<float>& A)
{
    // Sanity check on the shape of the array

    if ((A.n_rows == 0) && (A.n_cols == 0))
    {
        throw H5FileException("HDF5File::writeArray(): encountered array with shape (0,0)");
    }

    // Create a DataSpace defining the shape and type of the data 

    const hsize_t shape[2] = {A.n_rows, A.n_cols};
    H5::DataSpace arraySpace(2, shape);

    // Check if the array is not already in the file.
    // There seems to be only a dirty way of determining this:
    // try to access the dataset, and check if an exception is thrown.

    bool arrayIsAlreadyInFile = true;
    string arrayPath = groupName + "/" + arrayName;

    try 
    {  
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        H5::DataSet testDataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        arrayIsAlreadyInFile = false;
    }

    if (arrayIsAlreadyInFile)
    {
        throw H5GroupException("HDF5File::writeArray(): array " + groupName + "/" + arrayName + " already in file.");
    }

    // Inside the Images group, make room for the image array

    H5::DataSet arrayDataset = file->createDataSet(arrayPath.c_str(), H5::PredType::NATIVE_FLOAT, arraySpace);
    
    // Copy the Armadillo array to a vector, because the internally Armadillo stores the data column-major
    // while HDF5 assumes data to be stored row-major

    vector<float> temp(A.n_rows * A.n_cols);
    for (int n = 0; n < A.n_rows; n++)
    {
        for(int k = 0; k < A.n_cols; k++)
        {
            const int nk = n * A.n_cols + k;
            temp[nk] = A(n,k);
        }
    }

    // Copy the data from our image into the HDF5 file

    arrayDataset.write(temp.data(), H5::PredType::NATIVE_FLOAT);

    // That's it

    return;
}



// HDF5File::readArray() for 2D float armadillo arrays
//
// PURPOSE: read a 2D array from a specified group in the HDF5 file into an
//          armadillo array.
//
// INPUT: array:     2D armadillo array. Previous contents will be lost.
//        groupName: name of an existing HDF5 Group in the file. Starts with "/".
//        arrayName: unique name of the array in the group, e.g. "image000001"
//
// OUTPUT: None
void HDF5File::readArray(string groupName, string arrayName, arma::Mat<float>& A)
{
    // Construct the path of the dataset in the HDF5 file

    string arrayPath = groupName + "/" + arrayName;

    // Try to open the dataset

    H5::DataSet dataset;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        dataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        throw H5DatasetException("HDF5File::readArray(): " + arrayPath + " not in file.");
    }

    // Find out the number of rows and columns of the dataset

    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t shape[2];
    unsigned int Ndimensions = dataspace.getSimpleExtentDims(shape, NULL);
    int Nrows = shape[0];
    int Ncolumns = shape[1];


    // Make a temporary vector<> object. We cannot read directly into the Armadillo array
    // because the latter are column-major while HDF5 stores data row-major

    vector<float> temp(Nrows*Ncolumns);

    // Read the HDF5 dataset into the array

    dataset.read(temp.data(), H5::PredType::NATIVE_FLOAT);

    // Reset the size for the 2D Armadillo array, and copy the data.

    A.reset();
    A.set_size(Nrows, Ncolumns);

    for (int n = 0; n < A.n_rows; n++)
    {
        for(int k = 0; k < A.n_cols; k++)
        {
            const int nk = n * A.n_cols + k;
            A(n,k) = temp[nk];                   // Is stored column-major
        }
    }

    // That's it

    return;
}












/**
 * \brief  Read a 1D double array from a specified group in the HDF5 file into a vector<double>.
 * 
 * \param groupName  Name of an existing HDF5 Group in the file. Starts with "/".
 * \param arrayName  Unique name of the array in the group, e.g. "skyBackground"
 * \param vec        C++ vector<double>. Previous contents will be lost.
 *
 */

void HDF5File::readArray(string groupName, string arrayName, vector<double> &vec)
{
    // Construct the path of the dataset in the HDF5 file

    string arrayPath = groupName + "/" + arrayName;

    // Try to open the dataset

    H5::DataSet dataset;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        dataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        throw H5DatasetException("HDF5File::readArray(): " + arrayPath + " not in file.");
    }

    // Find out the size of the dataset

    H5::DataSpace dataspace = dataset.getSpace();

    int rank = dataspace.getSimpleExtentNdims();
    if (rank != 1)
    {
        throw H5DatasetException("HDF5File::readArray(): " + arrayPath + " is not 1D.");
    }

    hsize_t shape[1];
    unsigned int Ndimensions = dataspace.getSimpleExtentDims(shape, NULL);
    int size = shape[0];


    // Ensure that the vector has enough space to read all data

    vec.clear();
    vec.resize(size);

    // Read the HDF5 dataset into the array

    dataset.read(vec.data(), H5::PredType::NATIVE_DOUBLE);

    // That's it

    return;
}












/**
 * \brief  Read a 1D (unsigned int) array from a specified group in the HDF5 file into a vector<unsigned int>
 *
 * \param groupName  Name of an existing HDF5 Group in the file. Starts with "/".
 * \param arrayName  Unique name of the array in the group, e.g. "skyBackground"
 * \param vec        C++ vector<double>. Previous contents will be lost.
 *
 */

void HDF5File::readArray(string groupName, string arrayName, vector<unsigned int> &vec)
{
    // Construct the path of the dataset in the HDF5 file

    string arrayPath = groupName + "/" + arrayName;

    // Try to open the dataset

    H5::DataSet dataset;

    try
    {
        // Turn off the auto-printing when an exception is raised

        H5::Exception::dontPrint();

        // Try to open the dataset

        dataset = file->openDataSet(arrayPath.c_str());
    }
    catch (H5::FileIException error)
    {
        throw H5DatasetException("HDF5File::readArray(): " + arrayPath + " not in file.");
    }

    // Find out the size of the dataset

    H5::DataSpace dataspace = dataset.getSpace();

    int rank = dataspace.getSimpleExtentNdims();
    if (rank != 1)
    {
        throw H5DatasetException("HDF5File::readArray(): " + arrayPath + " is not 1D.");
    }

    hsize_t shape[1];
    unsigned int Ndimensions = dataspace.getSimpleExtentDims(shape, NULL);
    int size = shape[0];


    // Ensure that the vector has enough space to read all data

    vec.clear();
    vec.resize(size);

    // Read the HDF5 dataset into the array

    dataset.read(vec.data(), H5::PredType::NATIVE_UINT);

    // That's it

    return;
}












/**
 * \brief: include version of the simulator in the HDF5 file,.
 */

void HDF5File::writeVersionInformation()
{
    Log.info("HDF5File: writing version information to HDF5");

    // Make the parent group

    string parentGroup = "/Version";
    createGroup(parentGroup);

    writeAttribute(parentGroup, "Application", string("PlatoSim3"));
    writeAttribute(parentGroup, "GitVersion", string(GIT_DESCRIBE));

}











// /** TODO implement idependent time column!
//  * \brief: include the tranmsissionEfficiency values to the HDF5 file.
//  *
//  */
// void HDF5File::writeTime(double* array, int size)
// {
//     writeArray("Time/", "time", array, size);
// }










/**
 * \brief: include the tranmsissionEfficiency values to the HDF5 file.
 *
 */
void HDF5File::writeTransmissionEfficiencyValues(double* array, int size)
{
    writeArray("TransmissionEfficiency/", "transmissionEfficiency", array, size);
}





/**
 * /brief: Write smearing map into the HDF5 file.
 *
 */
void HDF5File::writeSmearingMap(arma::Mat<float>& smearingMap, bool includeQuantisation, int exposureNr)
{
    // Clear the string stream and compose the smearing map name
    stringstream myStream;
    myStream.str(string());      // insert empty string
    myStream.clear();            // clear eof bit

    myStream << "smearingMap" << setfill('0') << setw(7) << exposureNr;
    string smearingMapName = myStream.str();

    // Add the smearing map to the "SmearingMaps" group

    if (!includeQuantisation)
    {
        // Write the float array to HDF5
        writeArray("/SmearingMaps", smearingMapName, smearingMap);
    }
    else
    {
        if ((smearingMap.min() < 0) || (smearingMap.max() >= (1 << 16)))
        {
            throw ConfigurationException("Detector: quantisation was applied but smearing map values are not in [0, 2^16]");
        }

        // Convert the float matrix to an unsigned uint16_t matrix
	
        arma::Mat<uint16_t> uintMap = arma::conv_to<arma::Mat<uint16_t>>::from(smearingMap);
        writeArray("/SmearingMaps", smearingMapName, uintMap);
    }

}









/**
 * \brief: include the Straylight to the HDF5 file.
 *
 */  
void HDF5File::writeStraylight(vector<double>& straylight)
{
  writeArray("/Straylight/", "Moon", straylight.data(), straylight.size());

}



/**
 * \brief: includes the TelescopeACS to the HDF5 file.
 *
 * NOTE Time columns is written here and includes CCD code offset.
 */
void HDF5File::writeTelescopeACS(vector<double>& time, vector<double>& RA, vector<double>& dec,
				 vector<double>& yaw, vector<double>& pitch, vector<double>& roll)
{
    writeArray("/Telescope/", "time",           time.data(),    time.size());
    writeArray("/Telescope/", "telescopeRA",    RA.data(),      RA.size());         // [deg]
    writeArray("/Telescope/", "telescopeDec",   dec.data(),     dec.size());        // [deg]
    writeArray("/Telescope/", "telescopeYaw",   yaw.data(),     yaw.size());        // [arcsec]
    writeArray("/Telescope/", "telescopePitch", pitch.data(),   pitch.size());      // [arcsec]
    writeArray("/Telescope/", "telescopeRoll",  roll.data(),    roll.size());       // [arcsec]
}













/**
 * /brief: save the star positions to the HDF5 file.
 *
 * NOTE: keyValuePair is (key, value) pair, where key is also a pair consisting of the startTime and StarID
 */
void HDF5File::writeStarPositionByExposure(map<double, map<unsigned int, array<double, 6>>>& detectedStarInfo, int beginExposureNr)
{

    Log.info("HDF5File: writing star positions to HDF5 file");

    vector<double> time;
    for(auto keyValuePair: detectedStarInfo) time.push_back(keyValuePair.first);


    if (!time.empty())
    {
      writeArray("StarPositions/", "time", time.data(), time.size());
    }
    else
    {
      Log.warning("HDF5File: No star positions to write to HDF5 file.");
    }

    // For each of the exposures, make a subgroup and write the position and flux of all detected stars.
    // Because some stars at the edge may jitter in and out of the subfield from one exposure to the other,
    // the written arrays may not be equally long for each exposure.

    for (int n = 0; n < time.size(); n++)
    {
      // Make the sub-group

      stringstream myStream;
      myStream << "Exposure" << setfill('0') << setw(7) << beginExposureNr + n;
      const string exposureGroupName = "/StarPositions/" + myStream.str();
      createGroup(exposureGroupName);

      // Collect the different time series. For the positions, we only compute the sum, so we still need
      // to divide by N to compute the average, where N is the number of times the star was detected to be
      // in the subfield during an exposure.

      vector<unsigned int> starIDs;
      vector<double> xFPmm;
      vector<double> yFPmm;
      vector<double> rowPix;
      vector<double> colPix;
      vector<double> flux;

      for(auto keyValuePair: detectedStarInfo[time[n]])
      {
	const unsigned int starID = keyValuePair.first;
	starIDs.push_back(starID);                       // list of starIDs for this exposure only
	xFPmm.push_back(detectedStarInfo[time[n]][starID][0] / detectedStarInfo[time[n]][starID][5]);
	yFPmm.push_back(detectedStarInfo[time[n]][starID][1] / detectedStarInfo[time[n]][starID][5]);
	rowPix.push_back(detectedStarInfo[time[n]][starID][2] / detectedStarInfo[time[n]][starID][5]);
	colPix.push_back(detectedStarInfo[time[n]][starID][3] / detectedStarInfo[time[n]][starID][5]);
	flux.push_back(detectedStarInfo[time[n]][starID][4]);
      }

      // Write the time series to HDF5

      if(!starIDs.empty())
      {
	writeArray(exposureGroupName, "starID", starIDs.data(), starIDs.size());
	writeArray(exposureGroupName, "xFPmm",  xFPmm.data(),   xFPmm.size());
	writeArray(exposureGroupName, "yFPmm",  yFPmm.data(),   yFPmm.size());
	writeArray(exposureGroupName, "rowPix", rowPix.data(),  rowPix.size());
	writeArray(exposureGroupName, "colPix", colPix.data(),  colPix.size());
	writeArray(exposureGroupName, "flux",   flux.data(),    flux.size());
      }
    }
}












/**
 * /brief: save the star positions to the HDF5 file.
 * /Note: This is the new way of doing things!
 *
 */
void HDF5File::writeStarPositionByStarID(map<double, map<unsigned int, array<double, 6>>>& detectedStarInfo, vector<unsigned int> starIDs)
{
    Log.info("HDF5File: writing star positions to HDF5 file");
    map<unsigned int, map<double, array<double, 6>>> transformedDetectedStarInfo;
    vector<double> time;
    for(auto keyValuePair: detectedStarInfo) time.push_back(keyValuePair.first);

    // if (!time.empty())
    // {
    //   writeArray("StarPositions/", "Time", time.data(), time.size());
    // }
    // else
    // {
    //   Log.warning("HDF5File: No star positions to write to HDF5 file.");
    // }
    
    if (time.empty())
    {
      Log.warning("HDF5File: No star positions to write to HDF5 file.");
    }

    // For each starID, make a subgroup and write the position and flux of all detected stars.
    // Because some stars at the edge may jitter in and out of the subfield from one exposure to the other,
    // the written arrays may not be equally long for each exposure.

    for (int n = 0; n < time.size(); n++)
    {
      for(auto keyValuePair: detectedStarInfo[time[n]])
      {
	const unsigned int starID = keyValuePair.first;
	transformedDetectedStarInfo[starID][time[n]] = detectedStarInfo[time[n]][starID];
      }
    }


    for (int n = 0; n < starIDs.size(); n++)
    {

      // Collect the different time series. For the positions, we only compute the sum, so we still need
      // to divide by N to compute the average, where N is the number of times the star was detected to be
      // in the subfield during an exposure.

      vector<unsigned int> times;
      vector<double> xFPmm;
      vector<double> yFPmm;
      vector<double> rowPix;
      vector<double> colPix;
      vector<double> flux;

      for(auto keyValuePair: transformedDetectedStarInfo[starIDs[n]])
      {
	const double time = keyValuePair.first;
	times.push_back(time);
	xFPmm.push_back(transformedDetectedStarInfo[starIDs[n]][time][0] / transformedDetectedStarInfo[starIDs[n]][time][5]);
	yFPmm.push_back(transformedDetectedStarInfo[starIDs[n]][time][1] / transformedDetectedStarInfo[starIDs[n]][time][5]);
	rowPix.push_back(transformedDetectedStarInfo[starIDs[n]][time][2] / transformedDetectedStarInfo[starIDs[n]][time][5]);
	colPix.push_back(transformedDetectedStarInfo[starIDs[n]][time][3] / transformedDetectedStarInfo[starIDs[n]][time][5]);
	flux.push_back(transformedDetectedStarInfo[starIDs[n]][time][4]);
      }

      // Write the time series to HDF5

      if(!times.empty())
      {
        // Make the sub-group
        stringstream myStream;
        myStream << "starID" << setfill('0') << setw(7) << 0 + starIDs[n];
        const string exposureGroupName = "/StarPositions/" + myStream.str();
        createGroup(exposureGroupName);
        //writeArray(exposureGroupName, "time",   times.data(),   times.size());	
        writeArray(exposureGroupName, "xFPmm",  xFPmm.data(),   xFPmm.size());
        writeArray(exposureGroupName, "yFPmm",  yFPmm.data(),   yFPmm.size());
        writeArray(exposureGroupName, "rowPix", rowPix.data(),  rowPix.size());
        writeArray(exposureGroupName, "colPix", colPix.data(),  colPix.size());
        writeArray(exposureGroupName, "flux",   flux.data(),    flux.size());
      }
    }
}















/**
 * /brief: save the pointlike ghosts to the HDF5 file.
 * /Note: This will group the ghost by exposure.
 *
 */
void HDF5File::writePointlikeGhostByExposure(map<double, map<unsigned int, array<double, 6>>>& detectedPointLikeGhostInfo, int beginExposureNr)
{

    Log.info("HDF5File: writing pointlike ghost positions to HDF5 file");
    createGroup("/PointLikeGhostPositions");
    vector<double> time;
    for(auto keyValuePair: detectedPointLikeGhostInfo) time.push_back(keyValuePair.first);

    // if (!time.empty())
    // {
    //     writeArray("PointLikeGhostPositions/", "Time", time.data(), time.size());
    // }
    // else
    // {
    //     Log.warning("HDF5File: No point-like ghost positions to write to HDF5 file.");
    // }
    
    if (time.empty())
    {
        Log.warning("HDF5File: No point-like ghost positions to write to HDF5 file.");
    }

    for (int n = 0; n < time.size(); n++)
    {
      stringstream myStream;
      myStream << "Exposure" << setfill('0') << setw(7) << beginExposureNr + n;


      vector<unsigned int> starIDs;
      vector<double> xFPmm;
      vector<double> yFPmm;
      vector<double> rowPix;
      vector<double> colPix;
      vector<double> flux;
      vector<double> ghostRadius;

      for(auto keyValuePair: detectedPointLikeGhostInfo[time[n]])
      {
          const unsigned int starID = keyValuePair.first;
          starIDs.push_back(starID);                       // list of starIDs for this exposure only
          xFPmm.push_back(detectedPointLikeGhostInfo[time[n]][starID][0] / detectedPointLikeGhostInfo[time[n]][starID][5]);
          yFPmm.push_back(detectedPointLikeGhostInfo[time[n]][starID][1] / detectedPointLikeGhostInfo[time[n]][starID][5]);
          rowPix.push_back(detectedPointLikeGhostInfo[time[n]][starID][2] / detectedPointLikeGhostInfo[time[n]][starID][5]);
          colPix.push_back(detectedPointLikeGhostInfo[time[n]][starID][3] / detectedPointLikeGhostInfo[time[n]][starID][5]);
          flux.push_back(detectedPointLikeGhostInfo[time[n]][starID][4]);
      }

      const string pointLikeGhostGroupName = "/PointLikeGhostPositions/" + myStream.str();
      createGroup(pointLikeGhostGroupName);

      // Write the info for the point-like ghost star positions to HDF5
      if(!starIDs.empty())
      {
          writeArray(pointLikeGhostGroupName, "starID", starIDs.data(), starIDs.size());
          writeArray(pointLikeGhostGroupName, "xFPmm",  xFPmm.data(),   xFPmm.size());
          writeArray(pointLikeGhostGroupName, "yFPmm",  yFPmm.data(),   yFPmm.size());
          writeArray(pointLikeGhostGroupName, "rowPix", rowPix.data(),  rowPix.size());
          writeArray(pointLikeGhostGroupName, "colPix", colPix.data(),  colPix.size());
          writeArray(pointLikeGhostGroupName, "flux",   flux.data(),    flux.size());
      }

      // Write the info for the point like ghost star positions to HDF5

      Log.info("HDF5File: writing point like ghost positions to HDF5 file");

      starIDs.clear();
      xFPmm.clear();
      yFPmm.clear();
      rowPix.clear();
      colPix.clear();
      flux.clear();
      ghostRadius.clear();
    }
    time.clear();
}














/**
 * /brief: save the pointlike ghosts to the HDF5 file.
 * /Note: This will group the ghost by star id.
 *
 */
void HDF5File::writePointlikeGhostByStarID(map<double, map<unsigned int, array<double, 6>>>& detectedPointLikeGhostInfo)
{

    Log.info("HDF5File: writing pointlike ghost positions to HDF5 file");
    createGroup("/PointLikeGhostPositions");
    map<unsigned int, map<double, array<double, 6>>> transformedDetectedPointLikeGhostInfo;
    vector<unsigned int> starIDs;
    vector<double> time;
    for(auto keyValuePair: detectedPointLikeGhostInfo) time.push_back(keyValuePair.first);

    // if (!time.empty())
    // {
    //     writeArray("PointLikeGhostPositions/", "Time", time.data(), time.size());
    // }
    // else
    // {
    //     Log.warning("HDF5File: No point-like ghost positions to write to HDF5 file.");
    // }
    
    if (time.empty())
    {
        Log.warning("HDF5File: No point-like ghost positions to write to HDF5 file.");
    }

    // For each starID, make a subgroup and write the position and flux of all detected pointlike ghosts.
    // Because some ghosts at the edge may jitter in and out of the subfield from one exposure to the other,
    // the written arrays may not be equally long for each exposure.

    for (int n = 0; n < time.size(); n++)
    {
      for(auto keyValuePair: detectedPointLikeGhostInfo[time[n]])
      {
        const unsigned int starID = keyValuePair.first;
        if ( find(starIDs.begin(), starIDs.end(), starID) == starIDs.end())
        {
            starIDs.push_back(starID);
        }
        transformedDetectedPointLikeGhostInfo[starID][time[n]] = detectedPointLikeGhostInfo[time[n]][starID];
      }
    }


    for (int n = 0; n < starIDs.size(); n++)
    {

      // Write the info for the point-like ghost star positions to HDF5

      vector<unsigned int> times;
      vector<double> xFPmm;
      vector<double> yFPmm;
      vector<double> rowPix;
      vector<double> colPix;
      vector<double> flux;
      vector<double> ghostRadius;

      for(auto keyValuePair: transformedDetectedPointLikeGhostInfo[starIDs[n]])
      {
          const unsigned int time = keyValuePair.first;

          times.push_back(time);                       // list of times for this starID only
          xFPmm.push_back(transformedDetectedPointLikeGhostInfo[starIDs[n]][time][0] / transformedDetectedPointLikeGhostInfo[starIDs[n]][time][5]);
          yFPmm.push_back(transformedDetectedPointLikeGhostInfo[starIDs[n]][time][1] / transformedDetectedPointLikeGhostInfo[starIDs[n]][time][5]);
          rowPix.push_back(transformedDetectedPointLikeGhostInfo[starIDs[n]][time][2] / transformedDetectedPointLikeGhostInfo[starIDs[n]][time][5]);
          colPix.push_back(transformedDetectedPointLikeGhostInfo[starIDs[n]][time][3] / transformedDetectedPointLikeGhostInfo[starIDs[n]][time][5]);
          flux.push_back(transformedDetectedPointLikeGhostInfo[starIDs[n]][time][4]);
      }

      // Write the info for the point like ghost star positions to HDF5

      if(!times.empty())
      {
          // Make the subgroup
          stringstream myStream;
          myStream << "StarID" << setfill('0') << setw(7) << 0 + starIDs[n];
          const string pointLikeGhostGroupName = "/PointLikeGhostPositions/" + myStream.str();
          createGroup(pointLikeGhostGroupName);
          //writeArray(pointLikeGhostGroupName, "time",   times.data(),   times.size());  
          writeArray(pointLikeGhostGroupName, "xFPmm",  xFPmm.data(),   xFPmm.size());
          writeArray(pointLikeGhostGroupName, "yFPmm",  yFPmm.data(),   yFPmm.size());
          writeArray(pointLikeGhostGroupName, "rowPix", rowPix.data(),  rowPix.size());
          writeArray(pointLikeGhostGroupName, "colPix", colPix.data(),  colPix.size());
          writeArray(pointLikeGhostGroupName, "flux",   flux.data(),    flux.size());
      }

      Log.info("HDF5File: writing point like ghost positions to HDF5 file");

      xFPmm.clear();
      yFPmm.clear();
      rowPix.clear();
      colPix.clear();
      flux.clear();
      ghostRadius.clear();
    }
    starIDs.clear();

}








/**
 * /brief: save the extended ghosts to the HDF5 file.
 * /Note: This will group the extended ghosts by exposure.
 *
 */
void HDF5File::writeExtendedGhostByExposure(map<double, map<unsigned int, array<double, 7>>>& detectedExtendedGhostInfo, int beginExposureNr)
{
  Log.info("HDF5File: writing extended ghost positions to HDF5 file");
  createGroup("/ExtendedGhostPositions");
  vector<double> time;
  for(auto keyValuePair: detectedExtendedGhostInfo) time.push_back(keyValuePair.first);

  // if (!time.empty())
  // {
  //   writeArray("ExtendedGhostPositions/", "Time", time.data(), time.size());
  // }
  // else
  // {
  //   Log.warning("HDF5File: No extended ghost positions to write to HDF5 file.");
  // }
  
  if (time.empty())
  {
    Log.warning("HDF5File: No extended ghost positions to write to HDF5 file.");
  }

  for (int n = 0; n < time.size(); n++)
  {
    stringstream myStream;
    myStream << "Exposure" << setfill('0') << setw(7) << beginExposureNr + n;

    // Write the info for the extended ghost star positions to HDF5

    vector<unsigned int> starIDs;
    vector<double> xFPmm;
    vector<double> yFPmm;
    vector<double> rowPix;
    vector<double> colPix;
    vector<double> flux;
    vector<double> ghostRadius;


    for(auto keyValuePair: detectedExtendedGhostInfo[time[n]])
    {
      const unsigned int starID = keyValuePair.first;
      starIDs.push_back(starID);                       // list of starIDs for this exposure only
      xFPmm.push_back(detectedExtendedGhostInfo[time[n]][starID][0] / detectedExtendedGhostInfo[time[n]][starID][5]);
      yFPmm.push_back(detectedExtendedGhostInfo[time[n]][starID][1] / detectedExtendedGhostInfo[time[n]][starID][5]);
      rowPix.push_back(detectedExtendedGhostInfo[time[n]][starID][2] / detectedExtendedGhostInfo[time[n]][starID][5]);
      colPix.push_back(detectedExtendedGhostInfo[time[n]][starID][3] / detectedExtendedGhostInfo[time[n]][starID][5]);
      flux.push_back(detectedExtendedGhostInfo[time[n]][starID][4]);
      ghostRadius.push_back(detectedExtendedGhostInfo[time[n]][starID][6] / detectedExtendedGhostInfo[time[n]][starID][5]);
    }

    const string extendedGhostGroupName = "/ExtendedGhostPositions/" + myStream.str();
    createGroup(extendedGhostGroupName);

    if(!starIDs.empty())
    {
      writeArray(extendedGhostGroupName, "starID", starIDs.data(), starIDs.size());
      writeArray(extendedGhostGroupName, "xFPmm",  xFPmm.data(),   xFPmm.size());
      writeArray(extendedGhostGroupName, "yFPmm",  yFPmm.data(),   yFPmm.size());
      writeArray(extendedGhostGroupName, "rowPix", rowPix.data(),  rowPix.size());
      writeArray(extendedGhostGroupName, "colPix", colPix.data(),  colPix.size());
      writeArray(extendedGhostGroupName, "flux",   flux.data(),    flux.size());
      writeArray(extendedGhostGroupName, "radius", ghostRadius.data(), ghostRadius.size());
    }

    Log.info("HDF5File: writing extended ghosts positions to HDF5 file");

    starIDs.clear();
    xFPmm.clear();
    yFPmm.clear();
    rowPix.clear();
    colPix.clear();
    flux.clear();
    ghostRadius.clear();
  }
  time.clear();
}














/**
 * /brief: save the extended ghosts to the HDF5 file.
 * /Note: This will group the extended ghosts by star id.
 *
 */
void HDF5File::writeExtendedGhostByStarID(map<double, map<unsigned int, array<double, 7>>>& detectedExtendedGhostInfo)
{
  Log.info("HDF5File: writing extended ghost positions to HDF5 file");
  createGroup("/ExtendedGhostPositions");
  map<unsigned int, map<double, array<double, 7>>> transformedDetectedExtendedGhostInfo;
  vector<unsigned int> starIDs;
  vector<double> time;
  for(auto keyValuePair: detectedExtendedGhostInfo) time.push_back(keyValuePair.first);

  // if (!time.empty())
  // {
  //   writeArray("ExtendedGhostPositions/", "Time", time.data(), time.size());
  // }
  // else
  // {
  //   Log.warning("HDF5File: No extended ghost positions to write to HDF5 file.");
  // }
  
  if (time.empty())
  {
    Log.warning("HDF5File: No extended ghost positions to write to HDF5 file.");
  }

  // For each starID, make a subgroup and write the position and flux of all detected extended ghosts.
  // Because some ghosts at the edge may jitter in and out of the subfield from one exposure to the other,
  // the written arrays may not be equally long for each exposure.

  for (int n = 0; n < time.size(); n++)
  {
      for(auto keyValuePair: detectedExtendedGhostInfo[time[n]])
      {
          const unsigned int starID = keyValuePair.first;
          if ( find(starIDs.begin(), starIDs.end(), starID) == starIDs.end())
          {
              starIDs.push_back(starID);
          }
          transformedDetectedExtendedGhostInfo[starID][time[n]] = detectedExtendedGhostInfo[time[n]][starID];
      }
  }

  for (int n = 0; n < starIDs.size(); n++)
  {

    // Write the info for the extended ghost star positions to HDF5

    vector<unsigned int> times;
    vector<double> xFPmm;
    vector<double> yFPmm;
    vector<double> rowPix;
    vector<double> colPix;
    vector<double> flux;
    vector<double> ghostRadius;
    for(auto keyValuePair: transformedDetectedExtendedGhostInfo[starIDs[n]])
    {
      const unsigned int time = keyValuePair.first;

      times.push_back(time);                       // list of times for this starID only
      xFPmm.push_back(detectedExtendedGhostInfo[starIDs[n]][time][0] / detectedExtendedGhostInfo[starIDs[n]][time][5]);
      yFPmm.push_back(detectedExtendedGhostInfo[starIDs[n]][time][1] / detectedExtendedGhostInfo[starIDs[n]][time][5]);
      rowPix.push_back(detectedExtendedGhostInfo[starIDs[n]][time][2] / detectedExtendedGhostInfo[starIDs[n]][time][5]);
      colPix.push_back(detectedExtendedGhostInfo[starIDs[n]][time][3] / detectedExtendedGhostInfo[starIDs[n]][time][5]);
      flux.push_back(detectedExtendedGhostInfo[starIDs[n]][time][4]);
      ghostRadius.push_back(detectedExtendedGhostInfo[starIDs[n]][time][6] / detectedExtendedGhostInfo[starIDs[n]][time][5]);
    }

    // Write the info for the point like ghost star positions to HDF5

    if(!starIDs.empty())
    {
      // Make the subgroup
      stringstream myStream;
      myStream << "Exposure" << setfill('0') << setw(7) << 0 + starIDs[n];
      const string extendedGhostGroupName = "/ExtendedGhostPositions/" + myStream.str();
      createGroup(extendedGhostGroupName);
      //writeArray(extendedGhostGroupName, "times",  times.data(),       times.size());      
      writeArray(extendedGhostGroupName, "xFPmm",  xFPmm.data(),       xFPmm.size());
      writeArray(extendedGhostGroupName, "yFPmm",  yFPmm.data(),       yFPmm.size());
      writeArray(extendedGhostGroupName, "rowPix", rowPix.data(),      rowPix.size());
      writeArray(extendedGhostGroupName, "colPix", colPix.data(),      colPix.size());
      writeArray(extendedGhostGroupName, "flux",   flux.data(),        flux.size());
      writeArray(extendedGhostGroupName, "radius", ghostRadius.data(), ghostRadius.size());
    }

    Log.info("HDF5File: writing extended ghost positions to HDF5 file");

    xFPmm.clear();
    yFPmm.clear();
    rowPix.clear();
    colPix.clear();
    flux.clear();
    ghostRadius.clear();
  }
  starIDs.clear();
}


template<typename T>
void HDF5File::addDataToRaggedArray(map<string, vector<T>>& data, map<string, H5::DataSet>& datasets, const hsize_t offsetVal, const hsize_t n)
{
    const H5::PredType &dataType = Hdf5PredType<T>::get();

    hsize_t newSize[1] = { offsetVal + n };
    hsize_t offset[1]  = {offsetVal};
    hsize_t count[1]   = { n };

    for (const auto& [name, values] : data)
    {
      H5::DataSet& dataset = datasets.at(name);
      dataset.extend(newSize);

      H5::DataSpace fileSlab = dataset.getSpace();
      fileSlab.selectHyperslab(H5S_SELECT_SET, count, offset);
      H5::DataSpace memSpace(1, count);

      dataset.write(values.data(), dataType, memSpace, fileSlab);
    }
}




/**
 * /brief: save the cosmics to the HDF5 file.
 *
 */
void HDF5File::writeCosmics(
    RaggedArray &array, int exposureNr, vector<unsigned int> &entryRows,
    vector<unsigned int> &entryColumns, vector<double> &trailLengths,
    vector<double> &entryAngles, vector<double> &intensities,
    vector<unsigned int> &rows, vector<unsigned int> &cols, vector<double> &flux)
{
    hsize_t newSize[1] = { array.runningTotal + entryRows.size() };
    hsize_t offset[1] = {array.runningTotal};
    hsize_t count[1] = {entryRows.size()};

    map<string, vector<unsigned int>> uintData = { {"entryRows",  entryRows}, // 
						   {"entryColumns", entryColumns},
						   {"rows", rows},
						   {"columns", cols} };
    map<string, vector<double>> doubleData = { {"trailLengths", trailLengths},
					      {"entryAngles", entryAngles},
					      {"intensities", intensities},
					      {"flux", flux} };

    size_t n = uintData.empty() ? doubleData.begin()->second.size()
                                   : uintData.begin()->second.size();

    map<string, H5::DataSet> uintDataset = array.uintDatasets;
    map<string, H5::DataSet> doubleDataset = array.doubleDatasets;
    
    addDataToRaggedArray(uintData, uintDataset, array.runningTotal, n);
    addDataToRaggedArray(doubleData, doubleDataset, array.runningTotal, n);

    // // We add the double values
    // for (const auto& [name, values] : doubleData)
    // {
    //   H5::DataSet& dataset = array.doubleDatasets.at(name);
    //   dataset.extend(newSize);

    //   H5::DataSpace fileSlab = dataset.getSpace();
    //   fileSlab.selectHyperslab(H5S_SELECT_SET, count, offset);
    //   H5::DataSpace memSpace(1, count);

    //   dataset.write(values.data(), H5::PredType::NATIVE_DOUBLE, memSpace, fileSlab);
    // }

    array.runningTotal += entryRows.size();
    array.offsets[exposureNr + 1] = array.runningTotal;

}





// fileExists()
//
// PURPOSE: Check if file exists. This is done by trying to open the file,
//          and verifying if the "good" flag is raised.
//
// INPUT: filename: full path of the file
//
// OUTPUT: true if the file exists, false otherwise

bool fileExists(string filename)
{
    ifstream myFile(filename.c_str());

    if (myFile.good())
    {
        myFile.close();
        return true;
    }
    else
    {
        myFile.close();
        return false;
    }
}



void HDF5File::createGroup(string groupName, string arrayName, hsize_t dims[3], const H5::PredType &type)
{
    // Make sure the path of the group starts with "/" (i.e. the root of the
    // folder)
    if (!groupName.compare(0, 1, "/"))
        groupName.insert(0, "/");

    // Find the parent group
    // E.g. if the groupName is "/my/path/to/group1" then the parent group is
    //      "/my/path/to" and the subgroup is "group1".
    auto position = groupName.find_last_of("/");
    string parentGroupName = groupName.substr(0, position);
    string subGroupName = groupName.substr(position + 1);

    // If the parent group name is empty, it means that the group should be
    // placed in the root group. In that case, make sure that the parentGroupName
    // is "/".
    if (parentGroupName.empty())
    {
        parentGroupName = "/";
    }

    // Open the parent group
    H5::Group parentGroup = file->openGroup(parentGroupName.c_str());
    parentGroup.createGroup(groupName.c_str());

    // Fill the group with an array of the correct dimension
    string arrayPath = groupName + "/" + arrayName;
    H5::DataSpace fileSpace(3, dims); // defines the shape of the data

    // to fill the dataset efficiently we can define the chunks that will
    // fill the array.
    hsize_t chunkDims[3] = {1, dims[1], dims[2]};

    H5::DSetCreatPropList propList;
    propList.setChunk(3, chunkDims);
    propList.setDeflate(4); // compression of chunks

    // we start by making room for the image array
    file->createDataSet(arrayPath.c_str(), type,
                        fileSpace, propList);
}


// HDF5File::writeArray() for a 2D float armadillo array
//
// PURPOSE: write a 2D armadillo array to a specifigied group in the HDF5 file
//          for a certain time step.
// NOTE: the function assumes that the specified group contains a 3D array. This
//       array is filled as {timestep, data}. (where data is the 2D armadillo
//       array).
//
// OUTPUT: None
template <typename T>
void HDF5File::writeArray(string groupName, string arrayName, int timeStep,
                          const arma::Mat<T> &data)
{

    H5::PredType predType = getPredType(data);

    // Sanity check if the group exists
    if (!hasGroup(groupName))
    {
      throw H5FileException("HDF5File:writeArray(): Unknown group (" +
			    groupName + ") in HDF5 file " + file->getFileName());
    }

    if (!hasDataset(groupName, arrayName))
    {
      throw H5FileException("HDF5File:writeArray(): Unknown dataset (" + arrayName +
			    ") in group " + groupName + " of HDF5 file " + file->getFileName());
    }

    H5::Group group = file->openGroup(groupName.c_str());
    H5::DataSet dataset = group.openDataSet(arrayName.c_str());

    // Read back the dataset's shape
    H5::DataSpace fileSpace = dataset.getSpace();
    hsize_t dims[3];
    fileSpace.getSimpleExtentDims(dims);

    // Sanity check of the shape of the array
    if (!((data.n_rows == static_cast<int>(dims[1])) && (data.n_cols == static_cast<int>(dims[2]))))
    {
      throw H5FileException("HDF5File::writeArray(): encountered array with wrong shape,\nreceived shape (" + 
			    to_string(data.n_rows) + ", " + to_string(data.n_cols) + "), but expected shape (" + 
			    to_string(dims[1]) + ", " + to_string(dims[2]) + ")");

    }

    hsize_t offset[3] = {static_cast<hsize_t>(timeStep), 0, 0};
    hsize_t count[3] = {1, dims[1], dims[2]};

    fileSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memSpace(3, count);

    // Copy the Armadillo array to a vector
    vector<T> temp(data.n_rows * data.n_cols);
    for (int n = 0; n < data.n_rows; n++) {
      for (int k = 0; k < data.n_cols; k++) {
        const int nk = n * data.n_cols + k;
	temp[nk] = data(n, k);
      }
    }
    try {
      // dataset.write(data.data(), dataType, memSpace, fileSlab);
      dataset.write(temp.data(), predType, memSpace, fileSpace);
    }
    catch (H5::DataSetIException& e) {
      e.printErrorStack();   // dumps the full internal HDF5 error stack to stderr
      throw;  // or handle as appropriate
    }
}


template void HDF5File::writeArray<float>(std::string, std::string, int,
                                           const arma::Mat<float> &);
template void HDF5File::writeArray<uint16_t>(
    std::string, std::string, int, const arma::Mat<uint16_t>&);

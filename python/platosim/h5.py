#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract information from the PlatoSim HDF5 output file.

This class provides a direct interface to extract and inspect
the HDF5 output files that PlatoSim produce upon execution.

For usage, see the Jupyter tutorial notebooks available at:
"PlatoSim/docs/tutorials"
"""

# Built-in
import os

# PlatoSim standard
import h5py
import numpy as np
from platosim.utilities import fileMatch


#--------------------------------------------------------------#
#                          FUNCTIONS                           #
#--------------------------------------------------------------#


def h5paths(item, verbose=True, includeGroups=True,
            baseString="", flatList=None, classList=None):

    """Function to extract HDF5 paths.

    Parameters
    ----------
    item : HDF5 file 
        hdf5 file, or group from an hdf5 file
    verbose : bool
        If True, the attribue values are displayed on the output as well
    includeGroups : bool
        If False, only the final branches are returned -> to extract data
        If True, all nodes are returned, including the groups -> to display the structure
    baseString, flatList and classList: 
       Private variables, and mustn't be used

    Returns
    -------
    Two lists are returned:
        1. If includeGroups = False : 
           the list of full paths towards all 'final' objects (datasets and attributes)
           If includeGroups = True : 
           the list of full paths, incl. intermediate nodes (groups)
        2. The list of object type, indicated by the following string-codes: 
          'a' : attribute  (i.e. appearing below a h5py._hl.attrs.AttributeManager)
          'd' : dataset    (i.e. h5py._hl.dataset.Dataset)
    
    Example
    -------
    >>> paths, classes = h5paths(hfile)
    """

    # FORCE-INITIALISE THE LISTS ON FIRST CALL (without touching h5paths.__defaults__)
    # http://stackoverflow.com/questions/2335160/what-is-the-scope-of-a-defaulted-parameter-in-python

    if flatList is None: flatList = []
    if classList is None: classList = []    

    # CHECK IF ITEM HAS ATTRIBUTES

    groupkeys = item.attrs.keys()
    for key in groupkeys:
                if verbose: print("[a] " + baseString + '/' + key)
                flatList.append(str(baseString+'/'+key))
                classList.append('a')

    # GO OVER OTHER ITEMS : Handle Datasets and Recurse on Groups

    for i in item.items():
        if (isinstance(i[1],h5py._hl.group.Group)):
            if includeGroups:
                flatList += [str(baseString+'/'+ str(i[0]))]
                classList += ['g']
            flatList, classList = h5paths(i[1], verbose=verbose,
                                          baseString=baseString + '/' + i[0],
                                          flatList=flatList,
                                          classList=classList,
                                          includeGroups=includeGroups)
        elif(isinstance(i[1],h5py._hl.dataset.Dataset)):
            if verbose: print("[D] " + baseString + '/' + str(i[0]))
            flatList += [str(baseString+'/'+ str(i[0]))]
            classList += ['d']
        else:
            raise Exception("WARNING: Unexpected class: {}".format(i[1].__class__))
    return flatList,classList





def h5get(item, sel,verbose=True, caseSensitive=False, getNames=False, cs=False):
    
    """Function to fetch data within HDF5 file.

    Parameters
    ----------
    item : file
        HDF5 file or group from an HDF5 file
    sel : str
        String or list of strings
        -> all must be present within the target product name
        -> the order is not important
    verbose : bool
        If True, the name and nature of the result are displayed. 
        Nature is on of attribute, dataset or group
    caseSensitive and cs : bool
        By default, the search is not case sensitive, but this can be forced via 
        either 'cs' or 'caseSensitive'
    getNames : bool
        If True, a second output variable is provided, with the names (full paths) 
        to the matching objects

    Returns
    -------
    All datasets and attributes where all strings in 'sel' are present in the
    path will be returned Groups have not direct content, and are hence ignored
    By path, we mean the full path below the input "item" ("item" may not be the
    root of the hdf5 file).

    If one result is found, it is returned as a single variable
    If multiple results are found, they are returned in a list
    
    Examples
    --------
    >>> import h5py
    >>> hfile = h5py.File("myfile.hdf5", 'r')

    Example 1:
    >>> bias = h5get(hfile, ["bias", "27"])  # Method 1
    >>> bias = h5get(hfile, ["27", "Bias"])  # Method 2
    
    Example 2:
    >>> versions        = h5get(hfile,"Version")               # Method 1
    >>> versions, names = h5get(hfile, "version", getNames=1)  # Method 2
    """

    if not isinstance(sel,list): 
        sel = [sel]
    
    # BUILD FULL LIST OF PATHS IN THE INPUT PRODUCT
    
    paths, classes = h5paths(item,verbose=False,includeGroups=False)

    # MAKE SEARCH CASE INSENSITIVE

    if cs or caseSensitive:
        cs = caseSensitive = True
        sel2Match = sel
        paths2Match = paths
    else:
        sel2Match = [s.lower() for s in sel]
        paths2Match = [s.lower() for s in paths]

    # SELECT MATCHING ITEMS wrt all strings in input selection

    matchingItems = fileMatch(paths2Match,sel2Match)
    if verbose:
        print("selection strings: {}".format(sel))
        if len(matchingItems)>0:
            print("matching items:")
        else:
            print("No matching item")

    # RETRIEVAL

    result = []
    resultnames = []
    hclass = {"a":"Attribute","d":"Dataset  "}

    if len(matchingItems)>0:
      paths = np.array(paths)
      paths2Match = np.array(paths2Match)
      matchingItems = np.array(matchingItems)
      
      for objj in matchingItems:
        
        # Matching may have occurred on case-insensitive arrays
        # ==> identify the matched objects in the original input
        
        index = np.where(paths2Match==objj)[0][0]
        obj = paths[index]
        resultnames.append(obj)
        if verbose:
            print("    {0}       {1}".format(hclass[classes[index]],obj))
        
        # RETRIEVAL
        # Retrieval syntax is different for attributes than for datasets
        
        if classes[index] == 'a':
        
            # when group is not at root, the '/' at the start of its name must be removed
        
            group  = os.path.dirname(obj)[1:]
            attr = os.path.basename(obj)
            result.append(item[group].attrs[attr])
        elif classes[index] == 'd':
            result.append(np.array(item[obj]))
        else:
            if verbose:
                print(f"Object {obj}, is of type {obj.__class__} " +
                      "and doesn't directly contain data")
    
    # That's it!

    if getNames:
        if len(matchingItems)==1:
            return result[0],resultnames[0]
        else:
            return result,resultnames
    else:
        if len(matchingItems)==1:
            return result[0]
        else:
            return result




        
def h5ls(item, sel="", s="", indent=" "*4, verbose=True, summary=True,
         fullPath=False, fp=False, keepDeeper=False, baseString=""):

    """Function to display entries within HDF5 file.

    Parameters
    ----------
    item : file
        HDF5 file instance, or group in a HDF5 file
    sel : str
        If specified, only the items matching this string are printed
    s : str
        Base string, prepended to any output. This variable is primarily used 
        in the internal recursion
    indent : str
        Incremental indentation for any sub-level in the structure
    verbose : bool
        If True, the attribue values are displayed on the output as well
    summary : bool
        By default, 3+ consecutive objects starting with the same 2 characters
        will not all be displayed. We only display the first 2 and the last.
        If summary=False forces an exhaustive display of all items
    fullPath or fp [synomymous] : bool
        If at least one is True, the full path is displayed instead of just the item name.
    keepDeeper : bool
        Private variable used in the recursion. Must be False.
    baseString : str
        Private variable used in the recursion when full=True. Must be left empty.

    Returns
    -------
    Recursively prints the structure of an hdf5 file, or a group from an hdf5 file.
    If verbose=True (default), the attribute values are displayed as well
    [G], [D] or [a] is prepended to each line w.r.t. the nature of the item: 
    group, dataset or attribute.
    
    Examples
    --------
    >>> import h5py
    >>> import h5.py
    >>> file = h5py.File("myFile.hdf5", "r")
    >>> h5ls(file)
    >>> h5ls(file["InputParameters/ObservingParameters"])
    >>> h5ls(file["InputParameters/ObservingParameters"],"gain") 
    >>> h5ls(file["InputParameters"],"psf")    
    >>> h5ls(file["InputParameters/PSF")    
    """

    flag = 0
    if keepDeeper: flag = 1
    if sel: sel=sel.lower()
    
    # min number of identical characters to trigger a summarized output
    
    charEqual = 2
    try:
            groupkeys = item.attrs.keys()
            if sel and not keepDeeper:
                groupkeys = [k for k in groupkeys if k.lower().find(sel)>= 0]
            maxlength = max([len(k) for k in groupkeys])
            if verbose:
              for key in groupkeys:
                print("[a] " + s + baseString + '/' +
                      "{0}".format(key).rjust(maxlength), indent, item.attrs[key])
            else:
              for key in groupkeys:
                print(s + baseString+key)
    except:
            pass
    
    # SUMMARIZE THE OUTPUT FOR GROUPS OF GROUPS BEARING SIMILAR NAMES
    # IF AT LEAST 3 CONSECUTIVE GROUP NAMES START BY THE SAME 'charEqual'
    # CHARACTERS, ONLY DISPLAY THE FIRST TWO AND THE LAST
    # ref1 & ref2 keep the first characters of the last 2 items at the
    # current level in the tree
    
    ref2,ref1 = None,None
    
    # keep and shortenHere are flags controlling the ouput in case summary=True
    # To allow anything on the output, keep must be True
    # If instead shortenHere is True, we have just identified the 3rd simimar item in a row
    # Since we want to display "..." just once, shortenHere is reset to False inside the loop
    
    keep = True
    
    itemlist = list(item.items())
    nitems = len(itemlist)
    
    for j,i in enumerate(itemlist):
        shortenHere = False
    
        # Spot the identical groups of names  
    
        if summary:
    
          # Identify groups of at least 3 similar names
    
          if (j > 1) and (j<(nitems-1)):
              ref2,ref1 = itemlist[j-2][0][:charEqual],itemlist[j-1][0][:charEqual]
              if (ref2 == ref1) and (i[0][:charEqual] == ref1):
                  if keep:
                      keep = False
                      shortenHere = True
    
          # Identify the end of a sequence of similar names
          # Make sure the last item is always displayed
    
          if (j==(nitems-1)) or (i[0][:charEqual] != itemlist[j+1][0][:charEqual]):
              keep = True
        
        # This node matches the selection => sub-levels should be kept
        
        if i[0].lower().find(sel)>=0: 
            keepDeeper = True
        elif not flag:
            # This node doesn't match the selection, and superior nodes also not
            # => non-matching sub-levels shouldn't be kept
            keepDeeper = False

        # No selection requested, or matching selection, or matching identified
        # at higher level => display unless this item is similar to the previous
        # ones already displayed in which case we may not display it at all
        # (keep=False) or mark that we are entering "summary mode" (shortenHere=True)
        
        if not sel or (sel and ((i[0].lower().find(sel)>=0) or keepDeeper)):
            if keep: 
                print({h5py._hl.group.Group:"[G] ",
                       h5py._hl.dataset.Dataset:"[D] "}[i[1].__class__] +
                      s + baseString+'/'+i[0])
            elif shortenHere:
                print("    " + s + "...")
    
        # If this node is a group => recurse on lower level
        # (unless it's skipped due to 'summary')
        
        if isinstance(i[1], h5py._hl.group.Group) and keep:
            if fullPath or fp:
                h5ls(i[1], sel=sel, summary=summary, s=s+indent, verbose=verbose,
                     keepDeeper=keepDeeper,fullPath=True,baseString=baseString+'/'+i[0])
            else:
                h5ls(i[1], sel=sel, summary=summary, s=s+indent, verbose=verbose,
                     keepDeeper=keepDeeper,fullPath=False,baseString=baseString)

    # That's it!
    
    return

# -*- coding: utf-8 -*-
"""
SPDX-FileCopyrightText: 2024 German Aerospace Center (DLR)
SPDX-License-Identifier: MIT
"""

"""
Berlin, 03.03.2025

We update the code to deal with the PIC 2.1.0.1 release.

--------------------------------------------------------------------------------
Berlin, 09.08.2024

This is a release for the Github repository as part of the PLATO utilities 
project (https://github.com/PLATO-DLR/plato_utilities).

This script contains convenient routines (some more convinient, some less) to 
interface the PIC and PINE catalogues.

Juan Cabrera Pérez
"""

import numpy as np
from numpy.random import Generator, PCG64

import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import astropy.table as apt



#%% Star class
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
#_______________________________________________________________________________
#
# class Star
#_______________________________________________________________________________
class Star:
    """[Summary]
    
    This class manages the stellar parameters that we read from PIC. Typically,
    we will read a catalog of stars, which means that the attributes of Star 
    will be np.arrays. In total, one expect nelements in Star, while n is the
    internal counter for the iterators (see below).
    
    Attributes:
      id (int)      : star id
      ra (double)   : right ascension
      de (double)   : declination
      mag (double)  : stellar magnitude
      rs (double)   : stellar radius (solar units)
      urs (double)  : uncertainty in the stellar radius (solar units)
      ms (double)   : stellar mass (solar units)
      ums (double)  : uncertainty in the stellar mass (solar units)
      ts (double)   : effective temperature (K)
      uts (double)  : uncertainty in the effective temperature (K)
      plx (double)  : parallax of the star (in arcsec TBC).
      uplx (double) : uncertainty in the parallax of the star (in arcsec TBC).
      nsr (double)  : noise-to-signal value as taken from the PIC (EOL required)
      
      n (int)         : internal counter for iterators (see below).
      nelements (int) : number of elements in the attributes of Star.
      
    Methods:
      get()           : returns the elements of Star as an np.array.
      get_nelements() : returns the number of elements in Star.
    """
    
    #-----
    # Star::__init__()
    def __init__( self, ids  = None, 
                        ra   = None, 
                        de   = None, 
                        mag  = None, 
                        rs   = None, 
                        urs  = None,
                        ms   = None,
                        ums  = None,
                        ts   = None,
                        uts  = None,
                        plx  = None,
                        uplx = None,
                        nsr  = None):
        """[Summary]
        
        Method to initialize the Star class.
        
        Arguments:
          id (int)      : star id
          ra (double)   : right ascension
          de (double)   : declination
          mag (double)  : stellar magnitude
          rs (double)   : stellar radius (solar units)
          urs (double)  : uncertainty in the stellar radius (solar units)
          ms (double)   : stellar mass (solar units)
          ums (double)  : uncertainty in the stellar mass (solar units)
          ts (double)   : effective temperature (K)
          uts (double)  : uncertainty in the effective temperature (K)
          plx (double)  : parallax of the star (in arcsec TBC).
          uplx (double) : uncertainty in the parallax of the star (in arcsec TBC).
          nsr (double)  : noise-to-signal value as taken from the PIC (e.g. EOL required)        
        """
        # we initialize the attributes with the input arrays
        self.ids  = ids
        
        self.ra   = ra
        self.de   = de
        
        self.mag  = mag
        
        self.rs   = rs
        self.urs  = urs
        
        self.ms   = ms
        self.ums  = ums
        
        self.ts   = ts
        self.uts  = uts
        
        self.plx  = plx
        self.uplx = uplx

        self.nsr  = nsr
        
        # as long as we have ids defined, we initialize the number of elements
        # in the Star class (needed by the iterators).
        self.n = 0
        self.nelements = 0
        if ids is not None:
            if np.isscalar( ids):
                self.nelements = 1
            else:
                self.nelements = len( ids)
        
    #-----
    # Star::get
    def get( self):
        """[Summary]
        Returns the contents of the attributes in the Star class as a np array.
        """
        return np.column_stack( ( self.ids,
                                  self.ra,
                                  self.de,
                                  self.mag,
                                  self.rs,
                                  self.urs,
                                  self.ms,
                                  self.ums,
                                  self.ts,
                                  self.uts,
                                  self.plx,
                                  self.uplx,
                                  self.nsr))

    #-----
    # Star::get_nelements 
    def get_nelements( self):
        """[Summmary]
        
        Returns the number of elements in the Star.
        """
        return self.nelements

    #-----
    # Star::__str__() for the prints
    def __str__( self):
        if not isinstance( self.ids, np.ndarray) and self.ids is not None:
            return f'{self.ids:10d} {self.ra:6.2f} {self.de:6.2f} {self.mag:5.2f} {self.rs:4.2f} {self.ms:4.2f} {self.nsr:5.1f}'
        else:   
            return str( self.get())

    #-----
    # Star::__iter__() iterator
    def __iter__( self):
        self.n = 0
        return self

    #-----
    # Star::__next__() iterator 
    def __next__(self):
        if self.n < self.nelements:
            star = self.__getitem__( self.n)
            self.n += 1
            return star
        else:
            raise StopIteration

    #-----
    # Star::__getitem__() iterator 
    def __getitem__( self, key):
        if key < self.nelements:
            if self.ids is None:
                return Star()          
            return Star( ids    = self.ids [ key],
                         ra     = self.ra  [ key],
                         de     = self.de  [ key],
                         mag    = self.mag [ key],
                         rs     = self.rs  [ key],
                         urs    = self.urs [ key],
                         ms     = self.ms  [ key],
                         ums    = self.ums [ key],
                         ts     = self.ts  [ key],
                         uts    = self.uts [ key],
                         plx    = self.plx [ key],
                         uplx   = self.uplx[ key],
                         nsr    = self.nsr [ key])
        else:
            return None



#%% Catalog class
#_______________________________________________________________________________
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
#_______________________________________________________________________________
#
# class Catalog
#_______________________________________________________________________________
class Catalog:
    """[Summary]
    
    This is a generic (base) class to manage stellar catalogs like PIC or PINE.
    
    Attributes:
      column_input_id (int) : the column containing the stellar id in the 
        catalog.

      column_output_id (int) : the column that will contain the stellar id in 
        the datos attribute of the class (necessary, for example, to call the
        get method).
        
      fname (string)   : the name of input file with the catalog data.
      datos (np.array) : the variable with the data extracted from the catalog.
      
    Methods:
      readfile() : to read the inputs of the catalog from the input file.
      get_id()   : returns the stellar id as read from the catalog.
    """
    column_input_id  = 0
    column_output_id = 0

    #-----
    # Catalog::__init__()
    def __init__( self, fname=None):
        """[Summary]
        
        Method initializing the Catalog class.
        
        Arguments:
          fname (string) : the name of the input file with the catalog data.
        """
        if fname is None:
            return
        else:
            self.fname = fname
            self.datos = None

        self.readfile()

    #-----
    # Catalog::read file()
    def readfile( self):
        pass
    
    #-----
    # Catalog::get() returns a np.array with the values in the given column in 
    # the datos attribute.
    def get( self, column=None, indices=None):
        """[Summary]
        
        Method to extract a slice of the datos attribute defined by the
        colum (column) and range of index (indices).
        
        Arguments:
          column : the column from datos to be extracted.
          indices : the range of indices in the column to be extracted.
        """
        if indices is None:
            return self.datos[ :, column]
        else:
            return self.datos[ indices, column]
        
    #-----
    # Catalog::get_id() returns the stellar id array
    def get_id( self, indices=None):
        """[Summary]
        
        Method to extract the id values from datos.
        
        Arguments:
          indices : the range of indices in the column to be extracted.
        """
        return self.get( self.column_output_id, indices).astype( 'int')



#%% StarPIC is a class derived from Catalog to include the stellar parameters
# (and only the stellar parameters!) of the PIC stars.
#_______________________________________________________________________________
#
# class StarPIC 
#_______________________________________________________________________________
class StarPIC( Catalog):
    """[Summary]
    
    Generic class to manage PIC stellar catalogs, which are those including 
    only stellar parameters, not the NSR values.
    
    Actually, to do things properly, we should define the base methods and
    attributes here and only update as needed in the derived classes, which
    is not what is happening now.
    """


#_______________________________________________________________________________
#
# class StarPIC 1.1.0
#_______________________________________________________________________________
class StarPIC110( StarPIC):
    """[Summary]
    
    This is the class that manages the (stellar) PIC input catalog version 
    1.1.0 issued in December 2019). This is a derived class from StarPIC, which 
    in turn it is derived from Catalog, and inherits their attributes and 
    methods.
    
    Attributes:
      column_input_id (int)    : the column containing the stellar id in the 
                                 catalog.
      column_input_ra (int)    : the column containing the stellar right 
                                 ascension in the catalog.
      column_input_de (int)    : the column containing the stellar declination 
                                 in the catalog.
      column_input_gmag (int)  : the column containing the Gaia magnitude in 
                                 the catalog.
      column_input_vmag (int)  : the column containing the Johnsons V magnitude 
                                 in the catalog.
      column_input_teff (int)  : the column containing the stellar effective 
                                 temperature in the catalog.
      column_input_uteff (int) : the column containing the uncertainty in the 
                                 stellar effective temperature in the catalog.
      column_input_rs (int)    : the column containing the stellar radius in 
                                 the catalog.
      column_input_urs (int)   : the column containing the uncertainty in the 
                                 stellar radius in the catalog.
      column_input_ms (int)    : the column containing the stellar mass in the 
                                 catalog.
      column_input_ums (int)   : the column containing the uncertainty in the 
                                 stellar mass in the catalog.
      column_input_plx (int)   : the column containing the stellar parallax in 
                                 the catalog in mas.
      column_input_uplx (int)  : the column containing the uncertainty in the 
                                 stellar parallax in the catalog in mas.

      column_output_id (int)   : the column that will contain the stellar id in 
                                 the datos attribute of the class (necessary, 
                                 for example, to call the get() method).
      column_output_ra (int)   : the column that will contain the stellar right
                                 ascension in the datos attribute of the class.
      column_output_de (int)   : the column that will contain the stellar 
                                 declination in the datos attribute of the 
                                 class.
      column_output_gmag (int) : the column that will contain the Gaia 
                                 magnitude in the datos attribute of the class.
      column_output_vmag (int) : the column that will contain the Johnsons V 
                                 magnitude in the datos attribute of the class.
      column_output_ts (int)   : the column that will contain the stellar 
                                 effective temperature in the datos attribute 
                                 of the class.
      column_output_uts (int)  : the column that will contain the uncertainty 
                                 in the stellar effective temperature in the 
                                 datos attribute of the class.
      column_output_rs (int)   : the column that will contain the stellar 
                                 radius in the datos attribute of the class.
      column_output_urs (int)  : the column that will contain the uncertainty 
                                 in the stellar radius in the datos attribute 
                                 of the class.
      column_output_ms (int)   : the column that will contain the stellar 
                                 mass in the datos attribute of the class.
      column_output_ums (int)  : the column that will contain the uncertainty 
                                 in the stellar mass in the datos attribute of 
                                 the class.
      column_output_plx (int)  : the column that will contain the stellar 
                                 parallax in the datos attribute of the class 
                                 in mas.
      column_output_uplx (int) : the column that will contain the uncertainty 
                                 in the stellar parallax in the datos attribute 
                                 of the class in mas.
        
      fname (string)   : the name of input file with the catalog data.
      datos (np.array) : the variable with the data extracted from the catalog.
      
    Methods:
      readfile() : to read the inputs of the catalog from the input file.

      get_id()   : returns the stellar id as read from the catalog.
      get_ra()   : returns the stellar right ascension as read from the catalog.
      get_de()   : returns the declination array as read from the catalog.
      get_gmag() : returns the Gaia magnitude array as read from the catalog.
      get_vmag() : returns the Johnsons V mag array as read from the catalog.
      get_ts()   : returns the stellar effective temperature array as read from the catalog.
      get_uts()  : returns the uncertainty of the stellar effective temperature as read from the catalog.
      get_rs()   : returns the stellar radius array as read from the catalog.
      get_urs()  : returns the uncertainty of the stellar radius array as read from the catalog.
      get_ms()   : returns the stellar mass array as read from the catalog.
      get_ums()  : returns the uncertainty of the stellar mass array as read from the catalog.
      get_plx()  : returns the stellar parallax array as read from the catalog.
      get_uplx() : returns the uncertainty of the stellar parallax array as read from the catalog.

      conversion() : method to parse empty values in the StarPIC catalog.
    """

    column_input_ra    =  2
    column_input_de    =  4
    column_input_gmag  = 17
    column_input_vmag  = 49
    column_input_teff  = 51
    column_input_uteff = 52
    column_input_rs    = 53
    column_input_urs   = 54
    column_input_ms    = 55
    column_input_ums   = 56
    column_input_plx   =  6
    column_input_uplx  =  7

    column_output_ra    =  1
    column_output_de    =  2
    column_output_gmag  =  3
    column_output_vmag  =  4
    column_output_ts    =  5
    column_output_uts   =  6
    column_output_rs    =  7
    column_output_urs   =  8
    column_output_ms    =  9
    column_output_ums   = 10
    column_output_plx   = 11
    column_output_uplx  = 12

    #-----
    # StarPIC110::readfile() reads the catalog information from the catalog into a
    # np array.
    def readfile( self):
        self.datos = np.loadtxt( self.fname, 
                                 skiprows = 1, delimiter=',', 
                                 usecols = ( self.column_input_id, 
                                             self.column_input_ra, 
                                             self.column_input_de, 
                                             self.column_input_gmag, 
                                             self.column_input_vmag, 
                                             self.column_input_teff, 
                                             self.column_input_uteff, 
                                             self.column_input_rs, 
                                             self.column_input_urs, 
                                             self.column_input_ms, 
                                             self.column_input_ums,
                                             self.column_input_plx,
                                             self.column_input_uplx),
                                converters = { self.column_input_teff  : self.conversion, 
                                               self.column_input_uteff : self.conversion, 
                                               self.column_input_rs    : self.conversion, 
                                               self.column_input_urs   : self.conversion, 
                                               self.column_input_ms    : self.conversion, 
                                               self.column_input_ums   : self.conversion,
                                               self.column_input_plx   : self.conversion,
                                               self.column_input_uplx  : self.conversion,
                                             } )
    
        # because some of the uncertainties of the mass are non zero
        # while the mass value is zero, we have to correct them
        # awk 'BEGIN{FS=",";} $56 ~ /N/ { print $56, $57}' ../PIC1.1.0/asPIC110_performance.csv
        mask = ( self.get_ms() == 0)
        self.datos[ mask, self.column_output_ums] = 0.
        
    #-----
    # StarPIC110::get_ra() returns the right ascension array
    def get_ra( self, indices = None):
        return self.get( self.column_output_ra, indices)

    #-----
    # StarPIC110::get_de() returns the declination array
    def get_de( self, indices = None):
        return self.get( self.column_output_de, indices)

    #-----
    # StarPIC110::get_gmag() returns the Gaia magnitude array
    def get_gmag( self, indices = None):
        return self.get( self.column_output_gmag, indices)

    #-----
    # StarPIC110::get_vmag() returns the Johnsons V mag array
    def get_vmag( self, indices = None):
        return self.get( self.column_output_vmag, indices)

    #-----
    # StarPIC110::get_ts() returns the stellar effective temperature array
    def get_ts( self, indices = None):
        return self.get( self.column_output_ts, indices)

    #-----
    # StarPIC110::get_uts() returns the uncertainty of the stellar effective temperature array
    def get_uts( self, indices = None):
        return self.get( self.column_output_uts, indices)

    #-----
    # StarPIC110::get_rs() returns the stellar radius array
    def get_rs( self, indices = None):
        return self.get( self.column_output_rs, indices)

    #-----
    # StarPIC110::get_urs() returns the uncertainty of the stellar radius array
    def get_urs( self, indices = None):
        return self.get( self.column_output_urs, indices)

    #-----
    # StarPIC110::get_ms() returns the stellar mass array
    def get_ms( self, indices = None):
        return self.get( self.column_output_ms, indices)

    #-----
    # StarPIC110::get_ums() returns the uncertainty of the stellar mass array
    def get_ums( self, indices = None):
        return self.get( self.column_output_ums, indices)
    
    #-----
    # StarPIC110::get_plx() returns the stellar parallax array
    def get_plx( self, indices = None):
        return self.get( self.column_output_plx, indices)

    #-----
    # StarPIC110::get_uplx() returns the uncertainty of the stellar parallax array
    def get_uplx( self, indices = None):
        return self.get( self.column_output_uplx, indices)

    #-----
    # StarPIC110::conversion() method to parse empty values in the StarPIC catalog.
    # all \N are in columns 51 to 56, 60 to 62, and 67 and 68 (starting in 0)
    def conversion( self, fld):
        if( fld.find( b'N')) == 1:
            return 0.
        else:
            return float( fld)


"""
array([ 3.00000000e+00,  1.09836513e+02, -8.99472372e+01,  1.26231000e+01,
        1.28437550e+01,  5.48765716e+03,  1.97853624e+02,  1.66858600e+00,
        1.40025000e-01,  1.15879200e+00,  1.02069000e-01])
3,5188147152985808768,109.83651272443973,0.022126794650239128,-89.94723723655241,0.027992835674762167,1.91892667085185,0.028763623134024682,3.8270284032768687,0.04463915117590438,8.236279709961202,0.055820132337616256,9.081984907508186,0.05720079606260468,2015.5,168311.83573004865,29.93168011579497,
12.6231,0.000193064,82848.90361195648,94.15344076921012,13.0557,0.00123318,123124.31924237212,85.56333656860168,12.0361,0.000754253,1.22376,1.01961,0.432592,0.587019,302.87307758747386,-27.121828567896348,269.95489874687814,-66.61034476532608,513.513174715534,505.981046383827,521.268786435075,7.643870025623983,0.257474,0.082025,0.121188,0.039055,0.098189,0.029031,0,0.8984233586425782,12.539369106161796,3.9866111543934792,
12.843755006161796,3.8128473481584404,5487.657159,197.853624,1.668586,0.140025,1.158792,0.102069,1,0.0410424352728948,0.15664526899460773,\\N,\\N,\\N,15.899724011449221,6,16.82851601758153,3,20.548170089721683,1
"""

# 0 PICidDR1,
# 1 sourceId,
# 2 ra,
# 3 raError,
# 4 decl,
# 5 decError,
# 6 parallax,
# 7 parallaxError,
# 8 pmra,
# 9 pmraError,
#10 pmdec,
#11 pmdecError,
#12 pmtotal,
#13 pmtotalError,
#14 refEpoch,
#15 photGMeanFlux,
#16 photGMeanFluxError,
#17 photGMeanMag,
#18 photGMeanMagError,
#19 photBpMeanFlux,
#20 photBpMeanFluxError,
#21 photBpMeanMag,
#22 photBpMeanMagError,
#23 photRpMeanFlux,
#24 photRpMeanFluxError,
#25 photRpMeanMag,
#26 photRpMeanMagError,
#27 photBpRpExcessFactor,
#28 bpRp,
#29 bpG,
#30 Grp,
#31 l,
#32 b,
#33 eclLon,
#34 eclLat,
#35 rest,
#36 rlo,
#37 rhi,
#38 restError,
#39 ag,
#40 agError,
#41 ebprp,
#42 ebprpError,
#43 ebv,
#44 ebvError,
#45 extStatus,
#46 bpRp0,
#47 gaiaV0,
#48 BJgaiaMV0,
#49 gaiaV,
#50 BJgaiaMG0,
#51 teff,
#52 teffError,
#53 radius,
#54 radiusError,
#55 mass,
#56 massError,
#57 dwsgFlag,
#58 bpRp0Error,
#59 BJgaiaMG0Error,
#60 gaiaV0Error,
#61 BJgaiaMV0Error,
#62 gaiaVError,
#63 contGaiaMag60,
#64 contNumber60,
#65 contGaiaMag45,
#66 contNumber45,
#67 contGaiaMag30,
#68 contNumber30








#%% PINE class derived from Catalog
#_______________________________________________________________________________
#
# class PINE
#_______________________________________________________________________________
class PINE( Catalog):
    """[Summary]
    
    Generic class to manage PINE catalogs
    """

    

#_______________________________________________________________________________
#
# class PINE 2019
#_______________________________________________________________________________
class PINE2019( PINE):
    """[Summary]
    
    This is the class that manages the PINE input catalog as it was defined in
    2019. This is a derived class from Catalog that inherits its attributes and 
    methods.
    
    Attributes:
      column_input_id (int)     : the column containing the stellar id in the 
                                  catalog.
      column_input_vmag (int)   : the column containing the Johnsons V 
                                  magnitude in the catalog.
      column_input_random (int) : the column containing the random component of 
                                  the noise-to-signal ratio in the catalog.
      column_input_total (int)  : the column containing the total 
                                  noise-to-signal ratio in the catalog (random
                                  plus systematics).
      column_input_ncams (int)  : the column containing the number of 
                                  cameras seeing the star.

      column_output_id (int)     : the column that will contain the stellar id 
                                   in the datos attribute of the class 
                                   (necessary, for example, to call the get() 
                                   method).
      column_output_vmag (int)   : the column that will contain the Johnsons V 
                                   magnitude in the datos attribute of the 
                                   class.
      column_output_random (int) : the column that will contain the random 
                                   component of the noise-to-signal ration in 
                                   the datos attribute of the class.
      column_output_total (int)  : the column that will contain the total 
                                   noise-to-signal ration in the datos 
                                   attribute of the class.
      column_output_ncams (int)  : the column cthat will contain the number of 
                                   cameras seeing the star.
        
      fname (string)   : the name of input file with the catalog data.
      datos (np.array) : the variable with the data extracted from the catalog.
      
    Methods:
      readfile() : to read the inputs of the catalog from the input file.

      get_id()     : returns the stellar id as read from the catalog.
      get_vmag()   : returns the Johnsons V mag array as read from the catalog.
      get_random() : returns the random component of the noise-to-signal ratio as read from the catalog.
      get_total()  : returns the noise-to-signal ratio as read from the catalog.
      get_ncams()  : returns the number of cameras seeing the star.
    """

    column_input_vmag      =  1
    column_input_longitude =  2
    column_input_latitude  =  3 
    column_input_random    =  4
    column_input_total     =  5
    column_input_ncams     =  6

    column_output_vmag      =  1
    column_output_longitude =  2
    column_output_latitude  =  3 
    column_output_random    =  4
    column_output_total     =  5
    column_output_ncams     =  6

    #-----
    # PINE::readfile() reads the catalog information from the catalog into a
    # np array.
    def readfile( self):
        self.datos = np.loadtxt( self.fname, 
                                 usecols = ( self.column_input_id,
                                             self.column_input_vmag, 
                                             self.column_input_longitude,
                                             self.column_input_latitude,
                                             self.column_input_random, 
                                             self.column_input_total,
                                             self.column_input_ncams) )

    #-----
    # PINE::get_vmag() returns the Johnsons V mag array
    def get_vmag( self, indices = None):
        return self.get( self.column_output_vmag, indices)

    #-----
    # PINE::get_longitude() returns the Longitude in S/C coordinate system [deg]
    def get_longitude( self, indices = None):
        return self.get( self.column_output_longitude, indices)

    #-----
    # PINE::get_latitude() returns the Latitude in S/C coordinate system [deg]
    def get_latitude( self, indices = None):
        return self.get( self.column_output_latitude, indices)

    #-----
    # PINE::get_random() returns the random component of the noise-to-signal ratio array
    def get_random( self, indices = None):
        return self.get( self.column_output_random, indices)

    #-----
    # PINE::get_random() returns the noise-to-signal ratio array
    def get_total( self, indices = None):
        return self.get( self.column_output_total, indices)

    #-----
    # PINE::get_ncams() returns the number of cameras seeing the star
    def get_ncams( self, indices = None):
        return self.get( self.column_output_ncams, indices)


#  1. Column = star ID
#  2. Column = Magnitude [-]
#  3. Column = Longitude in S/C coordinate system [deg]
#  4. Column = Latitude in S/C coordinate system [deg]
#  5. Column = random NSR [ppm], system level in 1hr
#  6. Column = random + systematic NSR [ppm], system level in 1hr
#  7. Column = number of cameras seeing the star
#  8. Column = number of cameras getting saturated
# -------------------------------------------------------------------------------
# Simulation coverage [%]:       100.000
# -------------------------------------------------------------------------------					
#    36861223   12.63  -14.10   24.04 3.567e+002 3.568e+002      5      0


#_______________________________________________________________________________
#
# class PINE 2024
#_______________________________________________________________________________
class PINE2024( PINE2019):
    """[Summary]
    
    This is the class that manages the PINE input catalog as it was defined in
    2024. The only change in the output wrt 2019 is that we use ',' as 
    separator instead of white characters. This is a derived class from Catalog 
    that inherits its attributes and methods.
    
    Note that the 'vmag' column is computed by PINE from the PLATO magnitude,
    so it might not be 1:1 with the value computed by INAF/MPSSR.
    
    Attributes:
      column_input_id (int)     : the column containing the stellar id in the 
                                  catalog.
      column_input_mag (int)   : the column containing the magnitude in the 
                                 catalog.
      column_input_random (int) : the column containing the random component of 
                                  the noise-to-signal ratio in the catalog.
      column_input_total (int)  : the column containing the total 
                                  noise-to-signal ratio in the catalog (random
                                  plus systematics).
      column_input_ncams (int)  : the column containing the number of 
                                  cameras seeing the star.
      column_input_nsat (int)   : the column containing the number of 
                                  saturated cameras seeing the star.

      column_output_id (int)     : the column that will contain the stellar id 
                                   in the datos attribute of the class 
                                   (necessary, for example, to call the get() 
                                   method).
      column_output_mag (int)    : the column that will contain the magnitude 
                                   in the datos attribute of the class.
      column_output_random (int) : the column that will contain the random 
                                   component of the noise-to-signal ration in 
                                   the datos attribute of the class.
      column_output_total (int)  : the column that will contain the total 
                                   noise-to-signal ration in the datos 
                                   attribute of the class.
      column_output_ncams (int)  : the column that will contain the number of 
                                   cameras seeing the star.
      column_output_nsat (int)   : the column that will contain the number of 
                                   saturated cameras seeing the star.
        
      fname (string)   : the name of input file with the catalog data.
      datos (np.array) : the variable with the data extracted from the catalog.
      
    Methods:
      readfile() : to read the inputs of the catalog from the input file.

      get_id()     : returns the stellar id as read from the catalog.
      get_mag()    : returns the magnitude in the catalogue.
      get_random() : returns the random component of the noise-to-signal ratio as read from the catalog.
      get_total()  : returns the noise-to-signal ratio as read from the catalog.
      get_ncams()  : returns the number of cameras seeing the star.
      get_nsat()   : returns the number of saturated cameras seeing the star.
    """
    column_input_mag   =  1
    column_input_nsat  =  7
    
    column_output_mag  =  1
    column_output_nsat =  7

    #-----
    # PINE2024::readfile() reads the catalog information from the catalog into a
    # np array.
    def readfile( self):
        self.datos = np.loadtxt( self.fname, 
                                 delimiter = ',',
                                 usecols = ( self.column_input_id,
                                             self.column_input_vmag, 
                                             self.column_input_longitude,
                                             self.column_input_latitude,
                                             self.column_input_random, 
                                             self.column_input_total,
                                             self.column_input_ncams,
                                             self.column_input_nsat       ) )

    #-----
    # PINE2024::get_id() returns the stellar id array
    def get_id( self, indices=None):
        """[Summary]
        
        Method to extract the id values from datos.
        
        Arguments:
          indices : the range of indices in the column to be extracted.
        """
        return self.get( self.column_output_id, indices).astype( 'int64')

    #-----
    # PINE2024::get_mag() returns the magnitude in the catalogue
    def get_mag( self, indices = None):
        return self.get( self.column_output_mag, indices)

    #-----
    # PINE2024::get_nsat() returns the number of saturated cameras observing the target
    def get_nsat( self, indices = None):
        return self.get( self.column_output_nsat, indices)

#  1. Column = star ID
#  2. Column = Magnitude [-]
#  3. Column = Longitude in S/C coordinate system [deg]
#  4. Column = Latitude in S/C coordinate system [deg]
#  5. Column = random NSR [ppm], system level in 1hr
#  6. Column = random + systematic NSR [ppm], system level in 1hr
#  7. Column = number of cameras seeing the star
#  8. Column = number of cameras getting saturated
# -------------------------------------------------------------------------------
# Simulation coverage [%]:       100.000
# -------------------------------------------------------------------------------					
#      153097,    9.60,  -12.48,  -26.96, 0.000e+000, 0.000e+000,      0,      0



#%% PLATO Input Catalog (PIC) classes
#_______________________________________________________________________________
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
#_______________________________________________________________________________
#
# class PIC
#_______________________________________________________________________________
class PIC:
    """[Summary]
        
    This is a base class for PIC catalogs, including stellar information and
    NSR from PPT, sometimes called 'enhanced' PIC.

    Prior to 2023, these catalogues where generated merging a stellar PIC 
    catalog (StarPIC) containing only the stellar information, and a PINE
    catalog containing the NSR. Nowwadays we are going to try to extract
    all informatio from the PIC catalogues.

    However, as we expect PIC to evolve between now and 2026 (scheduled PLATO
    launch)
    """
    
    def __init__( self):
        pass


#_______________________________________________________________________________
#
# class PICMerge
#_______________________________________________________________________________
class PICMerge( PIC):
    """[Summary]
    
    This class manages the merge of two catalogs: a StarPIC with stellar 
    information and a PINE with NSR information, becoming an 'enhanced' PIC. 
    Basically, this class is a parser that combines the information from both 
    catalogs. There are probably better ways of doing this.

    Attributes:
      seed (int) : the seed for the random number generator.
      rng (numpy.random) : internal random number generator.

      pic (StarPIC)           : the PIC class.
      pine (PINE)             : the PINE class.
      indices_pic (np.array)  : the indices of the StarPIC elements that 
                                cross-match with the PINE targets.
      indices_pine (np.array) : the indices of the PINE elements that 
                                cross-match with the StarPIC targets.
        
      valid ()        :
      simulated_rs () :
      simulated_ms () :

    Methods:
      cross_match() : method that cross-matches the StarPIC and PINE catalogs and 
        initializes the attributes indices_pic and indices_pine, which are 
        np.arrays (resulting from np.intersect1d) with the indices of the 
        elements that have the same stellar id in both catalogs.
      calculate_valid() : method that initializes the attribute valid with a 
        mask that identifies the StarPIC elements having positive (non zero) values 
        of the stellar mass, radius, and effective temperature.
      simulate_stellar_parameters() : the StarPIC catalog contains a radius and a 
        mass estimate, including uncertainties, for each star. What we do here 
        for each star is to simulate a value for the stellar parameters drawn 
        from the expected value and uncertainty stated in the StarPIC. I am 
        actually not sure that this procedure is meaningful.

      get_nelements() : returns the number of elements in the merged catalog.
      get_valid_nelements() : returns the number of elements in the merged.

      get merge() : proxy for get_stars().get().
      get_stars() : builds a Star class with all valid entries in the catalog.

      get_id()           : returns the stellar id of the matches.
      get_ra()           : returns right ascension of the matches.
      get_de()           : returns the declination of the matches.
      get_vmag()         : returns the V mag of the matches.
      get_gmag()         : returns the G mag of the matches.
      get_vmagpine()     : returns the V mag from PINE of the matches.
      get_rs()           : returns the stellar radii of the matches.
      get_urs()          : returns the stellar radius uncertainties of the matches.
      get_ms()           : returns the stellar masses of the matches.
      get_ums()          : returns the stellar mass uncertainties of the matches.
      get_ts()           : returns the stellar effective temperature of the matches.
      get_uts()          : returns the stellar effective temperature uncertainties of the matches.
      get_plx()          : returns the stellar parallax of the matches.
      get_uplx()         : returns the stellar effective temperature uncertainties of the matches.
      get_random()       : returns the random noise value of the matches.
      get_total()        : returns the total noise value of the matches.
      get_simulated_rs() : returns the simulated stellar radii.
      get_simulated_ms() : returns the simulated stellar masses.

      get_valid_id()           : returns the stellar id of the valid matches.
      get_valid_ra()           : returns right ascension of the valid matches.
      get_valid_de()           : returns the declination of the valid matches.
      get_valid_vmag()         : returns the V mag of the valid matches.
      get_valid_gmag()         : returns the G mag of the valid matches.
      get_valid_vmagpine()     : returns the V mag from PINE of the valid matches.
      get_valid_rs()           : returns the stellar radii of the valid matches.
      get_valid_urs()          : returns the stellar radius uncertainties of the valid matches.
      get_valid_ms()           : returns the stellar masses of the valid matches.
      get_valid_ums()          : returns the stellar mass uncertainties of the valid matches.
      get_valid_ts()           : returns the stellar effective temperature of the valid matches.
      get_valid_uts()          : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_plx()          : returns the stellar parallax of the valid matches.
      get_valid_uplx()         : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_random()       : returns the random noise value of the valid matches.
      get_valid_total()        : returns the total noise value of the valid matches.
      get_valid_simulated_rs() : returns the simulated stellar radii for the valid matches.
      get_valid_simulated_ms() : returns the simulated stellar masses for the valid matches.
    """
    
    seed = 12345
    rng  = Generator( PCG64( seed))

    #-----
    # PICMerge::__init__()
    def __init__( self, pic = StarPIC(), pine = PINE()):
        """[Summary]
        
        Method initializing the Catalog class.
        
        Arguments:
          pic (StarPIC)   : the StarPIC class.
          pine (PINE) : the PINE class.
          
          indices_pic (np.array)  : the indices of the StarPIC elements that 
                                    cross-match with the PINE targets.
          indices_pine (np.array) : the indices of the PINE elements that 
                                    cross-match with the StarPIC targets.
        
          valid (mask)            : the mask with the valid values, which are 
                                    defined by the method calculate_valid(). 
                                    Currently we require positive (non zero) 
                                    values of mass, radius, and effective 
                                    temperature.
          simulated_rs (np.array) : the StarPIC catalog contains a radius estimate, 
                                    including uncertainties, for each star. 
                                    With the method 
                                    simulate_stellar_parameters() we simulate 
                                    for each star a value for the stellar 
                                    parameters (radius and mass) drawn from the 
                                    expected value and uncertainty stated in 
                                    the StarPIC. This array keeps the results for 
                                    the stellar radii. 
          simulated_ms (np.array) : same as simulated_rs but for the stellar 
                                    mass.
        """
        self.pic  = pic
        self.pine = pine

        self.indices_pic  = None
        self.indices_pine = None
        self.cross_match()
            
        self.valid = None
        self.calculate_valid()
            
        self.simulated_rs = None
        self.simulated_ms = None
        self.simulate_stellar_parameters()
    
    #-----
    # PICMerge::cross_match() cross-match of StarPIC and PINE catalogs
    def cross_match( self):
        """[Summary]
        
        Method that cross-matches the StarPIC and PINE catalogs and initializes 
        the attributes indices_pic and indices_pine, which are np.arrays 
        (resulting from np.intersect1d) with the indices of the elements that 
        have the same stellar id in both catalogs.        
        """
        intersect, self.indices_pic, self.indices_pine = np.intersect1d( self.pic.get_id(), self.pine.get_id(), assume_unique = True, return_indices = True)

    #-----
    # PICMerge::calculate_valid valid entries are those with positive R, M, T
    def calculate_valid( self):
        """[Summary]
        
        Method that initializes the attribute valid with a mask that identifies 
        the PIC elements having positive (non zero) values of the stellar mass,
        radius, and effective temperature.
        """
        self.valid = ( self.get_rs() > 0) * ( self.get_ms() > 0) * ( self.get_ts() > 0)

    #-----
    # PICMerge::simulate_stellar_parameters() calculates estimated stellar parameters from a distribution
    def simulate_stellar_parameters( self):
        """[Summary]
        
        The PIC catalog contains a radius and a  mass estimate, including 
        uncertainties, for each star. What we do here for each star is to 
        simulate a value for the stellar parameters drawn from the expected 
        value and uncertainty stated in the PIC. I am actually not sure that 
        this procedure is meaningful.
        """
        self.simulated_rs = -1 * np.ones( self.get_rs().shape)
        while np.any( self.simulated_rs < 0):
            mask = self.simulated_rs < 0
            self.simulated_rs[ mask] = self.rng.normal( self.get_rs()[ mask], self.get_urs()[ mask])
        self.simulated_ms   = -1 * np.ones( self.get_ms().shape)
        while np.any( self.simulated_ms < 0):
            mask = self.simulated_ms < 0
            self.simulated_ms[ mask] = self.rng.normal( self.get_ms()[ mask], self.get_ums()[ mask])

    #-----
    # PICMerge::get_nelements() returns the number of elements in the merged catalog
    def get_nelements( self):
        return len( self.indices_pic)

    #-----
    # PICMerge::get_valid_nelements() returns the number of elements in the merged 
    # catalog (using the self.valid mask)
    def get_valid_nelements( self):
        return len( np.where( self.valid)[0])

    #-----
    # PICMerge::get merge() proxy for get_stars().get()
    def get_merge( self):
        return self.get_stars().get()
    
    #-----
    # PICMerge::get_stars() builds a Star class with all valid entries in the
    # catalog.
    def get_stars( self):
        stars = Star( ids  = self.get_valid_id(),
                      ra   = self.get_valid_ra(),
                      de   = self.get_valid_de(),
                      mag  = self.get_valid_vmag(),
                      rs   = self.get_valid_simulated_rs(),
                      urs  = self.get_valid_urs(),
                      ms   = self.get_valid_simulated_ms(),
                      ums  = self.get_valid_ums(),
                      ts   = self.get_valid_ts(),
                      uts  = self.get_valid_uts(),
                      plx  = self.get_valid_plx(),
                      uplx = self.get_valid_uplx(),
                      nsr  = self.get_valid_total())
        return stars
        
    #-----
    # PICMerge::get_id() returns the stellar id of the matches.
    def get_id( self):
        return self.pine.get_id( self.indices_pine)

    #-----
    # PICMerge::get_ra() returns right ascension of the matches.
    def get_ra( self):
        return self.pic.get_ra( self.indices_pic)

    #-----
    # PICMerge::get_de() returns the declination of the matches.
    def get_de( self):
        return self.pic.get_de( self.indices_pic)

    #-----
    # PICMerge::get_vmag() returns the V mag of the matches.
    def get_vmag( self):
        return self.pic.get_vmag( self.indices_pic)

    #-----
    # PICMerge::get_gmag() returns the G mag of the matches.
    def get_gmag( self):
        return self.pic.get_gmag( self.indices_pic)

    #-----
    # PICMerge::get_vmagpine() returns the V mag from PINE of the matches.
    def get_vmagpine( self):
        return self.pine.get_vmag( self.indices_pine)
    
    #-----
    # PICMerge::get_rs() returns the stellar radii of the matches.
    def get_rs( self):
        return self.pic.get_rs( self.indices_pic)

    #-----
    # PICMerge::get_urs() returns the stellar radius uncertainties of the matches.
    def get_urs( self):
        return self.pic.get_urs( self.indices_pic)

    #-----
    # PICMerge::get_ms() returns the stellar masses of the matches.
    def get_ms( self):
        return self.pic.get_ms( self.indices_pic)

    #-----
    # PICMerge::get_ums() returns the stellar mass uncertainties of the matches.
    def get_ums( self):
        return self.pic.get_ums( self.indices_pic)

    #-----
    # PICMerge::get_ts() returns the stellar effective temperature of the matches.
    def get_ts( self):
        return self.pic.get_ts( self.indices_pic)

    #-----
    # PICMerge::get_uts() returns the stellar effective temperature uncertainties of the matches.
    def get_uts( self):
        return self.pic.get_uts( self.indices_pic)

    #-----
    # PICMerge::get_plx() returns the stellar parallax of the matches.
    def get_plx( self):
        return self.pic.get_plx( self.indices_pic)

    #-----
    # PICMerge::get_uplx() returns the stellar effective temperature uncertainties of the matches.
    def get_uplx( self):
        return self.pic.get_uplx( self.indices_pic)

    #-----
    # PICMerge::get_random() returns the random noise value of the matches.
    def get_random( self):
        return self.pine.get_random( self.indices_pine)

    #-----
    # PICMerge::get_total() returns the total noise value of the matches.
    def get_total( self):
        return self.pine.get_total( self.indices_pine)

    #-----
    # PICMerge::get_simulated_rs() returns the simulated stellar radii.
    def get_simulated_rs( self):
        return self.simulated_rs

    #-----
    # PICMerge::get_simulated_ms() returns the simulated stellar masses.
    def get_simulated_ms( self):
        return self.simulated_ms

    #-----
    # PICMerge::get_valid_id() returns the stellar id of the valid matches.
    def get_valid_id( self):
        return self.pine.get_id( self.indices_pine)[ self.valid]

    #-----
    # PICMerge::get_valid_ra() returns right ascension of the valid matches.
    def get_valid_ra( self):
        return self.pic.get_ra( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_de() returns the declination of the valid matches.
    def get_valid_de( self):
        return self.pic.get_de( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_vmag() returns the V mag of the valid matches.
    def get_valid_vmag( self):
        return self.pic.get_vmag( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_gmag() returns the G mag of the valid matches.
    def get_valid_gmag( self):
        return self.pic.get_gmag( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_vmagpine() returns the V mag from PINE of the valid matches.
    def get_valid_vmagpine( self):
        return self.pine.get_vmag( self.indices_pine)[ self.valid]
    
    #-----
    # PICMerge::get_valid_rs() returns the stellar radii of the valid matches.
    def get_valid_rs( self):
        return self.pic.get_rs( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_urs() returns the stellar radius uncertainties of the valid matches.
    def get_valid_urs( self):
        return self.pic.get_urs( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_ms() returns the stellar masses of the valid matches.
    def get_valid_ms( self):
        return self.pic.get_ms( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_ums() returns the stellar mass uncertainties of the valid matches.
    def get_valid_ums( self):
        return self.pic.get_ums( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_ts() returns the stellar effective temperature of the valid matches.
    def get_valid_ts( self):
        return self.pic.get_ts( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_uts() returns the stellar effective temperature uncertainties of the valid matches.
    def get_valid_uts( self):
        return self.pic.get_uts( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_plx() returns the stellar parallax of the valid matches.
    def get_valid_plx( self):
        return self.pic.get_plx( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_uplx() returns the stellar parallax uncertainties of the valid matches.
    def get_valid_uplx( self):
        return self.pic.get_uplx( self.indices_pic)[ self.valid]

    #-----
    # PICMerge::get_valid_random() returns the random noise value of the valid matches.
    def get_valid_random( self):
        return self.pine.get_random( self.indices_pine)[ self.valid]

    #-----
    # PICMerge::get_valid_total() returns the total noise value of the valid matches.
    def get_valid_total( self):
        return self.pine.get_total( self.indices_pine)[ self.valid]

    #-----
    # PICMerge::get_valid_simulated_rs() returns the simulated stellar radii for the valid matches.
    def get_valid_simulated_rs( self):
        return self.simulated_rs[ self.valid]

    #-----
    # PICMerge::get_valid_simulated_ms() returns the simulated stellar masses for the valid matches.
    def get_valid_simulated_ms( self):
        return self.simulated_ms[ self.valid]




#_______________________________________________________________________________
#
# class PIC2.0.0
#_______________________________________________________________________________
class PIC200( PIC):
    """[Summary]
    
    This class manages the PIC 2.0.0 released in May 2023.

    Attributes:
      seed (int) : the seed for the random number generator.
      rng (numpy.random) : internal random number generator.
      
      valid ()        :
      simulated_rs () :
      simulated_ms () :

    Methods:
      calculate_valid() : method that initializes the attribute valid with a 
        mask that identifies the StarPIC elements having positive (non zero) values 
        of the stellar mass, radius, and effective temperature.
      simulate_stellar_parameters() : the StarPIC catalog contains a radius and a 
        mass estimate, including uncertainties, for each star. What we do here 
        for each star is to simulate a value for the stellar parameters drawn 
        from the expected value and uncertainty stated in the StarPIC. I am 
        actually not sure that this procedure is meaningful.

      get_nelements() : returns the number of elements in the merged catalog.
      get_valid_nelements() : returns the number of elements in the merged.

      get merge() : proxy for get_stars().get().
      get_stars() : builds a Star class with all valid entries in the catalog.

      get_id()           : returns the stellar id of the matches.
      get_GDR3id()       : returns the Gaia DR3 id of the matches.
      get_GDR3idnumber() : returns the Gaia DR3 id number of the matches.
      get_ra()           : returns right ascension of the matches.
      get_de()           : returns the declination of the matches.
      get_vmag()         : returns the V mag of the matches.
      get_gmag()         : returns the G mag of the matches.
      get_pmag()         : returns the PLATO magnitude (N-CAM) of the matches.
      get_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the matches.
      get_pmagred()      : returns the PLATO magnitude red (F-CAM) of the matches.
      get_rs()           : returns the stellar radii of the matches.
      get_urs()          : returns the stellar radius uncertainties of the matches.
      get_ms()           : returns the stellar masses of the matches.
      get_ums()          : returns the stellar mass uncertainties of the matches.
      get_ts()           : returns the stellar effective temperature of the matches.
      get_uts()          : returns the stellar effective temperature uncertainties of the matches.
      get_plx()          : returns the stellar parallax of the matches.
      get_uplx()         : returns the stellar effective temperature uncertainties of the matches.
      get_random()       : returns the random noise value of the matches.
      get_total()        : returns the total noise value of the matches.
      get_bolrandom()    : returns the random noise value of the matches BOL.
      get_boltotal()     : returns the total noise value of the matches BOL.
      get_ncamsbol()     : returns the number of cameras observing the star BOL
      get_ncamseol()     : returns the number of cameras observing the star EOL
      get_simulated_rs() : returns the simulated stellar radii.
      get_simulated_ms() : returns the simulated stellar masses.
      get_skycoord()     : returns an Astropy SkyCoord object with the stars
      get_p1flag()       : returns an array with True value for the P1 targets
      get_p2flag()       : returns an array with True value for the P2 targets
      get_p4flag()       : returns an array with True value for the P4 targets
      get_p5flag()       : returns an array with True value for the P5 targets
      get_planetflag()   : returns an array with True value for the stars known to host a planet

      get_valid_id()           : returns the stellar id of the valid matches.
      get_valid_GDR3id()       : returns the Gaia DR3 id of the valid matches.
      get_valid_GDR3idnumber() : returns the Gaia DR3 id number of the valid matches.
      get_valid_ra()           : returns right ascension of the valid matches.
      get_valid_de()           : returns the declination of the valid matches.
      get_valid_vmag()         : returns the V mag of the valid matches.
      get_valid_gmag()         : returns the G mag of the valid matches.
      get_valid_pmag()         : returns the PLATO magnitude (N-CAM) of the valid matches.
      get_valid_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the valid matches.
      get_valid_pmagred()      : returns the PLATO magnitude red (F-CAM) of the valid matches.
      get_valid_rs()           : returns the stellar radii of the valid matches.
      get_valid_urs()          : returns the stellar radius uncertainties of the valid matches.
      get_valid_ms()           : returns the stellar masses of the valid matches.
      get_valid_ums()          : returns the stellar mass uncertainties of the valid matches.
      get_valid_ts()           : returns the stellar effective temperature of the valid matches.
      get_valid_uts()          : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_plx()          : returns the stellar parallax of the valid matches.
      get_valid_uplx()         : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_random()       : returns the random noise value of the valid matches.
      get_valid_total()        : returns the total noise value of the valid matches.
      get_valid_randombol()    : returns the random noise value BOL of the valid matches.
      get_valid_totalbol()     : returns the total noise value BOL of the valid matches.
      get_valid_ncamsbol()     : returns the number of cameras observing the star BOL of the valid matches.
      get_valid_ncamseol()     : returns the number of cameras observing the star EOL of the valid matches.
      get_valid_simulated_rs() : returns the simulated stellar radii for the valid matches.
      get_valid_simulated_ms() : returns the simulated stellar masses for the valid matches.
    """
    
    seed = 67890
    rng  = Generator( PCG64( seed))

    #-----
    # PIC200::__init__() 
    def __init__( self, fname = None):
        """[Summary]
        
        Method initializing the PIC 2.0.0 class.
        
        Arguments:
          fname (string) : the name of the input file with the catalog data.
        """
        
        if fname is not None:
            self.fname = fname
            self.datos = None

            self.readfile()
            self.calculate_valid()
            self.simulate_stellar_parameters()

    #-----
    # PIC200::readfile() reads the catalog information from the catalog into a
    # astropy Table
    def readfile( self):
        self.datos = apt.Table.read( self.fname, format = 'votable')
        
        def split_gaia_dr3_id( x):
            return x.split( ' ')[ -1]
        self.gaiaDR3no = np.array( [ split_gaia_dr3_id( x) for x in self.datos[ 'StarName']], dtype = 'uint64' )
    
    #-----
    # PIC200::calculate_valid valid entries are those with positive R, M, T
    def calculate_valid( self):
        """[Summary]
        
        Method that initializes the attribute valid with a mask that identifies 
        the PIC elements having positive (non zero) values of the stellar mass,
        radius, and effective temperature.
        """
        self.valid = ( self.get_rs() > 0) * ( self.get_ms() > 0) * ( self.get_ts() > 0)

    #-----
    # PIC200::simulate_stellar_parameters() calculates estimated stellar parameters from a distribution.
    def simulate_stellar_parameters( self):
        """[Summary]
        
        The PIC catalog contains a radius and a  mass estimate, including 
        uncertainties, for each star. What we do here for each star is to 
        simulate a value for the stellar parameters drawn from the expected 
        value and uncertainty stated in the PIC. I am actually not sure that 
        this procedure is meaningful.
        """
        self.simulated_rs = -1 * np.ones( self.get_rs().shape)
        while np.any( self.simulated_rs < 0):
            mask = self.simulated_rs < 0
            self.simulated_rs[ mask] = self.rng.normal( self.get_rs()[ mask], self.get_urs()[ mask])
        self.simulated_ms   = -1 * np.ones( self.get_ms().shape)
        while np.any( self.simulated_ms < 0):
            mask = self.simulated_ms < 0
            self.simulated_ms[ mask] = self.rng.normal( self.get_ms()[ mask], self.get_ums()[ mask])

    #-----
    # PIC200::get_nelements() returns the number of elements in the catalog
    def get_nelements( self):
        return len( self.datos)

    #-----
    # PIC200::get_valid_nelements() returns the number of elements in the  
    # catalog (using the self.valid mask)
    def get_valid_nelements( self):
        return len( np.where( self.valid)[ 0])

    
    #-----
    # PIC200::get_stars() builds a Star class with all valid entries in the
    # catalog.
    def get_stars( self):
        stars = Star( ids  = self.get_valid_id(),
                      ra   = self.get_valid_ra(),
                      de   = self.get_valid_de(),
                      mag  = self.get_valid_vmag(),
                      rs   = self.get_valid_simulated_rs(),
                      urs  = self.get_valid_urs(),
                      ms   = self.get_valid_simulated_ms(),
                      ums  = self.get_valid_ums(),
                      ts   = self.get_valid_ts(),
                      uts  = self.get_valid_uts(),
                      plx  = self.get_valid_plx(),
                      uplx = self.get_valid_uplx(),
                      nsr  = self.get_valid_total())
        return stars
        
    #-----
    # PIC200::get_id() returns the stellar id from the catalog.
    def get_id( self):
        return self.datos[ 'PICid']

    #-----
    # PIC200::get_GDR3id() returns the stellar id from the Gaia DR3 catalogue.
    def get_GDR3id( self):
        return self.datos[ 'StarName']

    #-----
    # PIC200::get_GDR3idnumber() returns the stellar id number from the Gaia DR3 catalogue.
    def get_GDR3idnumber( self):
        return self.gaiaDR3no

    #-----
    # PIC200::get_ra() returns right ascension from the catalog.
    def get_ra( self):
        return self.datos[ 'RAdeg']

    #-----
    # PIC200::get_de() returns the declination from the catalog.
    def get_de( self):
        return self.datos[ 'DEdeg']

    #-----
    # PIC200::get_vmag() returns the V mag from the catalog.
    def get_vmag( self):
        return self.datos[ 'Vmag']

    #-----
    # PIC200::get_gmag() returns the G mag from the catalog.
    def get_gmag( self):
        return self.datos[ 'Gmag']

    #-----
    # PIC200::get_pmag() returns the P mag (N-CAM) from the catalog.
    def get_pmag( self):
        return self.datos[ 'PlatoMagNCAM']
    
    #-----
    # PIC200::get_pmagblue() returns the P mag F-CAM blue from the catalog.
    def get_pmagblue( self):
        return self.datos[ 'PlatoMagFCAMb']

    #-----
    # PIC200::get_pmagred() returns the P mag F-CAM red from the catalog.
    def get_pmagred( self):
        return self.datos[ 'PlatoMagFCAMr']

    #-----
    # PIC200::get_rs() returns the stellar radii from the catalog.
    def get_rs( self):
        return self.datos[ 'Radius']

    #-----
    # PIC200::get_urs() returns the stellar radius uncertainties from the catalog.
    def get_urs( self):
        return self.datos[ 'eRadius']

    #-----
    # PIC200::get_ms() returns the stellar masses from the catalog.
    def get_ms( self):
        return self.datos[ 'Mass']

    #-----
    # PIC200::get_ums() returns the stellar mass uncertainties from the catalog.
    def get_ums( self):
        return self.datos[ 'eMass']

    #-----
    # PIC200::get_ts() returns the stellar effective temperature from the catalog.
    def get_ts( self):
        return self.datos[ 'Teff']

    #-----
    # PIC200::get_uts() returns the stellar effective temperature from the catalog.
    def get_uts( self):
        return self.datos[ 'eTeff']

    #-----
    # PIC200::get_plx() returns the stellar parallax from the catalog.
    def get_plx( self):
        return self.datos[ 'Plx']

    #-----
    # PIC200::get_uplx() returns the stellar effective temperature uncertainties from the catalog.
    def get_uplx( self):
        return self.datos[ 'ePlx']

    #-----
    # PIC200::get_random() returns the random noise value from the catalog.
    def get_random( self):
        return self.datos[ 'EOLrandomNSR']

    #-----
    # PIC200::get_total() returns the total noise value from the catalog.
    def get_total( self):
        return self.datos[ 'EOLrandomSysNSR']

    #-----
    # PIC200::get_randombol() returns the random noise value from the catalog BOL.
    def get_randombol( self):
        return self.datos[ 'BOLrandomNSR']

    #-----
    # PIC200::get_totalbol() returns the total noise value from the catalog BOL.
    def get_totalbol( self):
        return self.datos[ 'BOLrandomSysNSR']

    #-----
    # PIC200::get_ncamsbol() returns the number of cameras observing the star BOL
    def get_ncamsbol( self):
        return self.datos[ 'EOL24nCameraObs']

    #-----
    # PIC200::get_ncamseol() returns the number of cameras observing the star EOL
    def get_ncamseol( self):
        return self.datos[ 'EOLnCameraObs']

    #-----
    # PIC200::get_simulated_rs() returns the simulated stellar radii.
    def get_simulated_rs( self):
        return self.simulated_rs

    #-----
    # PIC200::get_simulated_ms() returns the simulated stellar masses.
    def get_simulated_ms( self):
        return self.simulated_ms

    #-----
    # PIC200::get_skycoord() returns an Astropy SkyCoord object with the
    # coordinates of the stars in the catalogue.
    def get_skycoord( self):
        return SkyCoord( ra = self.get_ra(), dec = self.get_de(), frame = 'icrs')

    #-----
    # PIC200::get_p1flag() returns an array with True value for the P1 targets
    def get_p1flag( self):
        # P1 sample bitmask is 2 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # but P1 is included in P5 (bitmask 8), so all P1 stars have bitmask 10
        # If a P1 stars hosts a planet (bitmask 32), it will be then 42.
        # Note: P2 are also P1, but I am not including P2 in this schema!
        return ( self.datos[ 'BOLsourceFlag'] == 10) + ( self.datos[ 'BOLsourceFlag'] == 42)

    #-----
    # PIC200::get_p2flag() returns an array with True value for the P2 targets
    def get_p2flag( self):
        # P2 sample bitmask is 4 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # but P2 is included in P1 (bitmask 2), which is included in P5 (bitmask 8), 
        # so all P2 stars have bitmask 14, unless they host a planet (bitmask 32), 
        # in which case they have bitmask 46.
        return ( self.datos[ 'BOLsourceFlag'] == 14) + ( self.datos[ 'BOLsourceFlag'] == 46)

    #-----
    # PIC200::get_p4flag() returns an array with True value for the P4 targets
    def get_p4flag( self):
        # P4 sample bitmask is 16 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # unless they host a planet (bitmask 32), in which case they have bitmask 48.
        return ( self.datos[ 'BOLsourceFlag'] == 16) + ( self.datos[ 'BOLsourceFlag'] == 48)

    #-----
    # PIC200::get_p5flag() returns an array with True value for the P5 targets
    def get_p5flag( self):
        # P5 sample bitmask is 8 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # unless they host a planet (bitmask 32), in which case they have bitmask 40.
        return ( self.datos[ 'BOLsourceFlag'] == 8) + ( self.datos[ 'BOLsourceFlag'] == 40)

    #-----
    # PIC200::get_planet() returns an array with True value for the stars known to host a planet
    def get_planetflag( self):
        # P1 sample bitmask (10) hosting a planet (bitmask 32) is 42
        # P2 sample bitmask (14) hosting a planet (bitmask 32) is 46
        # P4 sample bitmask (16) hosting a planet (bitmask 32) is 48
        # P5 sample bitmask ( 8) hosting a planet (bitmask 32) is 40
        return ( self.datos[ 'BOLsourceFlag'] == 42) + ( self.datos[ 'BOLsourceFlag'] == 46) + ( self.datos[ 'BOLsourceFlag'] == 48) + ( self.datos[ 'BOLsourceFlag'] == 40)

    #-----
    # PIC200:get_valid_id() returns the stellar id of the valid entries in the catalog.
    def get_valid_id( self):
        return self.get_id()[ self.valid]

    #-----
    # PIC200::get_valid_ra() returns right ascension of the valid entries in the catalog.
    def get_valid_ra( self):
        return self.get_ra()[ self.valid]

    #-----
    # PIC200::get_valid_de() returns the declination of the valid entries in the catalog.
    def get_valid_de( self):
        return self.get_de()[ self.valid]

    #-----
    # PIC200::get_valid_vmag() returns the V mag of the valid entries in the catalog.
    def get_valid_vmag( self):
        return self.get_vmag()[ self.valid]

    #-----
    # PIC200::get_valid_gmag() returns the G mag of the valid entries in the catalog.
    def get_valid_gmag( self):
        return self.get_gmag()[ self.valid]

    #-----
    # PIC200::get_valid_pmag() returns the P mag (N-CAM) of the valid entries in the catalog.
    def get_valid_pmag( self):
        return self.get_pmag()[ self.valid]
    
    #-----
    # PIC200::get_valid_pmagblue() returns the P mag F-CAM blue of the valid entries in the catalog.
    def get_valid_pmagblue( self):
        return self.get_pmagblue()[ self.valid]

    #-----
    # PIC200::get_valid_pmagred() returns the P mag F-CAM red of the valid entries in the catalog.
    def get_valid_pmagred( self):
        return self.get_pmagred()[ self.valid]

    #-----
    # PIC200::get_valid_rs() returns the stellar radii of the valid entries in the catalog.
    def get_valid_rs( self):
        return self.get_rs()[ self.valid]

    #-----
    # PIC200::get_valid_urs() returns the stellar radius uncertainties of the valid entries in the catalog.
    def get_valid_urs( self):
        return self.get_urs()[ self.valid]

    #-----
    # PIC200::get_valid_ms() returns the stellar masses of the valid entries in the catalog.
    def get_valid_ms( self):
        return self.get_ms()[ self.valid]

    #-----
    # PIC200::get_valid_ums() returns the stellar mass uncertainties of the valid entries in the catalog.
    def get_valid_ums( self):
        return self.get_ums()[ self.valid]

    #-----
    # PIC200::get_valid_ts() returns the stellar effective temperature of the valid entries in the catalog.
    def get_valid_ts( self):
        return self.get_ts()[ self.valid]

    #-----
    # PIC200::get_valid_uts() returns the stellar effective temperature uncertainties of the valid entries in the catalog.
    def get_valid_uts( self):
        return self.get_uts()[ self.valid]

    #-----
    # PIC200::get_valid_plx() returns the stellar parallax of the valid entries in the catalog.
    def get_valid_plx( self):
        return self.get_plx()[ self.valid]

    #-----
    # PIC200::get_valid_uplx() returns the stellar parallax uncertainties of the valid entries in the catalog.
    def get_valid_uplx( self):
        return self.get_uplx()[ self.valid]

    #-----
    # PIC200::get_valid_random() returns the random noise value of the valid entries in the catalog.
    def get_valid_random( self):
        return self.get_random()[ self.valid]

    #-----
    # PIC200::get_valid_total() returns the total noise value of the valid entries in the catalog.
    def get_valid_total( self):
        return self.get_total()[ self.valid]

    #-----
    # PIC200::get_valid_randombol() returns the random noise value BOL of the valid entries in the catalog.
    def get_valid_randombol( self):
        return self.get_randombol()[ self.valid]

    #-----
    # PIC200::get_valid_totalbol() returns the total noise value BOL of the valid entries in the catalog.
    def get_valid_totalbol( self):
        return self.get_totalbol()[ self.valid]

    #-----
    # PIC200::get_valid_ncamsbol() returns the number of cameras observing the star BOL of the valid matches.
    def get_valid_ncamsbol( self):
        return self.get_ncamsbol()[ self.valid]

    #-----
    # PIC200::get_valid_ncamseol() returns the number of cameras observing the star EOL of the valid matches.
    def get_valid_ncamseol( self):
        return self.get_ncamseol()[ self.valid]

    #-----
    # PIC200::get_valid_simulated_rs() returns the simulated stellar radii for the valid entries in the catalog.
    def get_valid_simulated_rs( self):
        return self.get_simulated_rs()[ self.valid]

    #-----
    # PIC200::get_valid_simulated_ms() returns the simulated stellar masses for the valid entries in the catalog.
    def get_valid_simulated_ms( self):
        return self.get_simulated_ms()[ self.valid]

# ['PICid',
# 'PICname',
# 'StarName',
# 'RAdeg',
# 'eRAdeg',
# 'DEdeg',
# 'eDEdeg',
# 'Plx',
# 'ePlx',
# 'pmRA',
# 'epmRA',
# 'pmDE',
# 'epmDE',
# 'pm',
# 'epm',
# 'distance',
# 'eDistance',
# 'refEpoch',
# 'GLON',
# 'GLAT',
# 'ELON',
# 'ELAT',
# 'PlatoMagNCAM',
# 'ePlatoMagNCAM',
# 'PlatoMagFCAMb',
# 'ePlatoMagFCAMb',
# 'PlatoMagFCAMr',
# 'ePlatoMagFCAMr',
# 'Gmag',
# 'eGmag',
# 'BPmag',
# 'eBPmag',
# 'RPmag',
# 'eRPmag',
# 'Ksmag',
# 'eKsmag',
# 'AG',
# 'eAG',
# 'EBPRP',
# 'eEBPRP',
# 'extStatus',
# 'BPRP0',
# 'eBPRP0',
# 'V0mag',
# 'eV0mag',
# 'Vmag',
# 'eVmag',
# 'MG0',
# 'eMG0',
# 'Teff',
# 'eTeff',
# 'Radius',
# 'eRadius',
# 'Mass',
# 'eMass',
# 'BOLsourceFlag',
# 'EOLsourceFlag',
# 'EOL24sourceFlag',
# 'NSSflag',
# 'qualityFlag',
# 'statusFlag',
# 'BOLrandomNSR',
# 'BOLrandomSysNSR',
# 'BOLnCameraObs',
# 'BOLnCameraSat',
# 'EOLrandomNSR',
# 'EOLrandomSysNSR',
# 'EOLnCameraObs',
# 'EOLnCameraSat',
# 'EOL24randomNSR',
# 'EOL24randomSysNSR',
# 'EOL24nCameraObs',
# 'EOL24nCameraSat']




#_______________________________________________________________________________
#
# class PIC 2.0.0 released for NSR computation
#_______________________________________________________________________________
class PIC200NSRinputs( PIC):
    """[Summary]
    
    This class manages the csv files released by the PIC team (MPSSR) to compute 
    the NSR values for the cPIC, fgPIC, and scvPIC 2.0.0 release scheduled for 
    2025. The files were released to DLR in June 2024.

    Attributes:
      fname (string) : name of the input file
        
    Methods:

      get_nelements() : returns the number of elements in the merged catalog.

      get_id()           : returns the stellar id of the targets.
      get_gaia_id()      : returns the Gaia DR3 id of the targets.
      get_ra()           : returns right ascension of the targets.
      get_de()           : returns the declination of the targets.
      get_pmag()         : returns the PLATO magnitude (N-CAM) of the targets.
      get_pmagb()        : returns the PLATO blue magnitude (F-CAM blue) of the targets.
      get_pmagr()        : returns the PLATO red magnitude (F-CAM red) of the targets.
      get_gmag()         : returns the Gaia G magnitude of the targets.
      get_gbpmag()       : returns the Gaia BP magnitude of the targets.
      get_grpmag()       : returns the Gaia RP magnitude of the targets.
      get_vmag()         : returns the Johnsons V magnitude of the targets.
      get_teff()         : returns the stellar effective temperature of the targets.
      get_uteff()        : returns the uncertainty on the stellar effective temperature of the targets.
      get_radius()       : returns the stellar radius of the targets.
      get_uradius()      : returns the uncertainty on the stellar radius of the targets.
      get_mass()         : returns the stellar mass of the targets.
      get_umass()        : returns the uncertainty on the stellar mass of the targets.
      get_skycoord()     : returns an Astropy SkyCoord object with the stars
                         in the catalogue.
    """
    
    #-----
    # PIC200NSRinputs::__init__() 
    def __init__( self, fname = None):
        """[Summary]
        
        Method initializing the class managing the files released by the PIC 
        team (MPSSR) to compute the NSR values for the cPIC, fgPIC, and scvPIC
        2.0.0 release scheduled for 2025.
        
        Arguments:
          fname (string) : the name of the input file with the catalog data.
        """
                
        if fname is not None:
            self.fname = fname
            self.datos = None
            
            self.readfile()
    
    #-----
    # PIC200NSRinputs::readfile() reads the catalog information from the catalog into
    # a numpy array
    def readfile( self):

        # conversion
        def conversion( fld):
            if fld == '\\N':
                return np.nan
            if fld == '':
                return np.nan
            else:
                return float( fld)

        self.datos = pd.read_csv( self.fname, sep = ',', header = 'infer', 
                                  converters = { 'PlatoMag' : conversion, 'Gmag' : conversion, 'RPmag' : conversion, 'BPmag' : conversion, 'Hpmag' : conversion, 'Vmag' : conversion})
    
    #-----
    # PIC200NSRinputs::get_nelements() returns the number of elements in the catalog
    def get_nelements( self):
        return len( self.datos)
    
    #-----
    # PIC200NSRinputs::get_id() returns the stellar id from the catalog.
    def get_id( self):
        if 'cPICid' in self.datos.keys():
            return np.array( self.datos[ 'cPICid'], dtype = 'int64')
        elif 'fgPICid' in self.datos.keys():
            return np.array( self.datos[ 'fgPICid'], dtype = 'int64')
        elif 'scvPICid' in self.datos.keys():
            return np.array( self.datos[ 'scvPICid'], dtype = 'int64')
        else:
            return np.zeros( len( self.datos), dtype = 'int64')
    
    #-----
    # PIC200NSRinputs::get_gaia_id() returns Gaia DR3 id from the catalog.
    def get_gaia_id( self):
        return np.array( self.datos[ 'StarName'], dtype = 'str')

    #-----
    # PIC200NSRinputs::get_ra() returns right ascension from the catalog.
    def get_ra( self):
        return np.array( self.datos[ 'RAdeg'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_de() returns the declination from the catalog.
    def get_de( self):
        return np.array( self.datos[ 'DEdeg'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_pmag() returns the PLATO mag from the catalog.
    def get_pmag( self):    
        return np.array( self.datos[ 'PlatoMagNCAM'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_pmagb() returns the PLATO mag from the catalog.
    def get_pmagb( self):    
        return np.array( self.datos[ 'PlatoMagFCAMb'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_pmagr() returns the PLATO mag from the catalog.
    def get_pmagr( self):    
        return np.array( self.datos[ 'PlatoMagFCAMr'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_gmag() returns the Gaia G from the catalog.
    def get_gmag( self):
        return np.array( self.datos[ 'Gmag'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_gbpmag() returns the Gaia BP from the catalog.
    def get_gbpmag( self):
        return np.array( self.datos[ 'BPmag'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_grpmag() returns the Gaia RP from the catalog.
    def get_grpmag( self):
        return np.array( self.datos[ 'RPmag'], dtype = 'float')
            
    #-----
    # PIC200NSRinputs::get_vmag() returns the Johnsons V magnitude from the catalog.
    def get_vmag( self):
        return np.array( self.datos[ 'Vmag'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_teff() returns the stellar effective temperature from the catalog.
    def get_teff( self):
        return np.array( self.datos[ 'Teff'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_uteff() returns the uncertainty on the stellar effective temperature from the catalog.
    def get_uteff( self):
        return np.array( self.datos[ 'eTeff'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_radius() returns the stellar radius from the catalog.
    def get_radius( self):
        return np.array( self.datos[ 'Radius'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_uradius() returns the uncertainty on teh stellar radius from the catalog.
    def get_uradius( self):
        return np.array( self.datos[ 'eRadius'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_mass() returns the stellar mass from the catalog.
    def get_mass( self):
        return np.array( self.datos[ 'Mass'], dtype = 'float')
    
    #-----
    # PIC200NSRinputs::get_umass() returns the uncertainty on teh stellar mass from the catalog.
    def get_umass( self):
        return np.array( self.datos[ 'eMass'], dtype = 'float')

    #-----
    # PIC200NSRinputs::get_skycoord() returns an Astropy SkyCoord object with the
    # coordinates of the stars in the catalogue.
    def get_skycoord( self):
        return SkyCoord( ra = self.get_ra()*u.deg, dec = self.get_de()*u.deg, frame = 'icrs')



#_______________________________________________________________________________
#
# class PIC2.1.0 for NSR computation
#_______________________________________________________________________________
class PIC210NSRinputs( PIC):
    """[Summary]
    
    This class manages the files released by the PIC team to compute the NSR
    values for the tPIC 2.1.0 release scheduled for 2025. The files were 
    released to DLR in March 2024.
    
    It works also for the PIC 2.2.0.1 inputs provided to DLR in June 2025.

    Attributes:
      fname (string) : name of the input file
        
    Methods:

      get_nelements() : returns the number of elements in the merged catalog.

      get_id()           : returns the stellar id of the targets.
      get_ra()           : returns right ascension of the targets.
      get_de()           : returns the declination of the targets.
      get_pmag()         : returns the PLATO magnitude (N-CAM) of the targets.
      get_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the targets.
      get_pmaredg()      : returns the PLATO magnitude red (F-CAM) of the targets.
      get_gmag()         : returns the Gaia G magnitude of the targets.
      get_gbpmag()       : returns the Gaia BP magnitude of the targets.
      get_grpmag()       : returns the Gaia RP magnitude of the targets.
      get_hpmag()        : returns the Hypparcos magnitude of the targets.
      get_vmag()         : returns the Johnsons V magnitude of the targets.
      get_skycoord()     : returns an Astropy SkyCoord object with the stars
                         in the catalogue.
    """
    
    #-----
    # PIC210NSRinputs::__init__() 
    def __init__( self, fname = None):
        """[Summary]
        
        Method initializing the class managing the files released by the PIC 
        team to compute the NSR values for the PIC 2.1.0 release scheduled in 
        2025.
        
        Arguments:
          fname (string) : the name of the input file with the catalog data.
        """
                
        if fname is not None:
            self.fname = fname
            self.datos = None
            
            self.readfile()
    
    #-----
    # PIC210NSRinputs::readfile() reads the catalog information from the catalog into a
    # numpy array
    def readfile( self):

        # conversion
        def conversion( fld):
            if fld == '\\N':
                return np.nan
            if fld == '':
                return np.nan
            else:
                return float( fld)

        self.datos = pd.read_csv( self.fname, sep = ',', header = 'infer', 
                                  converters = { 'PlatoMag' : conversion, 'Gmag' : conversion, 'RPmag' : conversion, 'BPmag' : conversion, 'Hpmag' : conversion, 'Vmag' : conversion})
    
    #-----
    # PIC210NSRinputs::get_nelements() returns the number of elements in the catalog
    def get_nelements( self):
        return len( self.datos)
    
    #-----
    # PIC210NSRinputs::get_id() returns the stellar id from the catalog.
    def get_id( self):
        if 'tPICid' in self.datos.keys():
            return np.array( self.datos[ 'tPICid'], dtype = 'int64')
        elif 'InternalContaminantId' in self.datos.keys():
            return np.array( self.datos[ 'InternalContaminantId'], dtype = 'int64')
        elif 'scvPICid' in self.datos.keys():
            return np.array( self.datos[ 'scvPICid'], dtype = 'int64')
        elif 'fgPICid' in self.datos.keys():
            return np.array( self.datos[ 'fgPICid'], dtype = 'int64')
        elif 'cPICid' in self.datos.keys():
            return np.array( self.datos[ 'cPICid'], dtype = 'int64')
        else:
            return np.zeros( len( self.datos), dtype = 'int64')
    
    #-----
    # PIC210NSRinputs::get_ra() returns right ascension from the catalog.
    def get_ra( self):
        return np.array( self.datos[ 'RAdeg'], dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_de() returns the declination from the catalog.
    def get_de( self):
        return np.array( self.datos[ 'DEdeg'], dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_pmag() returns the PLATO mag from the catalog.
    def get_pmag( self):    
        return np.array( self.datos[ 'PlatoMag'], dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_pmagblue() returns the PLATO mag for F-CAM blue from the catalog.
    def get_pmagblue( self):    
        if 'PlatoMagFCAMb' in self.datos.keys():
            return np.array( self.datos[ 'PlatoMagFCAMb'], dtype = 'float')
        else:
            return np.zeros( self.get_pmag().shape)

    #-----
    # PIC210NSRinputs::get_pmagred() returns the PLATO mag for F-CAM blue from the catalog.
    def get_pmagred( self):    
        if 'PlatoMagFCAMr' in self.datos.keys():
            return np.array( self.datos[ 'PlatoMagFCAMr'], dtype = 'float')
        else:
            return np.zeros( self.get_pmag().shape)

    #-----
    # PIC210NSRinputs::get_gmag() returns the Gaia G from the catalog.
    def get_gmag( self):
        if 'Gmag' in self.datos.keys():
            return np.array( self.datos[ 'Gmag'], dtype = 'float')
        else:
            return np.zeros( len( self.datos), dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_gbpmag() returns the Gaia BP from the catalog.
    def get_gbpmag( self):
        if 'BPmag' in self.datos.keys():
            return np.array( self.datos[ 'BPmag'], dtype = 'float')
        else:
            return np.zeros( len( self.datos), dtype = 'float')

    #-----
    # PIC210NSRinputs::get_grpmag() returns the Gaia RP from the catalog.
    def get_grpmag( self):
        if 'RPmag' in self.datos.keys():
            return np.array( self.datos[ 'RPmag'], dtype = 'float')
        else:
            return np.zeros( len( self.datos), dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_hpmag() returns the Hypparcos magnitude from the catalog.
    def get_hpmag( self):
        if 'Hpmag' in self.datos.keys():
            return np.array( self.datos[ 'Hpmag'], dtype = 'float')
        else:
            return np.zeros( len( self.datos), dtype = 'float')
        
    #-----
    # PIC210NSRinputs::get_vmag() returns the Johnsons V magnitude from the catalog.
    def get_vmag( self):
        if 'Vmag' in self.datos.keys():
            return np.array( self.datos[ 'Vmag'], dtype = 'float')
        else:
            return np.zeros( len( self.datos), dtype = 'float')
    
    #-----
    # PIC210NSRinputs::get_skycoord() returns an Astropy SkyCoord object with the
    # coordinates of the stars in the catalogue.
    def get_skycoord( self):
        return SkyCoord( ra = self.get_ra()*u.deg, dec = self.get_de()*u.deg, frame = 'icrs')





#_______________________________________________________________________________
#
# class dealing with the preliminary release of the tPIC 2.1.0.2 in November
# 2024 (see email P. Marrese 05.11.2024) for computations for the PSWT in 
# November 2024 (28.-29.11.2024 in Cannes)
#_______________________________________________________________________________
class TPIC2102preliminary( PIC):
    """[Summary]
    
    This class manages the preliminary release of the tPIC 2.1.0.2 in November
    2024. It is built following the structure of the class PIC200.

    Attributes:
      seed (int) : the seed for the random number generator.
      rng (numpy.random) : internal random number generator.
      
      valid ()        :
      simulated_rs () :
      simulated_ms () :

    Methods:
      calculate_valid() : method that initializes the attribute valid with a 
        mask that identifies the StarPIC elements having positive (non zero) values 
        of the stellar mass, radius, and effective temperature.
      simulate_stellar_parameters() : the StarPIC catalog contains a radius and a 
        mass estimate, including uncertainties, for each star. What we do here 
        for each star is to simulate a value for the stellar parameters drawn 
        from the expected value and uncertainty stated in the StarPIC. I am 
        actually not sure that this procedure is meaningful.

      get_nelements() : returns the number of elements in the merged catalog.
      get_valid_nelements() : returns the number of elements in the merged.

      get merge() : proxy for get_stars().get().
      get_stars() : builds a Star class with all valid entries in the catalog.

      get_id()           : returns the stellar id of the matches.
      get_GDR3id()       : returns the Gaia DR3 id of the matches.
      get_GDR3idnumber() : returns the Gaia DR3 id number of the matches.
      get_ra()           : returns right ascension of the matches.
      get_de()           : returns the declination of the matches.
      get_vmag()         : returns the V mag of the matches.
      get_gmag()         : returns the G mag of the matches.
      get_pmag()         : returns the PLATO magnitude (N-CAM) of the matches.
      get_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the matches.
      get_pmagred()      : returns the PLATO magnitude red (F-CAM) of the matches.
      get_rs()           : returns the stellar radii of the matches.
      get_urs()          : returns the stellar radius uncertainties of the matches.
      get_ms()           : returns the stellar masses of the matches.
      get_ums()          : returns the stellar mass uncertainties of the matches.
      get_ts()           : returns the stellar effective temperature of the matches.
      get_uts()          : returns the stellar effective temperature uncertainties of the matches.
      get_plx()          : returns the stellar parallax of the matches.
      get_uplx()         : returns the stellar effective temperature uncertainties of the matches.
      get_random()       : returns the random noise value of the matches.
      get_total()        : returns the total noise value of the matches.
      get_bolrandom()    : returns the random noise value of the matches BOL.
      get_boltotal()     : returns the total noise value of the matches BOL.
      get_ncamsbol()     : returns the number of cameras observing the star BOL
      get_ncamseol()     : returns the number of cameras observing the star EOL
      get_simulated_rs() : returns the simulated stellar radii.
      get_simulated_ms() : returns the simulated stellar masses.
      get_skycoord()     : returns an Astropy SkyCoord object with the stars
      get_p1flag()       : returns an array with True value for the P1 targets
      get_p2flag()       : returns an array with True value for the P2 targets
      get_p4flag()       : returns an array with True value for the P4 targets
      get_p5flag()       : returns an array with True value for the P5 targets
      get_planetflag()   : returns an array with True value for the stars known to host a planet

      get_valid_id()           : returns the stellar id of the valid matches.
      get_valid_GDR3id()       : returns the Gaia DR3 id of the valid matches.
      get_valid_GDR3idnumber() : returns the Gaia DR3 id number of the valid matches.
      get_valid_ra()           : returns right ascension of the valid matches.
      get_valid_de()           : returns the declination of the valid matches.
      get_valid_vmag()         : returns the V mag of the valid matches.
      get_valid_gmag()         : returns the G mag of the valid matches.
      get_valid_pmag()         : returns the PLATO magnitude (N-CAM) of the valid matches.
      get_valid_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the valid matches.
      get_valid_pmagred()      : returns the PLATO magnitude red (F-CAM) of the valid matches.
      get_valid_rs()           : returns the stellar radii of the valid matches.
      get_valid_urs()          : returns the stellar radius uncertainties of the valid matches.
      get_valid_ms()           : returns the stellar masses of the valid matches.
      get_valid_ums()          : returns the stellar mass uncertainties of the valid matches.
      get_valid_ts()           : returns the stellar effective temperature of the valid matches.
      get_valid_uts()          : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_plx()          : returns the stellar parallax of the valid matches.
      get_valid_uplx()         : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_random()       : returns the random noise value of the valid matches.
      get_valid_total()        : returns the total noise value of the valid matches.
      get_valid_randombol()    : returns the random noise value BOL of the valid matches.
      get_valid_totalbol()     : returns the total noise value BOL of the valid matches.
      get_valid_ncamsbol()     : returns the number of cameras observing the star BOL of the valid matches.
      get_valid_ncamseol()     : returns the number of cameras observing the star EOL of the valid matches.
      get_valid_simulated_rs() : returns the simulated stellar radii for the valid matches.
      get_valid_simulated_ms() : returns the simulated stellar masses for the valid matches.
    """
    
    seed = 67890
    rng  = Generator( PCG64( seed))

    #-----
    # TPIC2102preliminary::__init__() 
    def __init__( self, fname = None):
        """[Summary]
        
        Method initializing the tPIC 2.1.0.2 class.
        
        Arguments:
          fname (string) : the name of the input file with the catalog data.
        """
        
        if fname is not None:
            self.fname = fname
            self.datos = None

            self.readfile()
            self.calculate_valid()
            self.simulate_stellar_parameters()

    #-----
    # TPIC2102preliminary::readfile() reads the catalog information from the catalog into a
    # astropy Table
    def readfile( self):

        # conversion
        def conversion( fld):
            if fld == '\\N':
                return np.nan
            if fld == '':
                return np.nan
            else:
                return float( fld)

        self.datos = pd.read_csv( self.fname, sep = ',', header = 'infer', 
                                  converters = { 'PlatoMag' : conversion, 'Gmag' : conversion, 'RPmag' : conversion, 'BPmag' : conversion, 'Hpmag' : conversion, 'Vmag' : conversion})

        self.datos = apt.Table.from_pandas( self.datos)

        # here we have an issue with HIP 28393 B. As for now, we set it's 
        # Gaia DR3 number to 0 and hope for the best
        def split_gaia_dr3_id( x):
            split = x.split( ' ')
            if split[ 0] == 'Gaia':
                return split[ -1]
            else:
                return 0
        
        self.gaiaDR3no = np.array( [ split_gaia_dr3_id( x) for x in self.datos[ 'StarName']], dtype = 'uint64' )
    
    #-----
    # TPIC2102preliminary::calculate_valid valid entries are those with positive R, M, T
    def calculate_valid( self):
        """[Summary]
        
        Method that initializes the attribute valid with a mask that identifies 
        the PIC elements having positive (non zero) values of the stellar mass,
        radius, and effective temperature.
        
        """
        self.valid = ( self.get_rs() > 0) * ( self.get_ms() > 0) * ( self.get_ts() > 0) * ( self.get_total() > 0)

    #-----
    # TPIC2102preliminary::simulate_stellar_parameters() calculates estimated stellar parameters from a distribution.
    def simulate_stellar_parameters( self):
        """[Summary]
        
        The PIC catalog contains a radius and a  mass estimate, including 
        uncertainties, for each star. What we do here for each star is to 
        simulate a value for the stellar parameters drawn from the expected 
        value and uncertainty stated in the PIC. I am actually not sure that 
        this procedure is meaningful.
        """
        self.simulated_rs = -1 * np.ones( self.get_rs().shape)
        while np.any( self.simulated_rs < 0):
            mask = self.simulated_rs < 0
            self.simulated_rs[ mask] = self.rng.normal( self.get_rs()[ mask], self.get_urs()[ mask])
        self.simulated_ms   = -1 * np.ones( self.get_ms().shape)
        while np.any( self.simulated_ms < 0):
            mask = self.simulated_ms < 0
            self.simulated_ms[ mask] = self.rng.normal( self.get_ms()[ mask], self.get_ums()[ mask])

    #-----
    # TPIC2102preliminary::get_nelements() returns the number of elements in the catalog
    def get_nelements( self):
        return len( self.datos)

    #-----
    # TPIC2102preliminary::get_valid_nelements() returns the number of elements in the  
    # catalog (using the self.valid mask)
    def get_valid_nelements( self):
        return len( np.where( self.valid)[ 0])

    
    #-----
    # TPIC2102preliminary::get_stars() builds a Star class with all valid entries in the
    # catalog.
    def get_stars( self):
        stars = Star( ids  = self.get_valid_id(),
                      ra   = self.get_valid_ra(),
                      de   = self.get_valid_de(),
                      mag  = self.get_valid_vmag(),
                      rs   = self.get_valid_simulated_rs(),
                      urs  = self.get_valid_urs(),
                      ms   = self.get_valid_simulated_ms(),
                      ums  = self.get_valid_ums(),
                      ts   = self.get_valid_ts(),
                      uts  = self.get_valid_uts(),
                      plx  = self.get_valid_plx(),
                      uplx = self.get_valid_uplx(),
                      nsr  = self.get_valid_total())
        return stars
        
    #-----
    # TPIC2102preliminary::get_id() returns the stellar id from the catalog.
    # not defined in the catalogue
    def get_id( self):
        return self.get_GDR3idnumber()

    #-----
    # TPIC2102preliminary::get_GDR3id() returns the stellar id from the Gaia DR3 catalogue.
    def get_GDR3id( self):
        return self.datos[ 'StarName']

    #-----
    # TPIC2102preliminary::get_GDR3idnumber() returns the stellar id number from the Gaia DR3 catalogue.
    def get_GDR3idnumber( self):
        return self.gaiaDR3no

    #-----
    # TPIC2102preliminary::get_ra() returns right ascension from the catalog.
    def get_ra( self):
        return self.datos[ 'RAdeg']

    #-----
    # TPIC2102preliminary::get_de() returns the declination from the catalog.
    def get_de( self):
        return self.datos[ 'DEdeg']

    #-----
    # TPIC2102preliminary::get_vmag() returns the V mag from the catalog.
    def get_vmag( self):
        return self.datos[ 'VmagCalculated']

    #-----
    # TPIC2102preliminary::get_gmag() returns the G mag from the catalog.
    def get_gmag( self):
        return self.datos[ 'Gmag']

    #-----
    # TPIC2102preliminary::get_pmag() returns the P mag (N-CAM) from the catalog.
    def get_pmag( self):
        return self.datos[ 'PlatoMagNCAM']
    
    #-----
    # TPIC2102preliminary::get_pmagblue() returns the P mag F-CAM blue from the catalog.
    def get_pmagblue( self):
        return self.datos[ 'PlatoMagFCAMb']

    #-----
    # TPIC2102preliminary::get_pmagred() returns the P mag F-CAM red from the catalog.
    def get_pmagred( self):
        return self.datos[ 'PlatoMagFCAMr']

    #-----
    # TPIC2102preliminary::get_rs() returns the stellar radii from the catalog.
    def get_rs( self):
        return self.datos[ 'Radius']

    #-----
    # TPIC2102preliminary::get_urs() returns the stellar radius uncertainties from the catalog.
    def get_urs( self):
        return self.datos[ 'eRadius']

    #-----
    # TPIC2102preliminary::get_ms() returns the stellar masses from the catalog.
    def get_ms( self):
        return self.datos[ 'Mass']

    #-----
    # TPIC2102preliminary::get_ums() returns the stellar mass uncertainties from the catalog.
    def get_ums( self):
        return self.datos[ 'eMass']

    #-----
    # TPIC2102preliminary::get_ts() returns the stellar effective temperature from the catalog.
    def get_ts( self):
        return self.datos[ 'Teff']

    #-----
    # TPIC2102preliminary::get_uts() returns the stellar effective temperature from the catalog.
    def get_uts( self):
        return self.datos[ 'eTeff']

    #-----
    # TPIC2102preliminary::get_plx() returns the stellar parallax from the catalog.
    def get_plx( self):
        return self.datos[ 'Plx']

    #-----
    # TPIC2102preliminary::get_uplx() returns the stellar effective temperature uncertainties from the catalog.
    def get_uplx( self):
        return self.datos[ 'ePlx']

    #-----
    # TPIC2102preliminary::get_random() returns the random noise value from the catalog.
    # currently, only BOL implemented
    def get_random( self):
        return self.get_randombol()

    #-----
    # TPIC2102preliminary::get_total() returns the total noise value from the catalog.
    # currently, only BOL implemented
    def get_total( self):
        return self.get_totalbol()

    #-----
    # TPIC2102preliminary::get_randombol() returns the random noise value from the catalog BOL.
    # currently, not implemented
    def get_randombol( self):
        return self.np.zeros( self.get_GDR3id().shape, dtype = 'float')

    #-----
    # TPIC2102preliminary::get_totalbol() returns the total noise value from the catalog BOL.
    def get_totalbol( self):
        return self.datos[ 'BOLrandomSysNSRNCAM_T']

    #-----
    # TPIC2102preliminary::get_ncamsbol() returns the number of cameras observing the star BOL
    def get_ncamsbol( self):
        return self.datos[ 'BOLnCameraObsNCAM_T']

    #-----
    # TPIC2102preliminary::get_ncamseol() returns the number of cameras observing the star EOL
    # currently, only BOL implemented
    def get_ncamseol( self):
        return self.get_ncamsbol()

    #-----
    # TPIC2102preliminary::get_simulated_rs() returns the simulated stellar radii.
    def get_simulated_rs( self):
        return self.simulated_rs

    #-----
    # TPIC2102preliminary::get_simulated_ms() returns the simulated stellar masses.
    def get_simulated_ms( self):
        return self.simulated_ms

    #-----
    # TPIC2102preliminary::get_skycoord() returns an Astropy SkyCoord object with the
    # coordinates of the stars in the catalogue.
    def get_skycoord( self):
        return SkyCoord( ra = self.get_ra(), dec = self.get_de(), unit = 'deg', frame = 'icrs')

    #-----
    # TPIC2102preliminary::get_p1flag() returns an array with True value for the P1 targets
    def get_p1flag( self):
        # P1 sample bitmask is 2 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # but P1 is included in P5 (bitmask 8), so all P1 stars have bitmask 10
        # If a P1 stars hosts a planet (bitmask 32), it will be then 42.
        # Note: P2 are also P1, but I am not including P2 in this schema!
        return ( self.datos[ 'BOLsourceFlag'] == 10) + ( self.datos[ 'BOLsourceFlag'] == 42)

    #-----
    # TPIC2102preliminary::get_p2flag() returns an array with True value for the P2 targets
    def get_p2flag( self):
        # P2 sample bitmask is 4 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # but P2 is included in P1 (bitmask 2), which is included in P5 (bitmask 8), 
        # so all P2 stars have bitmask 14, unless they host a planet (bitmask 32), 
        # in which case they have bitmask 46.
        return ( self.datos[ 'BOLsourceFlag'] == 14) + ( self.datos[ 'BOLsourceFlag'] == 46)

    #-----
    # TPIC2102preliminary::get_p4flag() returns an array with True value for the P4 targets
    def get_p4flag( self):
        # P4 sample bitmask is 16 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # unless they host a planet (bitmask 32), in which case they have bitmask 48.
        return ( self.datos[ 'BOLsourceFlag'] == 16) + ( self.datos[ 'BOLsourceFlag'] == 48)

    #-----
    # TPIC2102preliminary::get_p5flag() returns an array with True value for the P5 targets
    def get_p5flag( self):
        # P5 sample bitmask is 8 (in binary), see Table 4 in PLATO-SCI-UPD-TN-0020,
        # unless they host a planet (bitmask 32), in which case they have bitmask 40.
        return ( self.datos[ 'BOLsourceFlag'] == 8) + ( self.datos[ 'BOLsourceFlag'] == 40)

    #-----
    # TPIC2102preliminary::get_planet() returns an array with True value for the stars known to host a planet
    def get_planetflag( self):
        # P1 sample bitmask (10) hosting a planet (bitmask 32) is 42
        # P2 sample bitmask (14) hosting a planet (bitmask 32) is 46
        # P4 sample bitmask (16) hosting a planet (bitmask 32) is 48
        # P5 sample bitmask ( 8) hosting a planet (bitmask 32) is 40
        return ( self.datos[ 'BOLsourceFlag'] == 42) + ( self.datos[ 'BOLsourceFlag'] == 46) + ( self.datos[ 'BOLsourceFlag'] == 48) + ( self.datos[ 'BOLsourceFlag'] == 40)

    #-----
    # TPIC2102preliminary:get_valid_id() returns the stellar id of the valid entries in the catalog.
    def get_valid_id( self):
        return self.get_id()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ra() returns right ascension of the valid entries in the catalog.
    def get_valid_ra( self):
        return self.get_ra()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_de() returns the declination of the valid entries in the catalog.
    def get_valid_de( self):
        return self.get_de()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_vmag() returns the V mag of the valid entries in the catalog.
    def get_valid_vmag( self):
        return self.get_vmag()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_gmag() returns the G mag of the valid entries in the catalog.
    def get_valid_gmag( self):
        return self.get_gmag()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_pmag() returns the P mag (N-CAM) of the valid entries in the catalog.
    def get_valid_pmag( self):
        return self.get_pmag()[ self.valid]
    
    #-----
    # TPIC2102preliminary::get_valid_pmagblue() returns the P mag F-CAM blue of the valid entries in the catalog.
    def get_valid_pmagblue( self):
        return self.get_pmagblue()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_pmagred() returns the P mag F-CAM red of the valid entries in the catalog.
    def get_valid_pmagred( self):
        return self.get_pmagred()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_rs() returns the stellar radii of the valid entries in the catalog.
    def get_valid_rs( self):
        return self.get_rs()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_urs() returns the stellar radius uncertainties of the valid entries in the catalog.
    def get_valid_urs( self):
        return self.get_urs()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ms() returns the stellar masses of the valid entries in the catalog.
    def get_valid_ms( self):
        return self.get_ms()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ums() returns the stellar mass uncertainties of the valid entries in the catalog.
    def get_valid_ums( self):
        return self.get_ums()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ts() returns the stellar effective temperature of the valid entries in the catalog.
    def get_valid_ts( self):
        return self.get_ts()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_uts() returns the stellar effective temperature uncertainties of the valid entries in the catalog.
    def get_valid_uts( self):
        return self.get_uts()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_plx() returns the stellar parallax of the valid entries in the catalog.
    def get_valid_plx( self):
        return self.get_plx()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_uplx() returns the stellar parallax uncertainties of the valid entries in the catalog.
    def get_valid_uplx( self):
        return self.get_uplx()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_random() returns the random noise value of the valid entries in the catalog.
    def get_valid_random( self):
        return self.get_random()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_total() returns the total noise value of the valid entries in the catalog.
    def get_valid_total( self):
        return self.get_total()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_randombol() returns the random noise value BOL of the valid entries in the catalog.
    def get_valid_randombol( self):
        return self.get_randombol()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_totalbol() returns the total noise value BOL of the valid entries in the catalog.
    def get_valid_totalbol( self):
        return self.get_totalbol()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ncamsbol() returns the number of cameras observing the star BOL of the valid matches.
    def get_valid_ncamsbol( self):
        return self.get_ncamsbol()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_ncamseol() returns the number of cameras observing the star EOL of the valid matches.
    def get_valid_ncamseol( self):
        return self.get_ncamseol()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_simulated_rs() returns the simulated stellar radii for the valid entries in the catalog.
    def get_valid_simulated_rs( self):
        return self.get_simulated_rs()[ self.valid]

    #-----
    # TPIC2102preliminary::get_valid_simulated_ms() returns the simulated stellar masses for the valid entries in the catalog.
    def get_valid_simulated_ms( self):
        return self.get_simulated_ms()[ self.valid]

# ['StarName',
#  'RAdeg',
#  'eRAdeg',
#  'DEdeg',
#  'eDEdeg',
#  'pmRA',
#  'epmRA',
#  'pmDE',
#  'epmDE',
#  'pm',
#  'epm',
#  'Plx',
#  'ePlx',
#  'distance',
#  'eDistance',
#  'Glon',
#  'Glat',
#  'posEpoch',
#  'refEpoch',
#  'posPropFlag',
#  'PlatoMagNCAM',
#  'ePlatoMagNCAM',
#  'PlatoMagFCAMb',
#  'ePlatoMagFCAMb',
#  'PlatoMagFCAMr',
#  'ePlatoMagFCAMr',
#  'Gmag',
#  'eGmag',
#  'BPmag',
#  'eBPmag',
#  'RPmag',
#  'eRPmag',
#  'Hpmag',
#  'eHpmag',
#  'Ksmag',
#  'eKsmag',
#  'AG',
#  'eAG',
#  'AKs',
#  'eAKs',
#  'EBPRP',
#  'eEBPRP',
#  'extStatus',
#  'VmagCalculated',
#  'eVmagCalculated',
#  'Teff',
#  'eTeff',
#  'Radius',
#  'eRadius',
#  'Mass',
#  'eMass',
#  'caseFlag',
#  'tPICsourceFlagNCAM_BOL',
#  'NSSflag',
#  'qualityFlag',
#  'tPICplanetFlag',
#  'BOLrandomSysNSRNCAM_T',
#  'BOLnCameraObsNCAM_T',
#  'BOLnCameraSatNCAM_T',
#  'tPICscientificRanking']

#_______________________________________________________________________________
#
# class dealing with the 2.1.0.1 release in January/February 2025 of the PIC
#_______________________________________________________________________________
class PIC2101( PIC):
    """[Summary]
    
    This class manages the PIC release 2.1.0.1 in January/February 2025.
    
    There are several differences between this release and the previous major
    release of the PIC in 2023 (which was PIC 2.0.0 or PIC200):
    - The 2023 release only included the tPIC, while the 2025 release includes
      tPIC, scvPIC, cPIC, and fgPIC.
    - The 2023 release included all data (stellar parameters and full PPT NSR 
      analysis) in a single VOTable file, while the 2025 release separates
      the data in different fits files, one for the stellar parameters and 
      another for the NSR analysis.
    - We will not implement a 'simulate_stellar_parameters' anymore. This is a 
      different approach to previous analyses, but it should not be a big deal.

    Attributes:
      target_table_file_name (string) : the name of the FITS file with the PIC 
        target table for the LOPS2 pointing (see chapter 3 in 
        PLATO-SSDC-PDC-DD-0003).
      nsr_table_file_name (string) : the name of the FITS file with the PIC 
        target NSR table for the LOPS2 pointing (see chapter 6 in 
        PLATO-SSDC-PDC-DD-0003).
      stellar_data (numpy.ndarray) : the array with the data in PIC target 
        table extracted from the FITS file.
      nsr_data (numpy.ndarray) : the array with the data in the PIC NSR table 
        extracted from the FITS file
      gaiaDR3no (numpy.ndarray) : a int64 array with the Gaia DR3 id numbers
        of the stars in the catalogue (for convenience).

    Methods:
      readfiles() : method that accesses the fits files and extracts the data
        into the attributes 'stellar_data' and 'nsr_data'
      calculate_valid() : method that initializes the attribute valid with a 
        mask that identifies the StarPIC elements having positive (non zero) 
        values of the stellar mass, radius, effective temperature, and total 
        NSR.

      get_nelements() : returns the number of elements in the merged catalog.
      get_valid_nelements() : returns the number of elements in the merged.

      get merge() : proxy for get_stars().get().
      get_stars() : builds a Star class with all valid entries in the catalog.

      get_id()              : returns the stellar id of the matches.
      get_GDR3id()          : returns the Gaia DR3 id of the matches.
      get_GDR3idnumber()    : returns the Gaia DR3 id number of the matches.
      get_ra()              : returns right ascension of the matches.
      get_de()              : returns the declination of the matches.
      get_vmag()            : returns the V mag of the matches.
      get_gmag()            : returns the G mag of the matches.
      get_pmag()            : returns the PLATO magnitude (N-CAM) of the matches.
      get_pmagblue()        : returns the PLATO magnitude blue (F-CAM) of the matches.
      get_pmagred()         : returns the PLATO magnitude red (F-CAM) of the matches.
      get_rs()              : returns the stellar radii of the matches.
      get_urs()             : returns the stellar radius uncertainties of the matches.
      get_ms()              : returns the stellar masses of the matches.
      get_ums()             : returns the stellar mass uncertainties of the matches.
      get_ts()              : returns the stellar effective temperature of the matches.
      get_uts()             : returns the stellar effective temperature uncertainties of the matches.
      get_logg()            : returns the stellar log g of the matches.
      get_plx()             : returns the stellar parallax of the matches.
      get_uplx()            : returns the stellar effective temperature uncertainties of the matches.
      get_random()          : returns the random noise value of the matches.
      get_total()           : returns the total noise value of the matches.
      get_bolrandom()       : returns the random noise value of the matches BOL.
      get_boltotal()        : returns the total noise value of the matches BOL.
      get_ncamsbol()        : returns the number of cameras observing the star BOL.
      get_ncamseol()        : returns the number of cameras observing the star EOL.
      get_cPICrandomBOL()   : returns the random noise value at camera level BOL.
      get_cPICtotalBOL()    : returns the total noise value at camera level BOL.
      get_cPICrandomEOL()   : returns the random noise value at camera level EOL.
      get_cPICtotalEOL()    : returns the total noise value at camera level EOL
      get_fgPICbrandomBOL() : returns the random noise value at camera level BOL for the F-CAM blue.
      get_fgPICbtotalBOL()  : returns the total noise value at camera level BOL for the F-CAM blue
      get_fgPICbrandomEOL() : returns the random noise value at camera level EOL for the F-CAM blue.
      get_fgPICbtotalEOL()  : returns the total noise value at camera level EOL for the F-CAM blue.
      get_fgPICrrandomBOL() : returns the random noise value at camera level BOL for the F-CAM red.
      get_fgPICrtotalBOL()  : returns the total noise value at camera level BOL for the F-CAM red.
      get_fgPICrrandomEOL() : returns the random noise value at camera level EOL for the F-CAM red.
      get_fgPICrtotalBOL()  : returns the total noise value at camera level EOL for the F-CAM red.
      
      get_simulated_rs() : returns the simulated stellar radii.
      get_simulated_ms() : returns the simulated stellar masses.
      get_skycoord()     : returns an Astropy SkyCoord object with the stars

      get_p1flag()       : returns an array with True values for the P1 
        In principle, P2 belongs to P1, but I am not including P2 here on 
        purpose!
      get_p1flag_including_known_planet_hosts() : returns an array with True 
        value for the P1 targets including stars known to host a planet. In 
        principle, P2 belongs to P1, but I am not including P2 here on purpose!
      
      get_p2flag()       : returns an array with True value for the P2 targets
      get_p2flag_including_known_planet_hosts() : returns an array with True 
        value for the P2 targets including stars known to host a planet.
      
      get_p4flag()       : returns an array with True value for the P4 targets
      get_p4flag_including_known_planet_hostss() : returns an array with True 
        value for the P4 targets including stars known to host a planet.
      
      get_p5flag()       : returns an array with True value for the P5 targets
      get_planetflag()   : returns an array with True value for the stars known 
        to host a planet, irrespective of the sample id.

      get_planetflag() : returns an array with True value for P1, P2, P4, and P5 
        stars (FGK and M) known to host a planet

      get_tPICflag()   : returns an array with True value for the tPIC targets
      get_fgPICflag()  : returns an array with True value for the fgPIC targets
      get_cflag()      : returns an array with True value for the cPIC targets
      get_scvPICflag() : returns an array with True value for the scvPIC targets

      get_tPICP1flag() : returns an array with True value for the P1 targets in 
        the tPIC targets. The list includes P2 stars and stars hosting planets.
      get_tPICP2flag() : returns an array with True value for the P2 targets in 
        the tPIC targets. The list includes stars hosting planets.
      get_tPICP4flag() : returns an array with True value for the P4 targets in 
        the tPIC targets. The list includes stars hosting planets.
      get_tPICP5flag() : returns an array with True value for the P5 targets in 
        the tPIC targets. The list includes stars hosting planets.
      
      get_fgPICbflag() returns an array with True value for the stars in the fgPIC observed by F-CAM blue
      get_fgPICrflag() returns an array with True value for the stars in the fgPIC observed by F-CAM red

      get_cPICR1FCAMflag() returns an array with True value for the stars 
        belonging to the R1 sample (attitude and IGM) in the cPIC observed with 
        the F-CAMs. This result includes stars that also belong to R2, R3, and 
        R4 samples.
      get_cPICR2FCAMflag() returns an array with True value for the stars 
        belonging to the R2 sample (microscanning) in the cPIC observed with 
        the F-CAMs. This result includes stars that also belong to R1, R3, and 
        R4 samples.
      get_cPICR3FCAMflag() returns an array with True value for the stars 
        belonging to the R3 sample (best focus) in the cPIC observed with the 
        F-CAMs. This result includes stars that also belong to R1, R2, and R4 
        samples.
      get_cPICR4FCAMflag() returns an array with True value for the stars 
        belonging to the R4 sample (throughput) in the cPIC observed with the 
        F-CAMs. This result includes stars that also belong to R1, R2, and R3 
        samples.
      get_cPICR5FCAMflag() returns an array with True value for the stars 
        belonging to the R5 sample (outlier rejection) in the cPIC observed
        with the F-CAMs. 

      get_cPICR1flag() returns an array with True value for the stars 
        belonging to the R1 sample (attitude and IGM) in the cPIC observed with 
        the N-CAMs. This result includes stars that also belong to R2, R3, and 
        R4 samples.
      get_cPICR2flag() returns an array with True value for the stars 
        belonging to the R2 sample (microscanning) in the cPIC observed with the 
        N-CAMs. This result includes stars that also belong to R1, R3, and R4 
        samples.
      get_cPICR3flag() returns an array with True value for the stars 
        belonging to the R3 sample (best focus) in the cPIC observed with the 
        N-CAMs. This result includes stars that also belong to R1, R2, and R4 
        samples.
      get_cPICR4flag() returns an array with True value for the stars 
        belonging to the R4 sample (throughput) in the cPIC observed with the 
        N-CAMs. This result includes stars that also belong to R1, R2, and R3 
        samples.
      get_cPICR5flag() returns an array with True value for the stars 
        belonging to the R5 sample (outlier rejection) in the cPIC observed 
        with the N-CAMs. 
     
      get_scvPIC1aflag() returns an array with True value for the stars
        belonging to the SCV1a sample in the scvPIC (eclipsing binaries).
      get_scvPIC1bflag() returns an array with True value for the stars 
        belonging to the SCV1b sample in the scvPIC (astrometric binaries).
      get_scvPIC1cflag() returns an array with True value for the stars 
        belonging to the SCV1c sample in the scvPIC (wide binaries).
      get_scvPIC1dflag() returns an array with True value for the stars 
        belonging to the SCV1d sample in the scvPIC HW Vir-type binaries).
      get_scvPIC2eflag() returns an array with True value for the stars 
        belonging to the SCV1e sample in the scvPIC (wide white dwarf binaries).
      get_scvPIC2aflag() returns an array with True value for the stars 
        belonging to the SCV2a sample in the scvPIC (legacy and benchmark stars).
      get_scvPIC2bflag() returns an array with True value for the stars 
        belonging to the SCV2b sample in the scvPIC (legacy and benchmark stars).
      get_scvPIC3aflag() returns an array with True value for the stars 
        belonging to the SCV3a sample in the scvPIC (photometrically stable stars).
      get_scvPIC3bflag() returns an array with True value for the stars 
        belonging to the SCV3b sample in the scvPIC (photometrically stable stars).
      get_scvPIC4aflag() returns an array with True value for the stars 
        belonging to the SCV4a sample in the scvPIC (solar-like pulsators).
      get_scvPIC4bflag() returns an array with True value for the stars 
        belonging to the SCV4b sample in the scvPIC (solar-like pulsators).
      get_scvPIC5flag() returns an array with True value for the stars 
        belonging to the SCV5 sample in the scvPIC (gamma Dor stars).
      get_scvPIC6flag() returns an array with True value for the stars 
        belonging to the SCV6 sample in the scvPIC (transiting exoplanets).

      get_valid_id()           : returns the stellar id of the valid matches.
      get_valid_GDR3id()       : returns the Gaia DR3 id of the valid matches.
      get_valid_GDR3idnumber() : returns the Gaia DR3 id number of the valid matches.
      get_valid_ra()           : returns right ascension of the valid matches.
      get_valid_de()           : returns the declination of the valid matches.
      get_valid_vmag()         : returns the V mag of the valid matches.
      get_valid_gmag()         : returns the G mag of the valid matches.
      get_valid_pmag()         : returns the PLATO magnitude (N-CAM) of the valid matches.
      get_valid_pmagblue()     : returns the PLATO magnitude blue (F-CAM) of the valid matches.
      get_valid_pmagred()      : returns the PLATO magnitude red (F-CAM) of the valid matches.
      get_valid_rs()           : returns the stellar radii of the valid matches.
      get_valid_urs()          : returns the stellar radius uncertainties of the valid matches.
      get_valid_ms()           : returns the stellar masses of the valid matches.
      get_valid_ums()          : returns the stellar mass uncertainties of the valid matches.
      get_valid_ts()           : returns the stellar effective temperature of the valid matches.
      get_valid_uts()          : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_logg()         : returns the stellar logg of the valid matches.
      get_valid_plx()          : returns the stellar parallax of the valid matches.
      get_valid_uplx()         : returns the stellar effective temperature uncertainties of the valid matches.
      get_valid_random()       : returns the random noise value of the valid matches.
      get_valid_total()        : returns the total noise value of the valid matches.
      get_valid_randombol()    : returns the random noise value BOL of the valid matches.
      get_valid_totalbol()     : returns the total noise value BOL of the valid matches.
      get_valid_ncamsbol()     : returns the number of cameras observing the star BOL of the valid matches.
      get_valid_ncamseol()     : returns the number of cameras observing the star EOL of the valid matches.
      get_valid_simulated_rs() : returns the simulated stellar radii for the valid matches.
      get_valid_simulated_ms() : returns the simulated stellar masses for the valid matches.
    """
    
    #-----
    # PIC2101::__init__() 
    def __init__( self, target_table_file_name = None, nsr_table_file_name = None):
        """[Summary]
        
        Method initializing the PIC 2.1.0.1 class.
        
        Arguments:
          target_table_file_name (string) : the name of the FITS file with the 
            PIC target table for the LOPS2 pointing (see chapter 3 in 
            PLATO-SSDC-PDC-DD-0003).
          nsr_table_file_name (string) : the name of the FITS file with the 
            PIC target NSR table for the LOPS2 pointing (see chapter 6 in 
            PLATO-SSDC-PDC-DD-0003).
        """
        
        if ( target_table_file_name is not None) and ( nsr_table_file_name is not None):
            self.target_table_file_name = target_table_file_name
            self.nsr_table_file_name    = nsr_table_file_name
            
            self.stellar_data = None
            self.nsr_data = None

            self.readfiles()
            self.calculate_valid()

    #-----
    # PIC2101::readfiles() reads the catalog information from the fits files
    def readfiles( self):
        
        # read stellar data
        stellar_hdul = fits.open( self.target_table_file_name)
        self.stellar_data = stellar_hdul[ 1].data
        
        # read NSR data
        nsr_hdul = fits.open( self.nsr_table_file_name)
        self.nsr_data = nsr_hdul[ 1].data

        # here we propose a workaround for HIP 28393 B. As for now, we set it's 
        # Gaia DR3 number to 0 and hope for the best. Actually, it is funny
        # because the Hipparcos catalogue has only the A star of the system 
        # (HIP 28393) with Gaia DR3 4794830231453653888 and a magnitude of ~8.7
        # while the companion (HD 41004 B) has no Gaia DR3 id, magnitude ~12.3
        # and is accompanied by a Jupiter-sized planet/brown dwarf (20 MJ)
        # See Santos et al. 2002 and Zucker et al. 2004.
        def split_gaia_dr3_id( x):
            split = x.split( ' ')
            if split[ 0] == 'Gaia':
                return split[ -1]
            else:
                return 0
        
        self.gaiaDR3no = np.array( [ split_gaia_dr3_id( x) for x in self.stellar_data[ 'StarName']], dtype = 'uint64' )

    
    #-----
    # PIC2101::calculate_valid valid entries are those with positive R, M, T
    # and NSR > 0 (which by default removes nan)
    def calculate_valid( self):
        """[Summary]
        
        Method that initializes the attribute valid with a mask that identifies 
        the PIC elements having positive (non zero) values of the stellar mass,
        radius, effective temperature, and total NSR.
        
        """
        self.valid = ( self.get_rs() > 0) * ( self.get_ms() > 0) * ( self.get_ts() > 0) * ( self.get_total() > 0)


    #-----
    # PIC2101::get_nelements() returns the number of elements in the catalog
    def get_nelements( self):
        return len( self.stellar_data)

    #-----
    # PIC2101::get_valid_nelements() returns the number of elements in the  
    # catalog (using the self.valid mask)
    def get_valid_nelements( self):
        return len( np.where( self.valid)[ 0])

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_stars() builds a Star class with all valid entries in the
    # catalog.
    def get_stars( self):
        stars = Star( ids  = self.get_valid_id(),
                      ra   = self.get_valid_ra(),
                      de   = self.get_valid_de(),
                      mag  = self.get_valid_vmag(),
                      rs   = self.get_valid_simulated_rs(),
                      urs  = self.get_valid_urs(),
                      ms   = self.get_valid_simulated_ms(),
                      ums  = self.get_valid_ums(),
                      ts   = self.get_valid_ts(),
                      uts  = self.get_valid_uts(),
                      plx  = self.get_valid_plx(),
                      uplx = self.get_valid_uplx(),
                      nsr  = self.get_valid_total())
        return stars
        
    #-----
    # PIC2101::get_id() returns the stellar id from the catalog.
    # not defined in the catalogue
    def get_id( self):
        return self.get_GDR3idnumber()

    #-----
    # PIC2101::get_GDR3id() returns the stellar id from the Gaia DR3 catalogue.
    # note:
    # np.all( np.compare_chararrays( pic.stellar_data[ 'StarName'], pic.nsr_data[ 'StarName'], "==", False))
    # True
    def get_GDR3id( self):
        return self.stellar_data[ 'StarName']

    #-----
    # PIC2101::get_GDR3idnumber() returns the stellar id number from the Gaia DR3 catalogue.
    def get_GDR3idnumber( self):
        return self.gaiaDR3no

    #-----
    # PIC2101::get_ra() returns right ascension from the catalog.
    def get_ra( self):
        return self.stellar_data[ 'RAdeg']

    #-----
    # PIC2101::get_de() returns the declination from the catalog.
    def get_de( self):
        return self.stellar_data[ 'DEdeg']

    #-----
    # PIC2101::get_ra() returns right ascension from the catalog.
    def get_pmra( self):
        return self.stellar_data[ 'pmRA']

    #-----
    # PIC2101::get_de() returns the declination from the catalog.
    def get_pmde( self):
        return self.stellar_data[ 'pmDE']
    
    #-----
    # PIC2101::get_vmag() returns the V mag from the catalog.
    def get_vmag( self):
        return self.stellar_data[ 'VmagCalculated']

    #-----
    # PIC2101::get_gmag() returns the G mag from the catalog.
    def get_gmag( self):
        return self.stellar_data[ 'Gmag']

    #-----
    # PIC2101::get_pmag() returns the P mag (N-CAM) from the catalog.
    def get_pmag( self):
        return self.stellar_data[ 'PlatoMagNCAM']
    
    #-----
    # PIC2101::get_pmagblue() returns the P mag F-CAM blue from the catalog.
    def get_pmagblue( self):
        return self.stellar_data[ 'PlatoMagFCAMb']

    #-----
    # PIC2101::get_pmagred() returns the P mag F-CAM red from the catalog.
    def get_pmagred( self):
        return self.stellar_data[ 'PlatoMagFCAMr']

    #-----
    # PIC2101::get_rs() returns the stellar radii from the catalog.
    def get_rs( self):
        return self.stellar_data[ 'Radius']

    #-----
    # PIC2101::get_urs() returns the stellar radius uncertainties from the catalog.
    def get_urs( self):
        return self.stellar_data[ 'eRadius']

    #-----
    # PIC2101::get_ms() returns the stellar masses from the catalog.
    def get_ms( self):
        return self.stellar_data[ 'Mass']

    #-----
    # PIC2101::get_ums() returns the stellar mass uncertainties from the catalog.
    def get_ums( self):
        return self.stellar_data[ 'eMass']

    #-----
    # PIC2101::get_ts() returns the stellar effective temperature from the catalog.
    def get_ts( self):
        return self.stellar_data[ 'Teff']

    #-----
    # PIC2101::get_uts() returns the stellar effective temperature from the catalog.
    def get_uts( self):
        return self.stellar_data[ 'eTeff']

    #-----
    # PIC2101::get_logg() returns the stellar lgog from the catalog.
    def get_logg( self):
        return 4.44 + np.log10( self.get_ms()) - 2.*np.log10( self.get_rs()) 

    #-----
    # PIC2101::get_plx() returns the stellar parallax from the catalog.
    def get_plx( self):
        return self.stellar_data[ 'Plx']

    #-----
    # PIC2101::get_uplx() returns the stellar effective temperature uncertainties from the catalog.
    def get_uplx( self):
        return self.stellar_data[ 'ePlx']

    #-----
    # PIC2101::get_random() returns the random noise value from the catalog.
    # currently, only EOL implemented
    def get_random( self):
        return self.get_randomeol()

    #-----
    # PIC2101::get_total() returns the total noise value from the catalog.
    # currently, only EOL implemented
    def get_total( self):
        return self.get_totaleol()

    #-----
    # PIC2101::get_randombol() returns the random noise value from the catalog BOL.
    def get_randombol( self):
        return self.nsr_data[ 'BOLrandomNSRNCAM_T']

    #-----
    # PIC2101::get_totalbol() returns the total noise value from the catalog BOL.
    def get_totalbol( self):
        return self.nsr_data[ 'BOLrandomSysNSRNCAM_T']

    #-----
    # PIC2101::get_ncamsbol() returns the number of cameras observing the star BOL
    def get_ncamsbol( self):
        return self.nsr_data[ 'BOLnCameraObsNCAM_T']

    #-----
    # PIC2101::get_randomeol() returns the random noise value from the catalog ·OL.
    def get_randomeol( self):
        return self.nsr_data[ 'EOLrandomNSRNCAM_R']

    #-----
    # PIC2101::get_totaleol() returns the total noise value from the catalog BOL.
    def get_totaleol( self):
        return self.nsr_data[ 'EOLrandomSysNSRNCAM_R']

    #-----
    # PIC2101::get_ncamseol() returns the number of cameras observing the star BOL
    def get_ncamseol( self):
        return self.nsr_data[ 'EOLnCameraObsNCAM_R']
    
    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_cPICrandomBOL() returns the random noise value at camera 
    # level BOL (see PLATO-SSDC-PDC-DD-0003, section 6.1, page 32)
    def get_cPICrandomBOL( self):
        return self.nsr_data[ 'BOL1randomNSRNCAM_T']
    
    #-----
    # PIC2101::get_cPICtotalBOL() returns the total noise value at camera 
    # level BOL (see PLATO-SSDC-PDC-DD-0003, section 6.1, page 32)
    def get_cPICtotalBOL( self):
        return self.nsr_data[ 'BOL1randomSysNSRNCAM_T']
    
    #-----
    # PIC2101::get_cPICrandomEOL() returns the random noise value at camera 
    # level EOL (see PLATO-SSDC-PDC-DD-0003, section 6.1, page 33)
    def get_cPICrandomEOL( self):
        return self.nsr_data[ 'EOL1randomNSRNCAM_R']
    
    #-----
    # PIC2101::get_cPICtotalEOL() returns the total noise value at camera 
    # level EOL (see PLATO-SSDC-PDC-DD-0003, section 6.1, page 33)
    def get_cPICtotalEOL( self):
        return self.nsr_data[ 'EOL1randomSysNSRNCAM_R']
    
    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_fgPICbrandomBOL() returns the random noise value at camera 
    # level BOL for the F-CAM blue (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 33)
    def get_fgPICbrandomBOL( self):
        return self.nsr_data[ 'BOLrandomNSRFCAMB_T']
    
    #-----
    # PIC2101::get_fgPICbtotalBOL() returns the total noise value at camera 
    # level BOL for the F-CAM blue (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 33)
    def get_fgPICbtotalBOL( self):
        return self.nsr_data[ 'BOLrandomSysNSRFCAMB_T']
    
    #-----
    # PIC2101::get_fgPICbrandomEOL() returns the random noise value at camera 
    # level EOL for the F-CAM blue (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 34)
    def get_fgPICbrandomEOL( self):
        return self.nsr_data[ 'EOLrandomNSRFCAMB_R']
    
    #-----
    # PIC2101::get_fgPICbtotalEOL() returns the total noise value at camera 
    # level EOL for the F-CAM blue (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 34)
    def get_fgPICbtotalEOL( self):
        return self.nsr_data[ 'EOLrandomSysNSRFCAMB_R']


    #-----
    # PIC2101::get_fgPICrrandomBOL() returns the random noise value at camera 
    # level BOL for the F-CAM red (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 34)
    def get_fgPICrrandomBOL( self):
        return self.nsr_data[ 'BOLrandomNSRFCAMR_T']
    
    #-----
    # PIC2101::get_fgPICrtotalBOL() returns the total noise value at camera 
    # level BOL for the F-CAM red (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 34)
    def get_fgPICrtotalBOL( self):
        return self.nsr_data[ 'BOLrandomSysNSRFCAMR_T']
    
    #-----
    # PIC2101::get_fgPICrrandomEOL() returns the random noise value at camera 
    # level EOL for the F-CAM red (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 35)
    def get_fgPICrrandomEOL( self):
        return self.nsr_data[ 'EOLrandomNSRFCAMR_R']
    
    #-----
    # PIC2101::get_fgPICrtotalBOL() returns the total noise value at camera 
    # level EOL for the F-CAM red (see PLATO-SSDC-PDC-DD-0003, section 6.1, 
    # page 34)
    def get_fgPICrtotalEOL( self):
        return self.nsr_data[ 'EOLrandomSysNSRFCAMR_R']

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_simulated_rs() just in case
    def get_simulated_rs( self):
        return self.get_rs()

    #-----
    # PIC2101::get_simulated_ms() just in case
    def get_simulated_ms( self):
        return self.get_ms()

    #-----
    # PIC2101::get_skycoord() returns an Astropy SkyCoord object with the
    # coordinates of the stars in the catalogue.
    def get_skycoord( self):
        return SkyCoord( ra = self.get_ra(), dec = self.get_de(), unit = 'deg', frame = 'icrs')

    #---------- Added for PICSIM ---------------
    
    def get_caseFlag(self):
        return self.stellar_data['caseFlag']

    def get_PICmainSourceFlagBOL(self):
        return self.stellar_data['PICmainSourceFlagBOL']

    def get_tPICsourceFlagNCAM_BOL(self):
        return self.stellar_data['tPICsourceFlagNCAM_BOL']
    
    def get_fgPICsourceFlag(self):
        return self.stellar_data['fgPICsourceFlag']

    def get_cPICsourceFlag(self):
        return self.stellar_data['cPICsourceFlag']

    def get_scvPICsourceFlag(self):
        return self.stellar_data['scvPICsourceFlag']    
    
    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_p1flag() returns an array with True value for the P1 targets
    def get_p1flag( self):
        # P1 sample bitmask is 1 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        # In principle, P2 belongs to P1, but I am not including P2 here on purpose!
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 1)

    #-----
    # PIC2101::get_p1flag_including_known_planet_hosts() returns an array with True value for the P1 targets
    # including stars known to host a planet
    def get_p1flag_including_known_planet_hosts( self):
        # P1 sample bitmask is 1 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        # but planet hosting stars have bitmask 17 (16+1)
        # In principle, P2 belongs to P1, but I am not including P2 here on purpose!
        return (( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 1) +
                ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 17))

    #-----
    # PIC2101::get_p2flag() returns an array with True value for the P2 targets
    def get_p2flag( self):
        # P2 sample bitmask is 2 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        # but P2 is included in P1 (bitmask 1), so all P2 stars have bitmask 3 (2+1)
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 3) 

    #-----
    # PIC2101::get_p2flag()_including_known_planet_hosts returns an array with True value for the P2 targets
    # including stars known to host a planet
    def get_p2flag_including_known_planet_hosts( self):
        # P2 sample bitmask is 2 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        # but P2 is included in P1 (bitmask 1), so all P2 stars have bitmask 3 (2+1)
        # unless they host a planet (bitmask 16), in which case they have bitmask 19
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 3) + ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 19)

    #-----
    # PIC2101::get_p4flag() returns an array with True value for the P4 targets
    def get_p4flag( self):
        # P4 sample bitmask is 8 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 8)

    #-----
    # PIC2101::get_p4flag()_including_known_planet_hosts returns an array with True value for the P4 targets
    # including stars known to host a planet
    def get_p4flag_including_known_planet_hosts( self):
        # P4 sample bitmask is 8 (in binary) in tPICsourceFlagNCAM_BOL
        # as per PLATO-SSDC-PDC-DN-0001 (page 15) and Table 6 in PLATO-UPD-SCI-TN-0022
        # unless they host a planet (bitmask 16), in which case they have bitmask 24
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 8) + ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 24)

    #-----
    # PIC2101::get_p5flag() returns an array with True value for the P5 targets
    def get_p5flag( self):
        # P5 sample bitmask is 4 (in binary) in tPICsourceFlagNCAM_BOL
        # unless they host a planet (bitmask 16), in which case they have bitmask 24.
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 4)

    #-----
    # PIC2101::get_p5flag() returns an array with True value for the P5 targets
    # including stars known to host a planet
    def get_p5flag_including_known_planet_hosts( self):
        # P5 sample bitmask is 4 (in binary) in tPICsourceFlagNCAM_BOL
        # unless they host a planet (bitmask 16), in which case they have bitmask 24.
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 4) + ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 20)

    #-----
    # PIC2101::get_planetflag() returns an array with True value for P1, P2, P4, and P5 stars known to host a planet
    def get_planetflag( self):
        # planet hosting stars beyond FGK or M have bitmask 16 [we ignore those]
        # P1 sample bitmask ( 1) hosting a planet (bitmask 16) is 17
        # P2 sample bitmask ( 3) hosting a planet (bitmask 16) is 19
        # P4 sample bitmask ( 8) hosting a planet (bitmask 16) is 24
        # P5 sample bitmask ( 4) hosting a planet (bitmask 16) is 20
        return ( 
            ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 17) + 
            ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 19) + 
            ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 24) + 
            ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] == 20)   )

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_tPICflag() returns an array with True value for the stars belonging to the tPIC
    # see chapter 3.1, page 16, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICflag( self):
        return ( self.stellar_data[ 'PICmainSourceFlagBOL'] & 4 == 4)

    #-----
    # PIC2101::get_fgPICflag() returns an array with True value for the stars belonging to the fgPIC
    # see chapter 3.1, page 16, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_fgPICflag( self):
        return ( self.stellar_data[ 'PICmainSourceFlagBOL'] & 8 == 8)

    #-----
    # PIC2101::get_cPICflag() returns an array with True value for the stars belonging to the cPIC
    # see chapter 3.1, page 16, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICflag( self):
        return ( self.stellar_data[ 'PICmainSourceFlagBOL'] & 16 == 16)

    #-----
    # PIC2101::get_scvPICflag() returns an array with True value for the stars belonging to the scvPIC
    # see chapter 3.1, page 16, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPICflag( self):
        return ( self.stellar_data[ 'PICmainSourceFlagBOL'] & 32 == 32)

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_tPICP1flag() returns an array with True value for the stars belonging to the P1 sample in the tPIC
    # this result includes stars in the P2 sample and hosting planets
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICP1flag( self):
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] & 1 == 1)

    #-----
    # PIC2101::get_tPICP2flag() returns an array with True value for the stars belonging to the P2 sample in the tPIC
    # this result includes stars hosting planets
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICP2flag( self):
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] & 2 == 2)

    #-----
    # PIC2101::get_tPICP4flag() returns an array with True value for the stars belonging to the P4 sample in the tPIC
    # this result includes stars hosting planets
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICP4flag( self):
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] & 8 == 8)

    #-----
    # PIC2101::get_tPICP5flag() returns an array with True value for the stars belonging to the P5 sample in the tPIC
    # this result includes stars hosting planets
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICP5flag( self):
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] & 4 == 4)

    #-----
    # PIC2101::get_tPICplanetflag() returns an array with True value for the stars hosting planets
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_tPICplanetflag( self):
        return ( self.stellar_data[ 'tPICsourceFlagNCAM_BOL'] & 16 == 16)

    #--------------------------------------------------------------------

    #-----
    # PIC2101::get_fgPICbflag() returns an array with True value for the stars in the fgPIC observed by F-CAM blue
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_fgPICbflag( self):
        return ( self.stellar_data[ 'fgPICsourceFlag'] & 1 == 1)

    #-----
    # PIC2101::get_fgPICrflag() returns an array with True value for the stars in the fgPIC observed by F-CAM red
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_fgPICrflag( self):
        return ( self.stellar_data[ 'fgPICsourceFlag'] & 2 == 2)

    #--------------------------------------------------------------------

    #-----
    # PIC2101::get_cPICR1FCAMflag() returns an array with True value for the stars belonging to the R1 sample (attitude and IGM) in the cPIC observed with the F-CAMs
    # this result includes stars that also belong to R2, R3, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR1FCAMflag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 1 == 1)
    
    #-----
    # PIC2101::get_cPICR2FCAMflag() returns an array with True value for the stars belonging to the R2 sample (microscanning) in the cPIC observed with the F-CAMs
    # this result includes stars that also belong to R1, R3, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR2FCAMflag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 2 == 2)

    #-----
    # PIC2101::get_cPICR3FCAMflag() returns an array with True value for the stars belonging to the R3 sample (best focus) in the cPIC observed with the F-CAMs
    # this result includes stars that also belong to R1, R2, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR3FCAMflag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 4 == 4)

    #-----
    # PIC2101::get_cPICR4FCAMflag() returns an array with True value for the stars belonging to the R4 sample (throughput) in the cPIC observed with the F-CAMs
    # this result includes stars that also belong to R1, R2, and R3 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR4FCAMflag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 8 == 8)

    #-----
    # PIC2101::get_cPICR5FCAMflag() returns an array with True value for the stars belonging to the R5 sample (outlier rejection) in the cPIC observed with the F-CAMs
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR5FCAMflag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 16 == 16)

    #-----
    # PIC2101::get_cPICR1flag() returns an array with True value for the stars belonging to the R1 sample (attitude and IGM) in the cPIC observed with the N-CAMs
    # this result includes stars that also belong to R2, R3, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR1flag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 32 == 32)
    
    #-----
    # PIC2101::get_cPICR2flag() returns an array with True value for the stars belonging to the R2 sample (microscanning) in the cPIC observed with the N-CAMs
    # this result includes stars that also belong to R1, R3, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR2flag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 64 == 64)

    #-----
    # PIC2101::get_cPICR3flag() returns an array with True value for the stars belonging to the R3 sample (best focus) in the cPIC observed with the N-CAMs
    # this result includes stars that also belong to R1, R2, and R4 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR3flag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 128 == 128)

    #-----
    # PIC2101::get_cPICR4flag() returns an array with True value for the stars belonging to the R4 sample (throughput) in the cPIC observed with the N-CAMs
    # this result includes stars that also belong to R1, R2, and R3 samples
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR4flag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 256 == 256)

    #-----
    # PIC2101::get_cPICR5flag() returns an array with True value for the stars belonging to the R5 sample (outlier rejection) in the cPIC observed with the N-CAMs
    # see chapter 3.1, page 17, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_cPICR5flag( self):
        return ( self.stellar_data[ 'cPICsourceFlag'] & 512 == 512)

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101::get_scvPIC1aflag() returns an array with True value for the stars belonging to the SCV1a sample in the scvPIC (eclipsing binaries)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC1aflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 1 == 1)

    #-----
    # PIC2101::get_scvPIC1bflag() returns an array with True value for the stars belonging to the SCV1b sample in the scvPIC (astrometric binaries)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC1bflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 2 == 2)

    #-----
    # PIC2101::get_scvPIC1cflag() returns an array with True value for the stars belonging to the SCV1c sample in the scvPIC (wide binaries)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC1cflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 4 == 4)

    #-----
    # PIC2101::get_scvPIC1dflag() returns an array with True value for the stars belonging to the SCV1d sample in the scvPIC (HW Vir-type binaries)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC1dflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 8 == 8)

    #-----
    # PIC2101::get_scvPIC1eflag() returns an array with True value for the stars belonging to the SCV1e sample in the scvPIC (wide white dwarf binaries)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC1eflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 16 == 16)

    #-----
    # PIC2101::get_scvPIC2aflag() returns an array with True value for the stars belonging to the SCV2a sample in the scvPIC (legacy and benchmark stars)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC2aflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 32 == 32)

    #-----
    # PIC2101::get_scvPIC2bflag() returns an array with True value for the stars belonging to the SCV2b sample in the scvPIC (legacy and benchmark stars)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC2bflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 64 == 64)

    #-----
    # PIC2101::get_scvPIC3aflag() returns an array with True value for the stars belonging to the SCV3a sample in the scvPIC (photometrically stable stars)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC3aflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 128 == 128)

    #-----
    # PIC2101::get_scvPIC3bflag() returns an array with True value for the stars belonging to the SCV3b sample in the scvPIC (photometrically stable stars)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC3bflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 256 == 256)

    #-----
    # PIC2101::get_scvPIC4aflag() returns an array with True value for the stars belonging to the SCV4a sample in the scvPIC (solar-like pulsators)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC4aflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 512 == 512)

    #-----
    # PIC2101::get_scvPIC4bflag() returns an array with True value for the stars belonging to the SCV4b sample in the scvPIC (solar-like pulsators)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC4bflag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 1024 == 1024)

    #-----
    # PIC2101::get_scvPIC5flag() returns an array with True value for the stars belonging to the SCV5 sample in the scvPIC (gamma Dor stars)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC5flag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 2048 == 2048)

    #-----
    # PIC2101::get_scvPIC6flag() returns an array with True value for the stars belonging to the SCV6 sample in the scvPIC (transiting exoplanets)
    # see chapter 3.1, page 18, in PLATO-SSDC-PDC-DD-0003 for the bitmask
    # see chapter 5, page 7, in PLATO-SSDC-PDC-DN-0001 for the numbers
    def get_scvPIC6flag( self):
        return ( self.stellar_data[ 'scvPICsourceFlag'] & 4096 == 4096)

    #--------------------------------------------------------------------
    
    #-----
    # PIC2101:get_valid_id() returns the stellar id of the valid entries in the catalog.
    def get_valid_id( self):
        return self.get_id()[ self.valid]

    #-----
    # PIC2101::get_valid_ra() returns right ascension of the valid entries in the catalog.
    def get_valid_ra( self):
        return self.get_ra()[ self.valid]

    #-----
    # PIC2101::get_valid_de() returns the declination of the valid entries in the catalog.
    def get_valid_de( self):
        return self.get_de()[ self.valid]

    #-----
    # PIC2101::get_valid_vmag() returns the V mag of the valid entries in the catalog.
    def get_valid_vmag( self):
        return self.get_vmag()[ self.valid]

    #-----
    # PIC2101::get_valid_gmag() returns the G mag of the valid entries in the catalog.
    def get_valid_gmag( self):
        return self.get_gmag()[ self.valid]

    #-----
    # PIC2101::get_valid_pmag() returns the P mag (N-CAM) of the valid entries in the catalog.
    def get_valid_pmag( self):
        return self.get_pmag()[ self.valid]
    
    #-----
    # PIC2101::get_valid_pmagblue() returns the P mag F-CAM blue of the valid entries in the catalog.
    def get_valid_pmagblue( self):
        return self.get_pmagblue()[ self.valid]

    #-----
    # PIC2101::get_valid_pmagred() returns the P mag F-CAM red of the valid entries in the catalog.
    def get_valid_pmagred( self):
        return self.get_pmagred()[ self.valid]

    #-----
    # PIC2101::get_valid_rs() returns the stellar radii of the valid entries in the catalog.
    def get_valid_rs( self):
        return self.get_rs()[ self.valid]

    #-----
    # PIC2101::get_valid_urs() returns the stellar radius uncertainties of the valid entries in the catalog.
    def get_valid_urs( self):
        return self.get_urs()[ self.valid]

    #-----
    # PIC2101::get_valid_ms() returns the stellar masses of the valid entries in the catalog.
    def get_valid_ms( self):
        return self.get_ms()[ self.valid]

    #-----
    # PIC2101::get_valid_ums() returns the stellar mass uncertainties of the valid entries in the catalog.
    def get_valid_ums( self):
        return self.get_ums()[ self.valid]

    #-----
    # PIC2101::get_valid_ts() returns the stellar effective temperature of the valid entries in the catalog.
    def get_valid_ts( self):
        return self.get_ts()[ self.valid]

    #-----
    # PIC2101::get_valid_uts() returns the stellar effective temperature uncertainties of the valid entries in the catalog.
    def get_valid_uts( self):
        return self.get_uts()[ self.valid]

    #-----
    # PIC2101::get_valid_logg() returns the stellar logg of the valid entries in the catalog.
    def get_valid_logg( self):
        return self.get_logg()[ self.valid]

    #-----
    # PIC2101::get_valid_plx() returns the stellar parallax of the valid entries in the catalog.
    def get_valid_plx( self):
        return self.get_plx()[ self.valid]

    #-----
    # PIC2101::get_valid_uplx() returns the stellar parallax uncertainties of the valid entries in the catalog.
    def get_valid_uplx( self):
        return self.get_uplx()[ self.valid]

    #-----
    # PIC2101::get_valid_random() returns the random noise value of the valid entries in the catalog.
    def get_valid_random( self):
        return self.get_random()[ self.valid]

    #-----
    # PIC2101::get_valid_total() returns the total noise value of the valid entries in the catalog.
    def get_valid_total( self):
        return self.get_total()[ self.valid]

    #-----
    # PIC2101::get_valid_randombol() returns the random noise value BOL of the valid entries in the catalog.
    def get_valid_randombol( self):
        return self.get_randombol()[ self.valid]

    #-----
    # PIC2101::get_valid_totalbol() returns the total noise value BOL of the valid entries in the catalog.
    def get_valid_totalbol( self):
        return self.get_totalbol()[ self.valid]

    #-----
    # PIC2101::get_valid_ncamsbol() returns the number of cameras observing the star BOL of the valid matches.
    def get_valid_ncamsbol( self):
        return self.get_ncamsbol()[ self.valid]

    #-----
    # PIC2101::get_valid_ncamseol() returns the number of cameras observing the star EOL of the valid matches.
    def get_valid_ncamseol( self):
        return self.get_ncamseol()[ self.valid]

    #-----
    # PIC2101::get_valid_simulated_rs() returns the simulated stellar radii for the valid entries in the catalog.
    def get_valid_simulated_rs( self):
        return self.get_simulated_rs()[ self.valid]

    #-----
    # PIC2101::get_valid_simulated_ms() returns the simulated stellar masses for the valid entries in the catalog.
    def get_valid_simulated_ms( self):
        return self.get_simulated_ms()[ self.valid]

# dtype=(numpy.record, 
# [('PICid', '>i8'), 
# ('PICname', 'S17'), 
# ('StarName', 'S28'), 
# ('RAdeg', '>f8'), 
# ('eRAdeg', '>f8'), 
# ('DEdeg', '>f8'), 
# ('eDEdeg', '>f8'), 
# ('pmRA', '>f8'), 
# ('epmRA', '>f4'), 
# ('pmDE', '>f8'), 
# ('epmDE', '>f4'), 
# ('pm', '>f8'), 
# ('epm', '>f8'), 
# ('Plx', '>f8'), 
# ('ePlx', '>f4'), 
# ('distance', '>f8'), 
# ('edistance', '>f8'), 
# ('Glon', '>f8'), 
# ('Glat', '>f8'), 
# ('posEpoch', '>f4'), 
# ('refEpoch', '>f4'), 
# ('posPropFlag', '>i2'), 
# ('PlatoMagNCAM', '>f8'), 
# ('ePlatoMagNCAM', '>f8'), 
# ('PlatoMagFCAMb', '>f8'), 
# ('ePlatoMagFCAMb', '>f8'), 
# ('PlatoMagFCAMr', '>f8'), 
# ('ePlatoMagFCAMr', '>f8'), 
# ('Gmag', '>f8'), 
# ('eGmag', '>f8'), 
# ('BPmag', '>f8'), 
# ('eBPmag', '>f8'), 
# ('RPmag', '>f8'), 
# ('eRPmag', '>f8'), 
# ('Hpmag', '>f4'), 
# ('eHpmag', '>f4'), 
# ('Ksmag', '>f4'), 
# ('eKsmag', '>f4'), 
# ('AG', '>f8'), 
# ('eAG', '>f8'), 
# ('AKs', '>f8'), 
# ('eAKs', '>f8'), 
# ('EBPRP', '>f8'), 
# ('eEBPRP', '>f8'), 
# ('extStatus', '>i2'), 
# ('VmagCalculated', '>f4'), 
# ('eVmagCalculated', '>f4'), 
# ('Teff', '>f8'), 
# ('eTeff', '>f8'), 
# ('Radius', '>f8'), 
# ('eRadius', '>f8'), 
# ('Mass', '>f8'), 
# ('eMass', '>f8'), 
# ('caseFlag', '>i2'), 
# ('PICmainSourceFlagBOL', '>i2'), 
# ('tPICsourceFlagNCAM_BOL', '>i2'), 
# ('fgPICsourceFlag', '>i2'), 
# ('cPICsourceFlag', '>i2'), 
# ('scvPICsourceFlag', '>i2'), 
# ('NSSFlag', '>i2'), 
# ('qualityFlag', '>i2'), 
# ('tPICplanetFlag', '>i2'), 
# ('fgPICcPICvariabilityFlag', '>i2'), 
# ('targetStatusFlag', '>i2'), 
# ('BOLrandomSysNSRNCAM_T', '>f4'), 
# ('BOLnCameraObsNCAM_T', '>i2'), 
# ('BOLnCameraSatNCAM_T', '>i2'), 
# ('tPICscientificRanking', '>f4'), 
# ('scvPICscientificRanking', '>i2'), 
# ('scientificPriority', '>i2'), 
# ('scheduledTarget', '>i2')]))

# dtype=(numpy.record, [('PICid', '>i8'), 
# ('PICname', 'S17'), 
# ('StarName', 'S28'), 
# ('BOLrandomNSRNCAM_T', '>f4'), 
# ('BOLrandomSysNSRNCAM_T', '>f4'), 
# ('BOLnCameraObsNCAM_T', '>i2'), 
# ('BOLnCameraSatNCAM_T', '>i2'), 
# ('EOL24randomNSRNCAM_R', '>f4'), 
# ('EOL24randomSysNSRNCAM_R', '>f4'), 
# ('EOL24nCameraObsNCAM_R', '>i2'), 
# ('EOL24nCameraSatNCAM_R', '>i2'), 
# ('EOLrandomNSRNCAM_R', '>f4'), 
# ('EOLrandomSysNSRNCAM_R', '>f4'), 
# ('EOLnCameraObsNCAM_R', '>i2'), 
# ('EOLnCameraSatNCAM_R', '>i2'), 
# ('BOL1randomNSRNCAM_T', '>f4'), 
# ('BOL1randomSysNSRNCAM_T', '>f4'), 
# ('BOL1nCameraObsNCAM_T', '>i2'), 
# ('BOL1nCameraSatNCAM_T', '>i2'), 
# ('EOL1randomNSRNCAM_R', '>f4'), 
# ('EOL1randomSysNSRNCAM_R', '>f4'), 
# ('EOL1nCameraObsNCAM_R', '>i2'), 
# ('EOL1nCameraSatNCAM_R', '>i2'), 
# ('BOLrandomNSRFCAMB_T', '>f4'), 
# ('BOLrandomSysNSRFCAMB_T', '>f4'), 
# ('BOLnCameraObsFCAMB_T', '>i2'), 
# ('BOLnCameraSatFCAMB_T', '>i2'), 
# ('EOLrandomNSRFCAMB_R', '>f4'), 
# ('EOLrandomSysNSRFCAMB_R', '>f4'), 
# ('EOLnCameraObsFCAMB_R', '>i2'), 
# ('EOLnCameraSatFCAMB_R', '>i2'), 
# ('BOLrandomNSRFCAMR_T', '>f4'), 
# ('BOLrandomSysNSRFCAMR_T', '>f4'), 
# ('BOLnCameraObsFCAMR_T', '>i2'), 
# ('BOLnCameraSatFCAMR_T', '>i2'), 
# ('EOLrandomNSRFCAMR_R', '>f4'), 
# ('EOLrandomSysNSRFCAMR_R', '>f4'), 
# ('EOLnCameraObsFCAMR_R', '>i2'), 
# ('EOLnCameraSatFCAMR_R', '>i2')]))

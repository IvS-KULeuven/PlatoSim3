#!/usr/bin/env python3

"""
Python module containing astro-query functions used by picsim.py.

NOTE This class needs the Poetry install: 
     >> poetry install --with platonium 
"""

# Built-in
import os
import sys
import glob
import math
import time
import inspect
import pathlib
import http.client as httplib
import urllib.parse as urllib
from xml.dom.minidom import parseString
from tqdm import tqdm

# PlatoSim standard
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io.votable  import parse
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import MaxNLocator
from colorama import Fore, Style
from prettytable import PrettyTable
from scipy.ndimage import median_filter

# PLATOnium extra
from astroquery.simbad   import Simbad
from astroquery.mast     import Catalogs
from astroquery.gaia     import Gaia

# PlatoSim functions
import platosim.utilities as ut

#--------------------------------------------------------------#
#                      HIDDEN FUNCTIONS                        #
#--------------------------------------------------------------#


def _fetch_gaia_columns(flag_stellar, flag_variable, flag_quasar):

    """Function to fecth columns from Gaia database.
    """

    cols = {
        'default':['gaia.source_id',
                   'gaia.ra',
                   'gaia.dec',
                   'gaia.l',
                   'gaia.b',               
                   'gaia.phot_g_mean_mag',
                   'gaia.bp_rp',
                   'gaia.ag_gspphot',
                   'gaia.parallax', 'gaia.parallax_error',
                   'gaia.pm',
                   'gaia.pmra', 'gaia.pmra_error',
                   'gaia.pmdec', 'gaia.pmdec_error',
                   'gaia.ruwe'],
        'stellar':['gaia.mh_gspphot',    'gaia.mh_gspphot_lower',    'gaia.mh_gspphot_upper',
                   'gaia.logg_gspphot',  'gaia.logg_gspphot_lower',  'gaia.logg_gspphot_upper',
                   'gaia.teff_gspphot',  'gaia.teff_gspphot_lower',  'gaia.teff_gspphot_upper',
                   'astro.radius_flame', 'astro.radius_flame_lower', 'astro.radius_flame_upper',
                   'astro.mass_flame',   'astro.mass_flame_lower',   'astro.mass_flame_upper',
                   'astro.lum_flame',    'astro.lum_flame_lower',    'astro.lum_flame_upper',
                   'astro.spectraltype_esphs',
                   'astro.evolstage_flame'],
        'variable':['gaia.phot_variable_flag',
                    'astro.classlabel_espels',
                    'astro.activityindex_espcs', 'astro.activityindex_espcs_uncertainty'],
        'quasar':['quasar.vari_best_class_name',
                  'astro.classprob_dsc_combmod_quasar',
                  'quasar.vari_agn_membership_score',
                  'quasar.qso_variability',
                  'quasar.non_qso_variability',
                  'quasar.host_galaxy_detected',
                  'quasar.redshift_qsoc',
                  'quasar.redshift_qsoc_lower',
                  'quasar.redshift_qsoc_upper']
    }
    
    # Fetch global name
    columns = cols['default']
    if flag_stellar:
        c = cols['stellar']
        for i in c: columns.append(i)
    if flag_variable:
        c = cols['variable']
        for i in c: columns.append(i)
    if flag_quasar:
        c = cols['quasar']
        for i in c: columns.append(i)
        
    return ','.join(columns)



def _submit_gaia_query(query, ofile=False):

    """Main function query data from Gaia database.
    """

    # We use the urllib to keep the connection open because
    # the sky regions are huge which fails with Gaia.lunch_job
    params = urllib.urlencode({"REQUEST"        : "doQuery",
                               "LANG"           : "ADQL",
                               "FORMAT"         : "votable_plain",
                               "PHASE"          : "RUN",
                               "JOBNAME"        : "PLATO catalog",
                               "JOBDESCRIPTION" : "None", 
                               "QUERY"          : query})
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept"      : "text/plain"}

    # Information about server
    host = "gea.esac.esa.int"
    port = 443
    pathinfo = "/tap-server/tap/async"
    connection = httplib.HTTPSConnection(host, port)
    connection.request("POST", pathinfo, params, headers)

    # Get status    
    response = connection.getresponse()
    location = response.getheader("location")
    jobid    = location[location.rfind('/')+1:]
    connection.close()
    
    # Check job status, wait until finished

    while True:
        connection = httplib.HTTPSConnection(host, port)
        connection.request("GET",pathinfo+"/"+jobid)
        response = connection.getresponse()
        data = response.read()

        # XML response: parse it to obtain the current status
        # (you may use pathinfo/jobid/phase entry point to avoid XML parsing)
        dom = parseString(data)
        phaseElement = dom.getElementsByTagName('uws:phase')[0]
        phaseValueElement = phaseElement.firstChild
        phase = phaseValueElement.toxml()

        # Check finished
        if phase == 'COMPLETED':
            break
        
        # Wait and repeat
        time.sleep(0.2)

    connection.close()

    # Get results    
    connection = httplib.HTTPSConnection(host, port)
    connection.request("GET",pathinfo+"/"+jobid+"/results/result")
    response = connection.getresponse()
    data = response.read().decode('iso-8859-1')
    
    # Write output file
    if not ofile:
        ofile = 'starcatGaiaDR3.vot'
    outputFile = open(ofile, 'w')
    outputFile.write(data)
    outputFile.close()
    connection.close()

    # Load output file into a pandas df
    votable = parse(ofile)
    df = ut.votable2pandas(votable)
    os.remove(ofile)
    
    return df



def _rename_columns(df, flag_stellar, flag_variable, flag_quasar):

    # Rename columns
    if 'source_id' in df:
        df = df.rename(columns={'source_id': 'source_gaia_dr3'})
    elif 'SOURCE_ID' in df:
        df = df.rename(columns={'SOURCE_ID': 'source_gaia_dr3'})
    df = df.rename(columns={
        'phot_g_mean_mag': 'Gmag',
        'bp_rp': 'BP_RP',
        'ag_gspphot': 'Ag',
        'parallax': 'plx',
        'parallax_error': 'plx_err'})

    # Convert source ids to integers
    df.source_gaia_dr3 = df.source_gaia_dr3.astype(np.int64)
    
    # Change base on columns flag
   
    if flag_stellar:
        df = df.rename(columns={
            'mh_gspphot': 'mh',
            'mh_gspphot_lower': 'mh_low',
            'mh_gspphot_upper': 'mh_upp',
            'logg_gspphot': 'logg',
            'logg_gspphot_lower': 'logg_low',
            'logg_gspphot_upper': 'logg_upp',
            'teff_gspphot': 'Teff',
            'teff_gspphot_lower': 'Teff_low',
            'teff_gspphot_upper': 'Teff_upp',
            'radius_flame': 'R',
            'radius_flame_lower': 'R_low',
            'radius_flame_upper': 'R_upp',
            'mass_flame': 'M',
            'mass_flame_lower': 'M_low',
            'mass_flame_upper': 'M_upp',
            'lum_flame': 'L',
            'lum_flame_lower': 'L_low',
            'lum_flame_upper': 'L_upp',
            'spectraltype_esphs': 'spec',
            'evolstage_flame': 'evol',
        })
    
        # Round Teff column
        df = df.fillna(-1)
        df = df.astype({'Teff':int, 'Teff_low':int, 'Teff_upp':int})
        df = df.replace({-1:np.nan})

    if flag_variable:
        df = df.rename(columns={
            'phot_variable_flag': 'variable',
            'classlabel_espels': 'class',
            'activityindex_espcs': 'sindex',
            'activityindex_espcs_uncertainty': 'sindex_err',
        })
        
    if flag_quasar:        
        df = df.rename(columns={
            'vari_best_class_name': 'class_name',
            'classprob_dsc_combmod_quasar': 'p_quasar',
            'vari_agn_membership_score': 'agn_score',
            'qso_variability': 'qso_var',
            'non_qso_variability': 'qso_non',
            'host_galaxy_detected': 'host_galaxy',
            'redshift_qsoc': 'z',
            'redshift_qsoc_lower': 'z_lower',
            'redshift_qsoc_upper': 'z_upper',            
        })
        
        # Replace upper/lower error with redshift uncertainty
        dex = df.columns.get_loc('z') + 1
        df.insert(dex, 'z_err', np.abs(df.z_upper - df.z_lower) / df.z)
        df.drop(['z_lower', 'z_upper'], axis=1, inplace=True)
        
    return df

        
#--------------------------------------------------------------#
#                          FUNCTIONS                           #
#--------------------------------------------------------------#


def ticQuery(star, radius=2, Vmax=18, outFile=None):

    """Query TIC catalog for stars around a given named source below a givenV.

    Parameters
    ----------
    star : str
        Name of the star to query around.
    radius : float
        Radius in arcmin to query around the star.
    Vmax : float
        Maximum V magnitude to query for.
    outFile : str
        Path of the output file to write to. If None, no file is written.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing the results of the query. The named star will appear first if
        it is not removed by the Vmax cut.
    """


    # Get the coordinates of the star from Simbad
    
    result_table = Simbad.query_object(star)
    if result_table is None:
        raise ValueError(f"Could not find {star} in Simbad.")
    ra = result_table["RA"][0]
    dec = result_table["DEC"][0]
    coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

    # Query TIC for stars around the star, within the given radius
    
    results = Catalogs.query_region(coords, radius=radius * u.arcmin, catalog="TIC")
    if results is None:
        raise ValueError(f"Could not find any stars in TIC around {star}.")

    # Convert the results to a Pandas DataFrame
    
    results = results.to_pandas()
    results = results[results["Vmag"] < Vmax][["ra", "dec", "Vmag"]]

    # Optionally write the results to a txt file
    
    if outFile is not None:
        with open(outFile, "w") as f:
            f.write("# RA DEC Vmag\n")
            for i, row in results.iterrows():
                f.write(f"{row['ra']:.6f} {row['dec']:.6f} {row['Vmag']:.3f}\n")

    # That's it!
                
    return results





def gaiaQuery(star):

    """Query Gaia for a named star and return the Gaia DR2 ID.

    Parameters
    ----------
    star : str
        Name of the star to query around.

    Returns
    -------
    gaia_id : int
        The Gaia DR2 ID of the star.
    """

    # Query for target star
    result_table = Simbad.query_objectids(star)
    if result_table is None:
        raise LookupError(f"No Simbad results for {star} (probably not a star)")

    # If requested find all stars with 'radius'
    for row in result_table:
        if 'Gaia DR2' in row['ID']:
            gaia_id = row['ID']
            return int(gaia_id[9:])
    raise LookupError(f"No Gaia DR2 ID for {star} (probably a multiple star)")


def simbadQuery(star, radius=60, maglim=21):

    """Query Gaia for a named star and return the Gaia DR2 ID.

    Parameters
    ----------
    star : str
        Name of the star to query around.

    Returns
    -------
    gaia_id : int
        The Gaia DR2 ID of the star.
    """

    # Qucik check that target star exist
    result_table = Simbad.query_objectids(star)
    if result_table is None:
        raise LookupError(f"No Simbad results for {star} (probably not a star)")

    # Fetch the equatorial coordinates
    Simbad.reset_votable_fields()
    Simbad.remove_votable_fields('coordinates')
    Simbad.add_votable_fields('ra(:;A;ICRS;J2000)', 'dec(:;D;ICRS;2000)')
    table = Simbad.query_object(star, wildcard=False)
    coord = SkyCoord(ra=['{}h{}m{}s'.format(*ra.split(':')) for ra in table['RA___A_ICRS_J2000']], 
                     dec=['{}d{}m{}s'.format(*dec.split(':')) for dec in table['DEC___D_ICRS_2000']],
                     frame='icrs', equinox='J2000')
    raStar  = coord.ra.degree[0]
    decStar = coord.dec.degree[0]

    # Convert radius to from arcsec to deg
    radius /= 3600.

    query_cone = f"""SELECT 
    DISTANCE( POINT({raStar},{decStar}), POINT(ra,dec) )
    AS dis, source_id, ra, dec,
    phot_g_mean_mag, bp_rp,
    parallax, parallax_error,
    pmra, pmdec, ruwe,
    teff_gspphot, logg_gspphot
    FROM gaiadr3.gaia_source AS cat
    WHERE 1=CONTAINS(POINT({raStar}, {decStar}),
    CIRCLE(cat.ra, cat.dec, {radius}))
    AND cat.phot_g_mean_mag < {maglim} 
    ORDER BY dis ASC
    """

    # Launch Gaia query 

    job     = Gaia.launch_job(query_cone)
    results = job.get_results()

    # Convert astropy results table into pandas df

    df = results.to_pandas()

    # Rename columns

    df = df.rename(columns={'SOURCE_ID': 'gaiaDR3',
                            'phot_g_mean_mag': 'Gmag',
                            'bp_rp': 'BP_RP',
                            'parallax': 'plx',
                            'parallax_error': 'plxe',
                            'teff_gspphot': 'teff',
                            'logg_gspphot': 'logg'})

    
    # Make sure that target is the first entry    
    # for row in result_table:
    #     if 'source_id' in row['ID']:
    #         gaia_id = int(row['ID'][9:])            
    # row = df.index[df['source_id'] == gaia_id].tolist()
    # dex = row + [i for i in range(len(df)) if i != row[0]]
    # df = df.iloc[dex].reset_index(drop=True)

    # Convert Gmag to Pmag

    df['Pmag'] = ut.passbandConversionG2P(df.Gmag, df.BP_RP)
    
    # Relocate distance column [arcsec]
    
    df.dis = (df.dis-df.dis.iloc[0]) * 3600.

    # Sort and return
    
    return df.sort_values(by=['dis'])





def gaiaQueryCone(ra, dec, radius=1,
                  mag_min=0, mag_max=21,
                  flag_stellar=False,
                  flag_variable=False,
                  flag_quasar=False):

    """Query sky cone region using Gaia DR3.

    Parameters
    ----------
    star : str
        Name of the star to query around.

    Returns
    -------
    gaia_id : int
        The Gaia DR2 ID of the star.
    """

    # Fetch requested Gaia columns
    columns = _fetch_gaia_columns(flag_stellar, flag_variable, flag_quasar)

    # Construct query
    query = f"""SELECT TOP 100000000
    DISTANCE(POINT(ra, dec), POINT({ra}, {dec}))
    AS dis,{columns}
    FROM gaiadr3.gaia_source AS gaia
    JOIN gaiadr3.astrophysical_parameters AS astro
      ON gaia.source_id = astro.source_id
    WHERE 1=CONTAINS(
      POINT(gaia.ra, gaia.dec),
      CIRCLE({ra}, {dec}, {radius}))
    AND gaia.phot_g_mean_mag < {mag_max}
    """
    
    # Launch Gaia query
    job = None
    counter = 0
    counter_max = 10
    while job is None:
        try:
            job = Gaia.launch_job(query)
        except KeyboardInterrupt:
            ut.errorcode('error', 'User stopped script..')
        except:
            ut.errorcode('warning', 'Query failed due to a socket.gaierror: [Errno -3] ' +
                         'Temporary failure in name resolution.. Retrying in 10s ...')
            counter += 1
            if counter == counter_max:
                ut.errorcode('error', f'Query failed after {counter_max} attempts..')
            time.sleep(10)

    results = job.get_results()
    
    # Convert astropy results table into pandas df
    df = results.to_pandas()

    # Adjust data frame
    df = _rename_columns(df, flag_stellar, flag_variable, flag_quasar)

    # Relocate distance column [arcsec]
    df = df.sort_values(by=['dis'])
    df.dis = np.abs(df.dis-df.dis.iloc[0]) * 3600.

    # Reset index and return
    return df.reset_index(drop=True)





def gaiaQueryID(source_id, ra, dec, radius=0.01,
                flag_stellar=False, flag_variable=False, flag_quasar=False,
                ofile=False):

    """Function to query a target using it's Gaia DR3 ID.
    
    Parameters
    ----------
    source_id : int64, ndarray
=        Gaia DR3 source ID(s)
    ofile : str
        File name (without file extension) to be saved

    Return
    ------
    Data frame with information c.f. the columns of 'colname'.

    TODO update function to be dependent only on IDs!
    """

    # Fetch requested Gaia columns
    columns = _fetch_gaia_columns(flag_stellar, flag_variable, flag_quasar)
    
    # Construct query cone
    if flag_quasar:
        query = f"""SELECT
        {columns}
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.astrophysical_parameters AS astro
          ON gaia.source_id = astro.source_id
        JOIN gaiadr3.qso_candidates AS quasar
          ON gaia.source_id = quasar.source_id
        WHERE 1=CONTAINS(
          POINT(gaia.ra, gaia.dec),
          CIRCLE({ra}, {dec}, {radius}))
        """
    else:
        query = f"""SELECT
        {columns}
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.astrophysical_parameters AS astro
          ON gaia.source_id = astro.source_id
        WHERE 1=CONTAINS(
          POINT(gaia.ra, gaia.dec),
          CIRCLE({ra}, {dec}, {radius}))
        """
        
    # Submit job
    df = _submit_gaia_query(query, ofile)

    # Adjust data frame
    df = _rename_columns(df, flag_stellar, flag_variable, flag_quasar)

    # Order after magnitude
    df = df.sort_values(by=['Gmag'])
    
    # Reset index and return
    return df.reset_index(drop=True)



def gaiaQueryRegion(ra, dec, radius=1,
                    mag_min=0, mag_max=17,
                    flag_stellar=False, flag_variable=False, flag_quasar=False,
                    ofile=False):

    """Function to query a circular sky region from Gaia DR3.
    
    Parameters
    ----------
    ra : float, ndarray
        Right ascension of central point for query [deg]
    dec : float, ndarray
        Declination of central point for query [deg]
    radius : int, float
        Angular radius to search for stars within [deg]
    maglim_min : int, float
        Lower G-mean magitude limit for search.
    maglim_max : int, float
        Upper G-mean magitude limit for search.
    flag_stellar : bool
        Flag to include specific columns for stars.
    flag_variable : bool
        Flag to include specific columns for variability.
    flag_quasar : bool
        Flag to include specific columns for quasars.
    ofile : str
        File name (without file extension) to be saved

    Return
    ------
    Feather file given by the file name destination.

    Notes
    -----
    - Columns can be found at: https://gea.esac.esa.int/archive/
      See: "Search" -> "Advanced (ADQL)" -> "Gaia Data Release 3"
    - Query only valid for Gmag < 17, else gaps are intoduced.
      Use additional queries e.g. in bins of 0.5 mag for Gmag > 17.
    """

    # Fetch requested Gaia columns
    columns = _fetch_gaia_columns(flag_stellar, flag_variable, flag_quasar)

    
    if flag_quasar:
        query = f"""SELECT TOP 100000000
        {columns}
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.astrophysical_parameters AS astro
          ON gaia.source_id = astro.source_id
        JOIN gaiadr3.qso_candidates AS quasar
          ON gaia.source_id = quasar.source_id
        WHERE 1=CONTAINS(
          POINT(gaia.ra, gaia.dec),
          CIRCLE({ra}, {dec}, {radius}))
          AND gaia.phot_g_mean_mag > {mag_min}
          AND gaia.phot_g_mean_mag < {mag_max}
          AND astro.classprob_dsc_combmod_quasar > 0.99
        """ # AND quasar.vari_best_class_name = AGN
    else:
        query = f"""SELECT TOP 100000000
        {columns}
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.astrophysical_parameters AS astro
          ON gaia.source_id = astro.source_id
        WHERE 1=CONTAINS(
          POINT(gaia.ra, gaia.dec),
          CIRCLE({ra}, {dec}, {radius}))
          AND gaia.phot_g_mean_mag > {mag_min}
          AND gaia.phot_g_mean_mag < {mag_max}
        """

    # # Launch Gaia query
    # job = None
    # counter = 0
    # counter_max = 10
    # while job is None:
    #     try:
    #         job = Gaia.launch_job(query)
    #     except KeyboardInterrupt:
    #         ut.errorcode('error', 'User stopped script..')
    #     except:
    #         ut.errorcode('warning', 'Query failed due network error to Gaia database: Retrying in 10s..')
    #         counter += 1
    #         if counter == counter_max:
    #             ut.errorcode('error', f'Query failed after {counter_max} attempts..')
    #         time.sleep(10)
    
    # # Convert astropy results table into pandas df
    # results = job.get_results()
    # df = results.to_pandas()
        
    # Submit job
    df = None
    counter = 0
    counter_max = 10
    while df is None:
        try:
            df = _submit_gaia_query(query, ofile)
        except KeyboardInterrupt:
            ut.errorcode('error', 'User stopped script..')
        except:
            ut.errorcode('warning', 'Query failed due network error to Gaia database: Retrying in 10s..')
            counter += 1
            if counter == counter_max:
                ut.errorcode('error', f'Query failed after {counter_max} attempts..')
            time.sleep(10)
        
    # Adjust data frame
    df = _rename_columns(df, flag_stellar, flag_variable, flag_quasar)

    # Reset index and return
    return df.reset_index(drop=True)



def fetch_gaia_info(df, NED=False):
    """Fetch Gaia info for each source in data frame.
    Use NASA/IPAC Extragalactic Database (NED).
    """
    for i in tqdm(range(df.shape[0]), bar_format=ut.tqdmBar()):
        di = df.reset_index(drop=True).loc[i]
        if NED:
            ra, dec = di.RA, di.Dec
        else:
            ra, dec = di.ra, di.dec
        dx = gaiaQueryCone(ra, dec, radius=0.01, mag_max=21)
        if dx.shape[0] > 1:
            dx = dx[dx.dis == 0]
        if i == 0:
            dq = dx
        else:
            dq = pd.concat((dq, dx))
            
    # Alter data frame
    dq = dq.reset_index(drop=True)
    dq = dq.drop(columns='dis')
    if NED:
        dq.insert(0, 'source', df['Object Name'])
        dq['z'] = df['Redshift (z)']
        
    return dq

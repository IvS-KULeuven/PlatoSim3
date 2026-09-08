#!/usr/bin/env python3

# PLATO lightcurves
# J.J. Green
# $Id$


from __future__ import print_function
from builtins import str
from builtins import map
from builtins import range

# Python defaults
import re
import os
import sys
import math
import shutil
import getopt

# Other packages
import h5py
import yaml

import numpy as np
import pandas as pd
from astropy.io import fits as pf

import xml.dom.minidom as xdm
import matplotlib.pyplot as plt

import scipy.signal
import scipy.special

# L1 pipeline
import pislib3 as pis
from packaging.version import parse as parse_version
import spline2dbase
import libmask
import libcentroid

# PlatoSim
from platosim.simfile import SimFile
from platosim.utilities import errorcode

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value) 


def rotate2d (pos , angle) :
    return ( pos[0] * math.cos(angle) + pos[1] * math.sin(angle),
            -pos[0] * math.sin(angle) + pos[1] * math.cos(angle))

# convert to radians
def deg2rad(d) :
    return d * math.pi / 180

# convert to degrees
def rad2deg(r) :
    return r * 180 / math.pi

arc2rad = math.pi / 3600. / 180.  # arc second to radian



        
# sigma tablature
# Azimuth/rotation = [0:360,1]
# radius/alpha = [0:14*sqrt(2),sqrt(2)]
# magnitude = [8:16,0.5]
# Xccd/Yccd = [0:1,0.1]
def tab_sigma(table, az, rad, mag, Xc, Yc) :
    width = table['width'][:]
    azimuth = table['az'][:]
    angles = table['angles'][:]
    magnitude = table['mag'][:]
    dx = table['dx'][:]
    dy = table['dy'][:]

    
    if (az >= -90 and az <= 0.):
        iaz = np.argmin(np.abs(az - azimuth))
    else:
        print("Azimuth outside the range: ", az)
        return None

    # Added : little threshold to compare 2 float
    if (rad >= angles[0] and rad <= angles[-1] + 0.000001):
        irad = np.argmin(np.abs(rad - angles))
    else:
        print("Radius outside the range: ", rad)
        return None

    if (mag >= magnitude[0] and mag <= magnitude[-1]):
        imag = np.argmin(np.abs(mag - magnitude))
    else:
        print("magnitude outside the range: ", mag)
        return None

    if (Xc >= dx[0] and Xc <= dx[-1]):
        ix = np.argmin(np.abs(Xc - dx))
    else:
        # Case of only one value within table
        if (Xc >= -0.5 and Xc <= 0.5 and len(dx) == 1):
            # Everything seems to be OK, get the first value
            ix = 0
        else:
            print("Xccd outside the range :", Xc)
            return None

    if (Yc >= dy[0] and Yc <= dy[-1]):
        iy = np.argmin(np.abs(Yc - dy))
    else:
        # Case of only one value within table
        if (Yc >= -0.5 and Yc <= 0.5 and len(dy) == 1):
            # Everything seems to be OK, get the first value
            iy = 0
        else:
            print("Yccd outside the range: ", Yc)
            return None
    
    # the pixel corner is excluded
    if((math.fabs(dx[ix]) < 1e-5) and (math.fabs(dy[iy]) < 1e-5)):
        ix += 1
        iy += 1
 
    print(azimuth[iaz], angles[irad], magnitude[imag], dx[ix], dy[iy])
    Gsigma = width[iaz, irad, imag, ix, iy]

    return Gsigma





def clean_string(string):

    return re.sub("\s{1,}$", "", re.sub("^\s{1,}", "", string))



# extract a couple of items of interest from the PLATO
# xml configuration file -- these are 
# - the density of subpixels to pixels
# - the path of the psf

def in_dom(name, dom) :
    return dom.getElementsByTagName(name)




def from_dom(name, dom) :
    return dom.getElementsByTagName(name)[0].childNodes[0].data




def info_hdf5(idir, prefix):

    """This is a help function to fetch info from the HDF5
    """

    # Fetch output HDF5 file
    f = SimFile(f'{idir}/{prefix}.hdf5')

    # Subfield info [pixel]
    x0 = f.getInputParameter("SubField", "ZeroPointColumn")
    y0 = f.getInputParameter("SubField", "ZeroPointRow")
    nx = f.getInputParameter("SubField", "NumColumns")
    ny = f.getInputParameter("SubField", "NumRows")    
    xCCDc = x0 + nx/2.
    yCCDc = y0 + ny/2.
    pis.trace(f"Center of CCD subfield : ({xCCDc}, {yCCDc}) pixel")
    
    # Stellar catalogue
    ID, RA, dec, mag, xFPmm, yFPmm, xCCD, yCCD = f.getStarCatalog()

    # Focal plane coordinates [pixel]
    pixelSize  = f.getInputParameter("CCD", "PixelSize")      # [micron]
    plateScale = f.getInputParameter("Camera", "PlateScale")  # [arcsec/micron]
    pixelScale = pixelSize * plateScale                       # [arcsec]
    xFPpix = xFPmm[0] / pixelSize * 1e3
    yFPpix = yFPmm[0] / pixelSize * 1e3
    pis.trace(f"Center of CCD subfield in focal plane : ({xFPpix}, {yFPpix}) pixel")
    
    # Readout noise [e-/pix]
    ron = f.getReadoutNoise()
    pis.trace(f"Readout noise: {ron:.3f}")
    
    # Sky background [e-/pix]
    # TODO need to chnage
    t_exp   = float(from_dom('IntegrationTime',dom))
    t_trans = float(from_dom('ChargeTransferTime',dom))
    bg      = float(from_dom('SkyBackground',dom)) * (t_exp + t_trans)
    pis.trace(f"SkyBackground [e-] : {bg:.3f} (smearing included)")
    
    # Distance from OA
    AngularRadius = np.rad2deg(np.arctan(np.sqrt(xFPpix**2 + yFPpix**2) * pixelScale * arc2rad))
    Azimuth       = np.rad2deg(np.arctan2(yFPpix, xFPpix))
    PSFRadius     = AngularRadius
    pis.trace(f"Star angular radius: {AngularRadius}, azimuth:{Azimuth}")

    # Reference flux at magnitude m=0 [e-]
    flux0 = f.getInputParameter("ObservingParameters", "Fluxm0")
    A   = f.getInputParameter("Telescope", "LightCollectingArea")
    TPw = f.getInputParameter("Camera", "ThroughputBandwidth")
    TP  = f.getThroughput()
    TE  = f.getTransmissionEfficiency()
    flux_m0_ref = fluxm0 * 1e-4 * t_exp * A * TPw * TP * TE

    flux0Ref = pis.get_first_value('FluxM0Ref',dom)
    print(flux_m0_ref, flux0Ref)
    exit()
    
    # PSF information
    PSFbsres   = 10
    PSFbsktype = 1
    model = f.getInputParameter('PSF', 'Model')
    if model == 'MappedFromFile':
        PSFSubPixels = 64
    else:
        PSFSubPixels = 128


    
    if ( FluxM0Ref is None): 
        # this parameter is not present in the XML file
        # searching this parameter into the .info file generated by PIS
        file_info = simdir+'/'+prefix+'.info'
        
        with open(file_info) as myfile:
            buffer = myfile.readlines()
            for b in buffer:
                
                match = re.match('^FluxM0Ref',b)
                if(match):
                    b = re.sub('/.*$','',b) # suppress the comment (if any)
                    b = b.split(':')
                    FluxM0Ref = float(b[1])
                    break
    else:
        FluxM0Ref = float(FluxM0Ref)
        
    if  FluxM0Ref is None: 
        # ------------
        # PIS specific treatment 
        # for old version of PIS in which FluxM0Ref was not written into the .info file
        # ------------
        
        # PSF LocationDependent
        PSFLocationDependent = int(from_dom("PSFLocationDependent", dom))
        
        PSFLocationFileName = clean_string(from_dom('PSFLocationFileName', dom))

        # ------------------------------------------------------ 
        #    Compute reference flux (flux_m0_ref)
        # ------------------------------------------------------ 

        flux_m0 = float(from_dom('Fluxm0',dom))   # Flux from m=0 star [photons/(s * cm^2)
        quant_eff = float(from_dom('QuantumEfficiency',dom))
        trans_eff =   float(from_dom('TransmissionEfficiency',dom))         # Transmission Efficiency
        t_area = float(from_dom('LightCollectingArea',dom))
        FluxM0Ref = flux_m0*trans_eff*t_area*quant_eff*t_exp # reference flux in e- at magnitude m=0
 
        # --------------------
        # Figure out Radius
        # --------------------    
        PSFVignetting = math.cos(deg2rad(PSFRadius))**2 # 'natural' vignetting, used with platosim analytical PSF or PSI location undependant
        
        if(PSFLocationDependent == 1):
            PSFLocation_Filename2, PSFLocation_fileExtension = os.path.splitext(PSFLocationFileName)
            
            # Figure out PSF radial position
            if(PSFLocation_fileExtension == ".hdf5"):
                # For PIS and PlatoSim we look for the PSF within DataDir (grids)
                PSFLocationFullPathFileName = DataDir + PSFLocationFileName
                    
                pis.trace("Loading PSF HDF5 file : {0:}  ".format(PSFLocationFullPathFileName))
                # All PSF within a unique HDF5 file
                if (pis.load_hdf5PSF_properties(PSFLocationFullPathFileName) != 0):
                    pis.trace("ERROR while loading PSF")
                    sys.exit()
                radii = np.asarray([ psf.radius for psf in pis.HDF5Psf_Properties.psfs_properties])
                vignettings = np.asarray([ psf.total_vignetting for psf in pis.HDF5Psf_Properties.psfs_properties])
                i = 0
                for radius in radii:
                    if (i == 0):
                        PSFRadius = radius
                        PSFVignetting = vignettings[i]
                    if (math.fabs(radius - AngularRadius) < math.fabs(PSFRadius - AngularRadius)):
                        PSFRadius = radius
                        PSFVignetting = vignettings[i]
                    i += 1 
            else:
                
                f = open(DataDir + PSFLocationFileName)
                bufferPSF = f.readlines()
                f.close()
    
                i = 0
                for s in bufferPSF:
                    if(not re.match("^#", s)):
                        tmp = s.split()
                        radius = float(tmp[1])
                        vignetting = math.cos(deg2rad(radius))**2 * (1. - float(tmp[7]))  # natural x optical vignetting
                        if (i == 0):
                            PSFRadius = radius
                            PSFVignetting = vignetting
                        if (math.fabs(radius - AngularRadius) < math.fabs(PSFRadius - AngularRadius)):
                            PSFRadius = radius
                            PSFVignetting = vignetting
                            
                        i += 1
        pis.trace("PSF radial position: {radPos}".format(radPos=PSFRadius))
        pis.trace("PSF vignetting: {vignetting}".format(vignetting=PSFVignetting))

        FluxM0Ref *= PSFVignetting
        
        
    pis.trace("Flux at m=0 [e-] (including vignetting) : %e  " % (FluxM0Ref))


    pis.trace("PSF sub-pixel resolution = %i" % PSFSubPixels)

    return PSFSubPixels,PSFbsres,PSFbsktype,PSFRadius, Azimuth,FluxM0Ref,ron,bg







# Sum the matrix A over n x m blocks - the returned matrix
# has 1/n as many columns and 1/m as many rows, from a trick
# mentioned on the PyNum mailing list, neat!
# http://osdir.com/ml/python.numeric.general/2004-08/msg00076.html

def blocksum(A, m, n) :

    M, N = A.shape
    A = np.reshape(A, (int(M / m), m, int(N / n), n))

    return np.sum(np.sum(A, 3), 1)





# read the point spread function and calculate its centroid,
# then return the offset of the centroid from the centre.
# Note, to get the PSF the right way around one should 
# perform a
#
#    psf = psf[ ::-1 , : ]
#
# and then to get the array y-coordinates do the same
# thing to y, but of course you get the same answer.

def centroid_offset(psf) :

    n, m = psf.shape
    x, y = np.meshgrid(np.arange(0, m) + 0.5, np.arange(n) + 0.5)

    sx = float(np.sum(psf * x))
    sy = float(np.sum(psf * y))

    sp = float(np.sum(psf))

    return sx / sp - m / 2, sy / sp - n / 2






# extract the star data from a PLATOsim info file and
# return a list of (x,y,name,[]) quads of those stars
# with magnitude brighter than thresh.
#
# The info file is free-form, i.e., no systematic 
# format, so we do our best: we read lines until we 
# see "List of stars ...", then skip a line, then read 
# RA DECL X Y MAG

def star_info(path, thresh, naming) :

    stars = []

    st = open(path)
    
    line = st.readline()
    while line :

        if re.match(".*RA.*DECL.* X.*Y.*MAG.*", line) :
            break

        line = st.readline()

    line = st.readline()

    id = 0
    while line :

        line = line.rstrip('\n')
        a = list(map(float, line.split()))
       
        if (len(a) < 6):
            (RA, D, X, Y, M) = a
            id += 1
        elif (len(a) < 7):
            (id, RA, D, X, Y, M) = a
        else :
            (id, RA, D, X, Y, M, CCD, Xccd, Yccd, Xfp, Yfp) = a

        if thresh and M > thresh :
            line = st.readline()
            continue

        if naming == "pixel" :
            name = "%04i-%06i-%06i" % \
                tuple(map(round, (M * 100, X * 100, Y * 100)))
        elif naming == "astro" :
            name = "%04i-%06i-%06i" % \
                tuple(map(round, (M * 100, RA * 1000, D * 1000)))
        else:
            name = "%09i" % (id)
               
        stars.append(((X, Y), M, name))

        line = st.readline()

    st.close()

    return stars






def ppfile_info(timespath) :

    pps = []

    st = open(timespath)

    for line in st :
        line = line.rstrip('\n')
        name = line.split(' ')[1]
        (base, ext) = name.split('.')
        pp = "%s-pp.fits" % (base)
        pps.append(pp)

    st.close()

    return pps







# exit with a helpful message if the post-processed file which
# is its argument is not found

def require_pp(path) :

    if not os.access(path, os.F_OK) :
        print("cannot find post-processed file") 
        print("  %s" % (path))
        print("perhaps this directory is not yet post-processed?")
        sys.exit(1)

# get the size of a post processed image





def pp_size(path) :

    require_pp(path)

    # read the image
    img = pf.open(path)[0].data

    # size of image, note the order here!
    return img.shape[1], img.shape[0]





# calculate the efficiency and the normalisation factor of the mask
def efficiency(P, Pr, mask, x ,y , i0, j0) :
    # mask extent
    (Mpy, Mpx) = mask.shape
    
    Psf = pis.Image(subres=Pr,array=P)
    Psf = Psf.view(pis.Psf)
    Psf.normalize()
    Psf.init(subres=Pr)
    image = pis.build_star_image(Psf,x-i0,y-j0,Mpx,Mpy)
    image = pis.rebin2d(image,Mpy,Mpx)
      
    # mutiply PSF section by normalised mask
    T = mask * image
      
    # integrate and divide by integral of psf to 
    # get the mask efficiency
    Ef = np.sum(T.flatten()) 

    # mask characteristic factor (see section 2.2.2 in PLATO-GS-TN-237-LESIA)
    alpha = Ef  / np.sum((mask ** 2 * image).flatten())
    
    return Ef, alpha






def computeBinaryMaskSP(name,stars,Mpx,x,y,P,Pr,Mag,i0,j0,flux_m0_ref,ron,bg,gain,IncludeContaminants=False):
    # Version for sampled PSF
    # in this version the pixels are sorted in decreasing values while they should be sorted in decreasing SNR values
    # see Marchiori et al (2019) and PLATO-LESIA-PDC-DD-0022, i 1.0
    # obsolete version, kept for compatibility issue
    psf = P.view(pis.Psf)
    psf.init(subres=Pr)
    psf.normalize()
    image = pis.build_star_image(psf,x-i0,y-j0,Mpx,Mpx)
    ## print(pis.barycenter(image))
    imagec = pis.Image(sizex=Mpx,sizey=Mpx,subres=Pr)
    # figure out the position of the target at the start of the time-series
    for starn in stars :
        (x0, y0), _ , Name = starn[0:3]
        if(Name == name):
            break

    for starn in stars :
        (xn, yn), Mc , Namec = starn[0:3]
        if(Namec != name):
            # distance btw the contaminant and the target
            Dx = xn-x0
            Dy = yn-y0
            imagec += pis.build_star_image(psf,Dx+x-i0,Dy+y-j0,Mpx,Mpx)*10.**( -((Mc-Mag)/2.5) )
    imagelr = pis.rebin2d(image,Mpx,Mpx)
    imageclr = pis.rebin2d(imagec,Mpx,Mpx)
    I0 = flux_m0_ref*10.**( -(Mag/2.5) )
    I = imagelr.flatten()*I0
    Ic = imageclr.flatten()*I0
    index = np.argsort(I)[::-1]
    I = I[index]
    Ic = Ic[index]
    snrp = 0.
    S = I[0]
    C = Ic[0]
    B = bg
    R = ron**2
    Q = gain**2/12.
    if(IncludeContaminants):
        snrc = S/math.sqrt(S+B+R+Q+C)
    else:
        snrc = S/math.sqrt(S+B+R+Q)
    Mpx2 = Mpx**2
    mask = np.zeros((Mpx2))
    mask[index[0]] = 1
    i = 1
    cont = True
    #print I.shape,Mpx2
    ftot = S
    sprtot = C
    while( cont & (i < Mpx2)):
        S += I[i]
        C += Ic[i]
        B += bg
        R += ron**2
        Q += gain**2/12.
        snrp = snrc
        if(IncludeContaminants):
            snrc =  S/math.sqrt(S+B+R+Q+C)
        else:
            snrc = S/math.sqrt(S+B+R+Q)
        if(snrc>snrp):
            mask[index[i]] = 1
            cont = True
            snrp = snrc
            sprtot = C
            ftot = S
            i += 1
        else:
            cont = False
    key = 0
    for k in range(Mpx2):
        key += mask[k]*2**k
    sprtot = sprtot/ftot
    pis.trace('Optimal binary mask')
    pis.trace("Mask center at x,y = %f, %f" % (x,y))
    pis.trace("Optimal Binary Mask SNR = %f,  Pixel Nb = %i, mask key number: %i, SPRtot = %f" % (snrp,mask.sum(),key,sprtot))
    pis.trace("Readout noise = %f [e-],  Background level = %f [e-], star input flux = %e" % (ron,bg,I0))
    '''             
    print I0,i-1,snrp
    plt.figure(1)
    plt.clf()
    plt.imshow(psflr, interpolation="none") 
    
    plt.figure(2)
    plt.clf()
    plt.imshow(mask.reshape((Mpx,Mpx)), interpolation="none")
    
    plt.show(block=False)
    raw_input('')
    '''
    return mask.reshape((Mpx,Mpx)),snrp,sprtot






def computeBinaryMask(name,stars,Mpx,x,y,P,Pr,Mag,i0,j0,flux_m0_ref,ron,bg,IncludeContaminants=False,gain=0.):
    if(type(P) == pis.Psf ):
        if( P.bscoef is not None):
            # figure out the position of the target at the start of the time-series
            for starn in stars :
                (x0, y0), _ , Name = starn[0:3]
                if(Name == name):
                    break
            DxC = []
            DyC = []
            MagC = []
            for starn in stars :
                (xn, yn), Mc , Namec = starn[0:3]
                if(Namec != name):
                    # distance btw the contaminant and the target
                    DxC.append(xn-x0)
                    DyC.append(yn-y0)
                    MagC.append(Mc)
            mask ,snrp,sprtot = libmask.computeBinaryMaskBS(Mpx,x-i0,y-j0,P,Mag,MagC,DxC,DyC,flux_m0_ref,ron,bg,gain,
                                                            IncludeContaminants=IncludeContaminants,Verbose=True)
            return mask ,snrp,sprtot
    # version for sampled PSF
    pis.trace('WARNING ! no b-spline PSF available, the optimal aperture mask is that case computed using an outdated algorithm')
    return computeBinaryMaskSP(name,stars,Mpx,x,y,P,Pr,Mag,i0,j0,flux_m0_ref,ron,bg,gain,IncludeContaminants=IncludeContaminants)







def computeSPRTotSP(star,name,stars,Mpx,x,y,P,Pr):
    # version for sampled PSF
    (pos, Mag, _, (i0, j0), mask, Gsigma, E, fluxes, alpha,snr,sprtot) = star

    psf = P.view(pis.Psf)
    psf.init(subres=Pr)
    psf.normalize()
    image = pis.build_star_image(psf,x-i0,y-j0,Mpx,Mpx)
    imagec = pis.Image(sizex=Mpx,sizey=Mpx,subres=Pr)
    # figure out the position of the target at the start of the time-series
    for starn in stars :
        (x0, y0), _ , Name = starn[0:3]
        if(Name == name):
            break
    for starn in stars :
        (xn, yn), Mc , Namec = starn[0:3]
        if(Namec != name):
            # distance btw the contaminant and the target
            Dx = xn-x0
            Dy = yn-y0
            ## print(name,x,y,Dx,Dy,xn,yn,x0,y0)
            imagec += pis.build_star_image(psf,Dx+x-i0,Dy+y-j0,Mpx,Mpx)*10.**( -((Mc-Mag)/2.5) )
    imagelr = pis.rebin2d(image,Mpx,Mpx)
    imageclr = pis.rebin2d(imagec,Mpx,Mpx)
    contam = np.sum(imageclr*mask)
    ftarget = np.sum(imagelr*mask)
    sprtot = contam/ftarget
    return sprtot,ftarget







def computeMaskSNRSP(star,name,stars,Mpx,x,y,P,Pr,flux_m0_ref,ron,bg,gain):
    # version for sampled PSF

    (pos, Mag, _, (i0, j0), mask, Gsigma, E, fluxes, alpha,snr,sprtot) = star

    psf = P.view(pis.Psf)
    psf.init(subres=Pr)
    psf.normalize()
    image = pis.build_star_image(psf,x-i0,y-j0,Mpx,Mpx)
    imagec = pis.Image(sizex=Mpx,sizey=Mpx,subres=Pr)
    # figure out the position of the target at the start of the time-series
    for starn in stars :
        (x0, y0), _ , Name = starn[0:3]
        if(Name == name):
            break
    for starn in stars :
        (xn, yn), Mc , Namec = starn[0:3]
        if(Namec != name):
            # distance btw the contaminant and the target
            Dx = xn-x0
            Dy = yn-y0
            ## print(name,x,y,Dx,Dy,xn,yn,x0,y0)
            imagec += pis.build_star_image(psf,Dx+x-i0,Dy+y-j0,Mpx,Mpx)*10.**( -((Mc-Mag)/2.5) )
    imagelr = pis.rebin2d(image,Mpx,Mpx)
    imageclr = pis.rebin2d(imagec,Mpx,Mpx)
    I0 = flux_m0_ref*10.**( -(Mag/2.5) )
    C = np.sum(imageclr*mask)*I0
    T = np.sum(imagelr*mask)*I0
    B = bg
    R = ron**2
    Q = gain**2/12.
    MI = np.sum(mask)
    SNR = T/np.sqrt(T+C+MI*(B+R+Q))
    return SNR






def computeMaskSNR(star,name,stars,Mpx,x,y,P,Pr,flux_m0_ref,ron,bg,gain=0.):

    if(type(P) == pis.Psf ):
        if( P.bscoef is not None):
            (pos, Mag, _, (i0, j0), mask, Gsigma, E, fluxes, alpha,snr,sprtot) = star
            for starn in stars :
                (x0, y0), _ , Name = starn[0:3]
                if(Name == name):
                    break
            DxC = []
            DyC = []
            MagC = []
            for starn in stars :
                (xn, yn), Mc , Namec = starn[0:3]
                if(Namec != name):
                    # distance btw the contaminant and the target
                    DxC.append(xn-x0)
                    DyC.append(yn-y0)
                    MagC.append(Mc)
            return libmask.computeMaskSNRBS(mask,Mpx,x-i0,y-j0,P,Mag,MagC,DxC,DyC,flux_m0_ref,ron,bg,gain)

    return computeMaskSNRSP(star,name,stars,Mpx,x,y,P,Pr,flux_m0_ref,ron,bg,gain)







def computeSPRTot(star,name,stars,Mpx,x,y,P,Pr):
    if(type(P) == pis.Psf ):
        if( P.bscoef is not None):
            (pos, Mag, _, (i0, j0), mask, Gsigma, E, fluxes, alpha,snr,sprtot) = star
            for starn in stars :
                (x0, y0), _ , Name = starn[0:3]
                if(Name == name):
                    break
            DxC = []
            DyC = []
            MagC = []
            for starn in stars :
                (xn, yn), Mc , Namec = starn[0:3]
                if(Namec != name):
                    # distance btw the contaminant and the target
                    DxC.append(xn-x0)
                    DyC.append(yn-y0)
                    MagC.append(Mc)
            return libmask.computeSPRTotBS(mask,Mpx,x-i0,y-j0,P,Mag,MagC,DxC,DyC)

    return computeSPRTotSP(star,name,stars,Mpx,x,y,P,Pr)








def mask_gauss_width_fixinteg(xc, yc, maskinteg, width, sizex, sizey, eps=1e-5):
        '''
        Gaussian mask mask by sampling a Gaussian function over the pixels of the mask window.
        The width of the mask is here adjusted such that the maks intergral equals the input value 'maskinteg'

        '''
        width0 = width
        integ = 0.
        it = 0
        def integral(width):
                z = 1. / (math.sqrt(2.) * width)
                u = math.sqrt(math.pi / 2.) * width
                erfx = (scipy.special.erf((sizex - xc) * z) - scipy.special.erf(-xc * z)) * u
                erfy = (scipy.special.erf((sizey - yc) * z) - scipy.special.erf(-yc * z)) * u
                integ = erfx * erfy
                return integ
        
        while ((math.fabs(integ - maskinteg) > eps) & (it < 200)):
                
            integ = integral(width0)
            integ2 = integral(width0 + 0.1)

            dwidth0 = -(integ - maskinteg) / (integ2 - integ) * 0.1
            if((math.fabs(integ - maskinteg) <= eps)):
                break
            it += 1
            width0 += dwidth0

        return width0, integ





    
def getGsigma(xc, yc, nx, ny, MaskParameters, azimuth, radius, Mag):
    
    tab = MaskParameters['Table']
    if (tab != None):
        xcs = xc - round(xc)
        ycs = yc - round(yc)
        if (azimuth < -90.):
            azimuth = 360. + azimuth
        print(azimuth , radius, Mag, xcs, ycs, round(xc), round(yc))
        # Last table xcs, ycs = 0.5
        Gsigma = tab_sigma(tab, azimuth , radius, Mag, xcs, ycs)
        if (Gsigma is None):
            print('unable to derive the mask width')
            return -1.   

    pis.trace("Gaussian sigma is %.3f" % (Gsigma))
    
    return Gsigma





def checkMaskWithinImagette(Mpx, xc, yc, nx, ny, name, i0, j0):
    # check to see of the mask area will fall outside the
    # image, if it does we drop this star
    #print(Mpx, xc, yc, nx, ny, name, i0, j0)
    def in_range(i, n) :
        # for 6-mask and 6-window (real life), need <=n not < n
        return (i >= 0) and (i + Mpx <= n)

    if not (in_range(i0, nx) and in_range(j0, ny)) :
        pis.trace("no %i-mask at (xc,yc) = (%.4f,%.4f) for %s (i0,j0)=(%i,%i)" % (Mpx, xc, yc, name,i0,j0))
        return -1
    else:
        pis.trace("%i-mask at (xc,yc) = (%.4f,%.4f) with (i0,j0) = (%i,%i) for %s" % (Mpx, xc, yc, i0, j0, name))







        
def computeMask(xc, yc, i0, j0, Mpx, Gsigma):
    # pixel-centre distance to (xc,yc) (in pixels) squared, note 
    # that the 0.5 here accounts for the fact that the coordinates 
    # of the centre of a pixel are at (0.5,0.5), as in the 
    # PLATOsim manual section 2.5 (version 0.96)
    (RX, RY) = np.meshgrid(np.arange(0, Mpx) - xc + i0 + 0.5,
                           np.arange(0, Mpx) - yc + j0 + 0.5)
    D2 = RX * RX + RY * RY

    # gaussian (up to a constant)
    G = np.exp(-D2 / (2 * Gsigma * Gsigma))

    # G normalised to have maximum value of 1.0: the mask 
    mask = G / max(G.flatten())
    
    return mask






def mask_gauss_int(xc,yc,width,sizex,sizey):
    '''
    Gaussian mask derived by integrating a Gaussian function over the pixels of the mask window
    '''
    
    z = 1./(math.sqrt(2.)*width)
    # Error function
    erfx = scipy.special.erf((np.arange(0,sizex+1) - xc)*z)
    erfy = scipy.special.erf((np.arange(0,sizey+1) - yc)*z)
    # normalization required to have the flux density equals to one  at the center
    u = math.sqrt(math.pi/2.)*width   
    ##        u = 0.5 # normalization required to have the mask integral equals to one
    maskx = (erfx[1:sizex+1] - erfx[0:sizex])*u
    masky = (erfy[1:sizey+1] - erfy[0:sizey])*u
    mask = np.zeros((sizey,sizex))
    for i in range(sizey):
            for j in range(sizex):
                    mask[i,j] = masky[i]*maskx[j]
    return mask





def computIntegralMask(xc,yc, i0, j0, Mpx, Gsigma):
    return mask_gauss_int(xc-i0,yc-j0,Gsigma,Mpx,Mpx)






def computeOptimalWeightedMask(Mpx,x,y,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg,itermax=30,eps=1e-5):
    '''
    mask weights are computed such as to optimize the SNR

    '''
    psf = P.view(pis.Psf)
    psf.init(subres=Pr)
    psf = pis.build_star_image(psf,x-i0,y-j0,Mpx,Mpx)
    psflr = pis.rebin2d(psf,Mpx,Mpx)
    I0 = flux_m0_ref*10.**( -(Mag/2.5) )
    image0 = (psflr*I0).ravel()
    n = image0.size
    V = image0 + bg + (ron**2) # variance values

    go = True
    k = 0
    w = np.ones(n)
    wn =  image0.copy()
    x = np.arange(n)
    while(go & (k<itermax) ):
        for j in range(n):
            u = (x != j)
            wn[j] = image0[j]*((w[u]**2*V[u]).sum())/(V[j]*(w[u]*image0[u]).sum())
        epsj = ((w-wn)**2).sum()
        w[:] = wn[:]
        go = (epsj > eps)
        k += 1
##        print k,eps
    ## w = w/np.sum(w*psflr.ravel())
    snr = np.sum(w*image0)/math.sqrt(np.sum(V*w**2))
    w = w.reshape((Mpx,Mpx))        
    pis.trace("Optimal Weigthed Mask SNR = %f" % (snr))       
    pis.trace("Readout noise = %f [e-],  Background level = %f [e-], star input flux = %e" % (ron,bg,I0))     ##    print k,eps
    return w





def computeCircularMask(Mpx,xc,yc,i0, j0,Width):
    xg = (np.arange(0,Mpx)+0.5)
    yg = (np.arange(0,Mpx)+0.5)
    xg, yg = np.meshgrid(xg, yg)
    mask = np.zeros((Mpx,Mpx))
    mask[(((xg-xc+i0)**2 + (yg-yc+j0)**2) < Width**2)] = 1
    
    return mask





def computeHybridMask(name,stars,Mpx,x,y,xc,yc,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg,Width):
    mask,snr,sprtot = computeBinaryMask(name,stars,Mpx,x,y,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg)
    Gmask = computIntegralMask(xc,yc, i0, j0, Mpx,Width )
    u = (mask <1.0)
    mask[u] = Gmask[u]
    return mask






def extended_binary_mask(mask):
    ny,nx = mask.shape
    maske = np.zeros((ny,nx))
    maske[:,:] = mask[:,:]
    w = [1e-1,0.25,1.,0.25,1e-1]
    for j in range(ny):
        for i in range(nx):
            if(mask[j,i]>1.0-1e-5):
                for k in range(-2,3):
                    if(  (j+k>=0) & (j+k<ny) ):
                        for m in range(-2,3):
                            if( (i+m>=0) & (i+m<nx) ):
                                s = w[k+2]*w[m+2]
                                if(maske[j+k,i+m]+s < 1.0):
                                    maske[j+k,i+m] += s
            
    return maske





def computeNormalizedMask(stars, xc, yc, nx, ny, MaskParameters , P, Pr, Mag, name, i0, j0, x, y, azimuth, radius, flux_m0_ref,ron,bg,normalize):
    #print("computeNormalizedMask")
    Mpx = MaskParameters['Size']
    if( MaskParameters['MaskType'] == 0): # Weighted Gaussian mask
        Gsigma = MaskParameters['Gsigma']
        if ( (MaskParameters['UpdateSigma'] == True) or (Gsigma<0.)) :
            Gsigma = getGsigma(xc, yc, nx, ny, MaskParameters, azimuth, radius,  Mag)
            if( Gsigma <0.):
                return -1
        pis.trace("Gsigma = {}".format(Gsigma))
    
    # check to see of the mask area will fall outside the
    # image, if it does we drop this star
    if checkMaskWithinImagette(Mpx, xc, yc, nx, ny, name, i0, j0) == -1:
        #    return -1
        return None, -1., -1., -1., -1.,-1.

    if( (MaskParameters['MaskType'] == 1) | (MaskParameters['MaskType'] == 4)): #  Optimal or Extended binary mask
        Gsigma = -1.
        IncludeContaminants = MaskParameters['IncludeContaminants']
        ## mask,snr,sprtot = libmask.computeBinaryMaskSP(name,stars,Mpx,xc,yc,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg,IncludeContaminants=IncludeContaminants)


        mask,snr,sprtot = computeBinaryMask(name,stars,Mpx,xc,yc,P,Pr,Mag,i0, j0,flux_m0_ref,
                                            ron,bg,IncludeContaminants=IncludeContaminants)

        if( MaskParameters['MaskType'] == 4):
            mask = extended_binary_mask(mask)

    if( MaskParameters['MaskType'] == 2): #  Optimal weighted mask
        Gsigma = -1.
        maskZeroThresh = MaskParameters['maskZeroThresh']
        mask = computeOptimalWeightedMask(Mpx,xc,yc,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg)
        if( maskZeroThresh >0.):
            mask[mask<maskZeroThresh] = 0.

    if( MaskParameters['MaskType'] == 3): #  hybrid optimal mask
        Gsigma = -1.
        mask = computeHybridMask(name,stars,Mpx,x,y,xc,yc,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg,MaskParameters['Width'])

    if( MaskParameters['MaskType'] == 5): #  circular binary mask
        Gsigma = -1.
        mask = computeCircularMask(Mpx,xc,yc,i0, j0,MaskParameters['Width'])
        
        

    if( MaskParameters['MaskType'] == 0): # Weighted Gaussian mask
   
        # Old Method
        # Compute classic mask, value of mask = centroid of pixel gaussian value
    #     mask = computeMask(xc, yc, i0, j0, Mpx, Gsigma)
    #     print "Mask classic"
    #     print sum(mask.flatten())
    #     print np.array(mask)
    #     plt.imshow(mask, interpolation="none")
        
        # new method 2016/11/14
        # Compute classic mask, value of mask = integral of pixel gaussian value
        mask = computIntegralMask(xc,yc, i0, j0, Mpx, Gsigma)
#     print "Mask integral"
#     print sum(mask.flatten())
#     print np.array(mask)
#     plt.figure()
#     plt.imshow(mask, interpolation="none")
#    plt.show()
    
    # mask efficiency and characteristic factor alpha  (see section 2.2.2 in PLATO-GS-TN-237-LESIA)
    Ef, alpha = efficiency(P, Pr, mask, x, y, i0, j0)
    
    if(normalize):
        mask *= alpha  # change the normalization of the mask
        print('change the normalization of the mask, alpha = %f' % alpha)
    else:
        pass
        
    # print "sum mask normalized = {0}".format(sum(mask.flatten()))
    pis.trace("Mask efficiency = {0}".format(Ef))
    ## snr = computeMaskSNR(mask,x,y,P,Pr,Mag,i0, j0,flux_m0_ref,ron,bg)
    pis.trace("Mask SNR = {0}".format(snr))
    
    return mask, Gsigma, Ef, alpha, snr,sprtot
        

def star_update_mask(star, stars,  centroid_shift, mask_size, MaskParameters, P, Pr,
                     odir, azimuth, radius, flux_m0_ref,ron,bg,normalize,offx=0.,offy=0.) :
       
        # unpack star
        (dx, dy) = centroid_shift
        (nx, ny) = mask_size
        ([x, y], Mag, name, (i0, j0), mask, Gsigma, Ef, fluxes, alpha, snr,sprtot) = star
        # centroid corrected x,y coordinates
        xc = x + dx + offx
        yc = y + dy + offy

        pis.trace("Update mask of %s star" % (name))

        # For the update of mask, we don't update i0, j0,
        # We suppose to always have the same mask position within the imagette
        # cf.  Reza
        #(i0, j0) = (int(round(z - float(Mpx) / 2)) for z in (xc, yc))                    
        result = computeNormalizedMask(stars,xc, yc, nx, ny,MaskParameters, P, Pr, Mag, name, i0, j0, x, y, 
                          azimuth, radius, flux_m0_ref,ron,bg,normalize)
        
        if result == -1:
            return -1
        else:
            mask, Gsigma, Ef, alpha,snr,sprtot = result
        
        star = ([x, y], Mag, name, (i0, j0), mask, Gsigma, Ef, fluxes, alpha,snr,sprtot)
        
        return star
        
    
def star_init_mask(star0, stars, centroid_shift, mask_size, MaskParameters,P, Pr, writenbs, odir,azimuth, radius, flux_m0_ref,ron,bg,normalize,
                   offx = 0., offy = 0.) :


        # unpack star0, the first version of star structure
        (dx, dy) = centroid_shift
        (nx, ny) = mask_size
        (x, y), Mag, name = star0 

        # centroid corrected x,y coordinates
        xc = x + dx + offx
        yc = y + dy + offy
        Mpx = MaskParameters['Size']
        pis.trace("Star  %s" % (name))


        # bottom left pixel index - note that these are chosen
        # from the (x,y) positions of the star *after* it is 
        # centroid corrected, which is what PLATO will do (since
        # it only has the centroid of the star). This means that
        # the location of the mask depends on the centroid 
        # corrections (dx,dy)
        (i0, j0) = (int(round(z - float(Mpx) / 2)) for z in (xc, yc))
        ## print((x, y, dx, dy,  xc, yc, Mpx, i0, j0,offx,offy))
        result = computeNormalizedMask(stars,xc, yc, nx, ny,  MaskParameters , P, Pr, Mag, name, i0, j0, x, y, 
                           azimuth, radius, flux_m0_ref,ron,bg,normalize)
        if result == -1:
            return -1
        else:
            mask, Gsigma, Ef, alpha, snr, sprtot = result

        # save the result in new star struct
        star = ([x, y], Mag, name, (i0, j0), mask, Gsigma, Ef, [], alpha,snr,sprtot)

        return star
                
        
def stars_init_masks(stars0, centroid_shift, mask_size, MaskParameters, P, Pr, writenbs, odir, azimuth, radius, flux_m0_ref,ron,bg,normalize, offx=0., offy=0.) :

    (dx, dy) = centroid_shift
    (nx, ny) = mask_size
    stars = []
        
    for star0 in stars0 :
        # Loop over each star within imagette
        star = star_init_mask(star0, stars0, (dx, dy), (nx, ny), MaskParameters , P, Pr, writenbs, odir, azimuth, radius, flux_m0_ref,ron,bg,normalize,offx=offx,offy=offy)
        if star == -1 :
            continue  
        # unpack star
        (pos, Mag, name, (i0, j0), mask, Gsigma, Ef, fluxes, alpha,snr,sprtot) = star 
        
        # append to output struct
        stars.append(star)
        # write the neighbours files - note that a star will always
        # have one neighbour (itself) so we do not need to check that
        # these files are nonempty

        if writenbs and mask is not None:

            nbsfile = "%s/%s.nbs" % (odir, name)
            nbsst = open(nbsfile, "w")
            for starn in stars0 :
                (xn, yn), Mn, namen = starn 
                Efn, _ = efficiency(P, Pr, mask, xn, yn, i0, j0)
                if Efn > 0.0 :
                    nbsst.write("%.3f %.5f %.5f %.6f %s\n" % (Mn, xn, yn, Efn, namen))

            nbsst.close()   
        
    return stars


def get_flux(img, star,bg,CentroidType) :

    # unpack star
    (pos, Mag, name, (i0, j0), mask, Gsigma, E, fluxes, alpha,snr,sprtot) = star
   
    # extent of mask - note the order here
    (My, Mx) = mask.shape

    # imagette (subimg) multiplied by the mask, note 
    # the order here
    subimg = img[j0:j0 + My, i0:i0 + Mx]
    mimg = mask * subimg
        
    # the masked flux
    flux = np.sum(mimg.flatten())

    if flux :

        # centroid of the masked image relative to the mask
        if( CentroidType ==0):
            x, y = np.meshgrid(np.arange(0, Mx) + 0.5, np.arange(My) + 0.5)
            cx = np.sum(mimg * x) / flux
            cy = np.sum(mimg * y) / flux
            # store results, flux and masked image centroid 
            # relative to the imagette
        elif (CentroidType == 1):  
            ## Compute centroid using the Gaussian Analytical Method
            status, cx, cy = libcentroid.GaussianAnalyticalMethod(subimg,0.)
        elif (CentroidType == 2):
            cx, cy = libcentroid.SimpleCentroidMethod(subimg,10)

        fluxes.append((flux, cx, cy))

def writeExposureMask(exposureId, odir, star, dx, dy,offx=0.,offy=0.):
    
    # unpack star
    ([xc, yc], Mag, name, (i0, j0), mask, Gsigma, Ef, fluxes, alpha,snr,sprtot) = star
                
    maskpath = "{0}/{1}-{2:06d}-mask.fits".format(odir, name, exposureId)

    # mask data
    hdu = pf.PrimaryHDU(mask)

    # mask header
    hdr = hdu.header
    if 'set' not in dir(hdr):
        # Old Pyfits version
        hdr.update('i0', i0, 'mask i offset in image')
        hdr.update('j0', j0, 'mask j offset in image')
        hdr.update('x', xc + dx + offx - i0, 'star centroid x position in mask')
        hdr.update('y', yc + dy + offy- j0, 'star centroid y position in mask')
        hdr.update('dx', dx, 'centroid x offset')
        hdr.update('dy', dy, 'centroid y offset')
        hdr.update('E', Ef, 'mask efficiency')
        hdr.update('alpha', alpha, 'mask characteristic factor')
    else:
    # New Pyfits Version > 3.2
        hdr['i0'] = (i0, 'mask i offset in image')
        hdr['j0'] = (j0, 'mask j offset in image')
        hdr['x'] = (xc - i0, 'star centroid x position in mask')
        hdr['y'] = (yc - j0, 'star centroid y position in mask')
        hdr['dx'] = (dx, 'centroid x offset')
        hdr['dy'] = (dy, 'centroid y offset')
        hdr['E'] = (Ef, 'mask efficiency')
        hdr['alpha'] = (alpha, 'mask characteristic factor')
        hdr['snr'] = (snr, 'mask SNR')
        hdr['SPRtot'] = (sprtot, 'total Stellar Pollution Ratio')

    # remove existing file if present
    if os.access(maskpath, os.F_OK) :
        os.remove(maskpath)

    hdu.writeto(maskpath)




    
def run(idir, odir, thresh, naming, MaskParameters,
        writemask, writenbs, DataDir, h5in, normalize, bg,
        CentroidType, compute_sprtot,
        PSFFile, PSFSubPixels, PSFbsktype, PSFbsres,
        add_chromatic_abberation):
    
    # Load paths
    idir = idir.rstrip('/')
    prefix = idir.rpartition('/')[2]
    pis.trace(f"Using prefix {prefix}")

    data = info_hdf5(idir, prefix)
    PSFSubPixelsDef, PSFbsresDef, PSFbsktypeDef, Radius, Az, flux_m0_ref, ron, bg = data

    # read info file for target stars
    coordpath = "%s/%s_starcoord.dat" % (idir, prefix)
    infopath = "%s/%s.info" % (idir, prefix)

    if (os.access(coordpath, os.F_OK)):
        stars = star_info(coordpath, thresh, naming)
    else:
        stars = star_info(infopath, thresh, naming)


    if len(stars) == 0 :
        print("info file lists no suitable targets")
        return

    pis.trace("generating light curves for %i target(s)" % (len(stars)))

    # read the displacement file for update the mask
    displacementPath = "{0:s}/{1:s}_displacement.dat".format(idir, prefix)
    if (os.access(displacementPath, os.F_OK)):
        pis.trace("Read displacement file : {0:s}".format(displacementPath))
        displacements = np.loadtxt(displacementPath, usecols=(0, 1, 2))
        p = 24
        m = int(displacements.shape[0]/p)
        # displacement values are averaged over p exposures:
        displacement = np.zeros((m,3))
        displacement[:,0]  =  pis.rebin1d(displacements[0:p*m,0],m)/float(p)
        displacement[:,1]  =  pis.rebin1d(displacements[0:p*m,1],m)/float(p)
        displacement[:,2]  =  pis.rebin1d(displacements[0:p*m,2],m)/float(p)

    
    updateMask = False
    offx = 0.
    offy = 0.
    if ( (MaskParameters['UpdateThresh'] != np.inf) | MaskParameters['DriftMiddle']):
        if (MaskParameters['UpdateThresh'] != np.inf):
            displacement = tuple(displacement)
            updateMask = True
            maskUpdateThresh = MaskParameters['UpdateThresh']
        else:
            offx = displacement[int(m/2),1]
            offy = displacement[int(m/2),2]
            pis.trace("Mask offset: dX= %f, dY= %f" % (offx,offy))
            
    # Read HDF5 file
    f = SimFile(f'{idir}/{prefix}.hdf5')
    # Number of image : npp
    npp = f.getInputParameter("ObservingParameters", "NumExposures")
    # pps : A simple list of index of images [0, 1, 2, 3]
    pps = list(range(npp))
    # Images
    img = f.getImage()
    ny, nx = img[0].shape
    
    if npp == 0:
        pis.trace("ERROR : No images found, from time file or HDF5")
        return

    pis.trace("each curve with {0:d} time-sample(s)".format(npp))

    # use the first fits file to find the size of the 
    # images

    pis.trace("images    : {0:d} x {1:d}".format(nx, ny))
    Mpx = MaskParameters['Size']
    pis.trace("imagettes : {0:d} x {1:d}".format(Mpx, Mpx))

    # read psf
    if(PSFFile is None):
        psfpath = idir + '/' + prefix + '_psfccd.vec'
        if( os.path.exists(psfpath) ):
             PSFbsktype = PSFbsktypeDef
             PSFbsres = PSFbsresDef
        else:
            psfpath = idir + '/' + prefix + '_psfccd.fits'
        PSFSubPixels = PSFSubPixelsDef
    else:
        psfpath = PSFFile
    pis.trace('PSF file: %s' % psfpath)
    if (re.match('.*\.fits$', psfpath)) :
        # PSF CCD using fits format
        f = pf.open(psfpath)
        psf = f[0].data
        f.close()
    elif (re.match('.*\.vec', psfpath)) :
        if(PSFbsres<0 or PSFbsres is None):
            pis.trace('B-spline resolution not specified, assuming 20')
            PSFbsres = 20
        if(PSFbsktype<0 or PSFbsktype is None):
            pis.trace('B-spline nodes type not specified, assuming 1')
            PSFbsktype = 1
        psfbs = pis.readbinvec(psfpath)
        s = psfbs.shape[0]
        lx = int(math.sqrt(s))
        ly = lx
        psfbs = psfbs.reshape((ly,lx))
        PSFSizex = int(lx / PSFbsres)
        PSFSizey = int(ly / PSFbsres)
        # pixel representation of the PSF:
        PSFSubPixels = PSFSubPixelsDef
        pis.trace("b-spline PSF converted into sampled PSF with sub-pixel resolution of %i " % (PSFSubPixelsDef))
        psf = spline2dbase.Spline2Imagette(psfbs,PSFbsres,PSFSizex,PSFSizey,subres=PSFSubPixels,ktype=PSFbsktype)
        PSFcx,PSFcy = pis.barycenter(psf,subres=PSFSubPixels)
        pis.trace("PSF barycenter: %f , %f " % (PSFcx,PSFcy))
        psfsum = float(psf.sum())

        psf /= psfsum
        psf = psf.view(pis.Psf)
        psf.init(subres=PSFSubPixels)
        psf.normalize()
        psf.barycenter()
        pis.trace( psf.info())
        psf.bsres = PSFbsres
        psf.bsktype = PSFbsktype
        psf.bscoef = psfbs
        psf.bscoef /= psfsum

    else:
        psf = np.loadtxt(psfpath)
    pis.trace("PSF is %i x %i subpixels/pixel" % (PSFSubPixels, PSFSubPixels))

    # get centroid offsets in units of subpixel,

    (dx, dy) = centroid_offset(psf)

    pis.trace("PSF centroid ({0:.5f},{1:.5f}) subpixels from PSF centre".format(dx, dy))

    # convert to units of pixel

    dx /= PSFSubPixels
    dy /= PSFSubPixels

    if(not add_chromatic_abberation):
        dx = 0.
        dy = 0.
        pis.trace("star positions are assumed to include the chromatic aberration, assuming dx=0, dy=0")
    else:
        pis.trace("star positions are assumed to not include the chromatic aberration, "
                  "they are corrected by the offset dx = %f, dy=%f " % (dx,dy))


    # add the masks to the star tuples
    pis.trace("Generating masks")

    stars = stars_init_masks(stars, (dx, dy), (nx, ny), MaskParameters, psf, PSFSubPixels,
                             writenbs, odir, Az, Radius, flux_m0_ref, ron, bg, normalize,
                             offx=offx, offy=offy)

    # Get the fluxes
    
    pis.trace("Generating fluxes")

    # Mask updates
    if updateMask == True:
        pis.trace("Mask update : ENABLED")
        pis.trace("Mask update displacement threshold : {0} pixel".format(maskUpdateThresh))
        lastUpdatedDisplacement = displacement[0]    
        if (MaskParameters['UpdateSigma'] is True):
            pis.trace("Mask update sigma : ENABLED")
        else:
            pis.trace("Mask update sigma : DISABLED")    
    else:
        pis.trace("Mask update : DISABLED")
        
    if updateMask == True :
        if( (MaskParameters['MaskType'] == 1) | (MaskParameters['MaskType'] == 4) ):
            current_maskUpdateThresh = maskUpdateThresh
        else:
            current_maskUpdateThresh = np.random.uniform(low=maskUpdateThresh/3.,
                                                         high=maskUpdateThresh*3.)
            pis.trace("Current mask update threshold: {0}".format(current_maskUpdateThresh))
    
    maskUpdatePeriod = MaskParameters['maskUpdatePeriod']
    
    for starId in range(len(stars)):        
        # unpack star structure
        pos, Mag, name, (i0, j0), mask, Gsigma, E, fluxes, alpha, snr, sprtot = stars[starId]

        if(mask is None):
            pis.trace('no mask for star %s, skipping this star' % name)
            continue

        if(compute_sprtot):
            sprtotfile = open("%s/%s-sprtot.dat" % (odir, name), "w")

        for pp in pps :
            if pp == 0:
                # Write first mask
                writeExposureMask(pp, odir, stars[starId], dx, dy,offx=offx,offy=offy)
                # For compatibility we copy the new mask into the old mask name
                oldMask0Name = "{0}/{1}.fits".format(odir,name)
                newMask0Name = "{0}/{1}-000000-mask.fits".format(odir,name)             
                shutil.copyfile(newMask0Name,oldMask0Name)

            # Update Mask if needed
            # ---------------------------
            if updateMask == True :

                if ( (pp>0) & (pp % maskUpdatePeriod == 0)  & (pp/24 < len(displacement)) ):    

                    # Compute the distance since the last updated displacement
                    # See PLATO-LESIA-PDC-DD-003 : Calculation of a new mask
                    _, tpdx, tpdy = lastUpdatedDisplacement  # Last updated displacement
                    _, tcdx, tcdy = displacement[int(pp/24)-1]  # Current displacement

                    dxy = math.sqrt((tcdx - tpdx)**2 + (tcdy - tpdy)**2)
                    if dxy > current_maskUpdateThresh  :  # For example

                        # Update Mask
                        pis.trace("Exposure {0}, displacement since last update : {1}, ===> Update mask".format(pp, dxy))
                        if( MaskParameters['MaskType'] == 0):
                            current_maskUpdateThresh = np.random.uniform(low=maskUpdateThresh/3.,high=maskUpdateThresh*3.) # choose a random threshold
                            pis.trace("Current mask update threshold: {0}".format(current_maskUpdateThresh))
                        lastUpdatedDisplacement = displacement[int(pp / 24)-1]
                        offx = tcdx
                        offy = tcdy
                        pis.trace("Mask offset: dX= %f, dY= %f" % (offx,offy))
                        stars[starId] = star_update_mask(stars[starId], stars,
                                                         (dx, dy), (nx, ny), MaskParameters,
                                                         psf, PSFSubPixels, odir, Az, Radius,
                                                         flux_m0_ref, ron, bg,
                                                         normalize, offx=offx, offy=offy)
                        print(stars); exit()
                        # Write the masks to a fits file if requested        
                        if writemask:
                            pis.trace("mask saved")
                            writeExposureMask(pp, odir, stars[starId], dx, dy,offx=offx,offy=offy)
                                                  
                 
            # Build Flux
            get_flux(img[pp], stars[starId], bg, CentroidType)
            if compute_sprtot and (pp % 22 == 0):
                # compute SPRtot(t)
                # compute the star position for exposure pp
                (xc, yc), Mag, name, (i0, j0), mask, Gsigma, E, fluxes, alpha, snr, sprtot = stars[starId]
                x = xc + dx + displacements[pp,1]
                y = yc + dy + displacements[pp,2]
                sprtot_t, ftarget = computeSPRTot(stars[starId],name,stars,Mpx,x,y,psf,PSFSubPixels)
                snr = computeMaskSNR(stars[starId],name,stars,Mpx,x,y,psf,PSFSubPixels,flux_m0_ref,ron,bg)
                sprtotfile.write(("%i %f %f %f %f %f\n") % (pp,x-i0,y-j0,sprtot_t,ftarget,snr))

        if(compute_sprtot):
            sprtotfile.close()

    pis.trace("Writing fluxes")

    for star in stars :

        # unpack star structure
        (xc, yc), Mag, basename, (i0, j0), mask, Gsigma, E, fluxes, alpha, snr, sprtot = star
        
        if(mask is None):
            pis.trace('no mask for star %s, skipping this star' % basename)
            continue

        # filter stars with no data (see get_flux)
        if len(fluxes) != npp :
            print("  %s skipped (no data)" % (basename))            
            continue

        # write lightcurve in ascii dat file
        lcpath = "%s/%s.dat" % (odir, basename)
        fd = open(lcpath, "w")

        MaskKey = 0
        Mpx2 = Mpx**2
        maskflat = mask.flatten()
        for k in range(Mpx2):
            MaskKey += maskflat[k]*2**k
        fd.write("# starID: %s // star ID\n" % (basename)) 
        fd.write("# S: %i // mask size\n" % (Mpx))
        fd.write("# xc: %.4f // mask centre [pixel], X axis\n" % (xc+offx-i0))
        fd.write("# yc: %.4f // mask centre [pixel], Y axis\n" % (yc+offy-j0))
        fd.write("# I: %.6f // mask integral\n" % (np.sum(mask.flatten())))
        fd.write("# I2: %.6f // mask square integral (for a weighted mask)\n" % (np.sum(mask.flatten()**2)))
        fd.write("# SNR: %.4f // mask SNR\n" % (snr))
        fd.write("# MaskKey: %i // mask Key number\n" % (MaskKey))
        fd.write("# alpha: %.6f // mask characteristic factor alpha (for a weighted mask)\n" % (alpha))
        fd.write("# sigma: %.3f // Gaussian sigma (for weighted mask)\n" % (Gsigma))
        fd.write("# dx: %.4f // centroid offset [pixel], X axis\n" % (dx))
        fd.write("# dy: %.4f // centroid offset [pixel], Y axis\n" % (dy))
        fd.write("# i0: %i // mask offset, X axis\n" % (i0))
        fd.write("# j0: %i // mask offset, Y axis\n" % (j0))
        fd.write("# flux: %e // star flux [e-]\n" % (flux_m0_ref*10.**(-0.4*Mag)))
        fd.write("# mag: %f // star magnitude\n" %(Mag))
        fd.write("# SPRtot: %f // total Stellar Pollution Ratio\n" %(sprtot))
        # for flux in fluxes :
        #   fd.write("%.4f %.6f %.6f\n" % flux)
        # fd.close()

        # Save data to feather file
        flux = [fluxes[i][0] for i in range(len(fluxes))]
        df = pd.DataFrame()
        df['xc']       = [fluxes[i][1] for i in range(len(fluxes))]
        df['yc']       = [fluxes[i][2] for i in range(len(fluxes))]
        df['flux_oap'] = [fluxes[i][0] for i in range(len(fluxes))]
        df.to_feather(f"{odir}/{name}.ftr")

        pis.trace("  %s" % (basename))

    # xye.close()

def main() :

    def usage() :
        print("usage : photometry [options] <imagette dir>")
        print()
        print(" -a          : astronomical naming scheme")
        print(" -h          : this help")
        print(" -M          : write mask FITS files")
        print(" -m <mag>    : magnitude threshold (obsolete this is done by default)")
        print(" -n <pixels> : number of pixels (square) in mask")
        print(" -N          : write neighbours files (obsolete this is done by default)")
        print(" -o <dir>    : output directory")
        print(" -p          : pixel naming scheme")
        print(" -s <sigma>  : Gaussian mask sigma")
        print(" -u <sigma>  : positional uncertainty sigma (CCD units)")
        print(" -x <thresh> : mask zeroing threshold")
        print(" -B <dir>    : base directory where the input data directories are located.")
        print(" -t <file>   : sigma is determined from the table stored in <file>")
        print(" -T <type>   : mask type: 0  for Gaussian weighted mask, 1 for binary (defaut) ;")
        print("               2 optimal weighted mask ; 3: hybrid mask ; 4: extended optimal binary mask ; 5: circular binary mask")
        print(" -C <type>   : centroid calculation method, O (default) for mask-based method ; 1 for Gaussian Analytical Method (Wang et al 2015)")
        print(" --input-hdf5 : Input images are embeded in a HDF5 file")
        print(" --mask-integ <value> : compute a mask with a width adjusted such that the mask integral equals <value>")
        print(" --normalize            : apply the mask normalization factor 'alpha' stored in the header of the mask file")
        print(" --update-thres <value>   : displacement threshold to update mask (in pixels)")
        print(" --update-sigma   : update the sigma of the mask when --update-thres value is reached")
        print(" --update-period  <number> : period at which the mask is updated (number of exposure, default: 24)")
        print(" --bg  : impose a given bacground level [e-] out noise level, otherwise taken from the configuration file")
        print(" --difkerwidth  : width of the difusion kernel used to compute hybrid masks")
        print(" --drift-middle  : compute optimal mask at the middle of the drift path")
        print(" --spr_tot  : compute SPR_tot as a function of time")
        print(" --include-contaminants: include the contaminants in the calculation of the mask SNR, only applicable for binary masks") 
        print(" -I  <path>    : use the PSF given by <path>")
        print(" -R  <integer> : sub-pixel resolution of the input PSF")
        print(" --bsres <integer>   : PSF is decomposed using b-splines with the resolution given in input")
        print(" --bsktype <integer> : for the b-spline decomposition, type of Knots used (0-> 'simple', 1-> Dierckx's distribution) (default: 1)")
        print(" --add_chromatic_abberation : star positions are assumed to not already include the chromatic_abberation")

        print()
        
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ahMm:Nn:o:pr:s:B:t:T:C:x:I:R:", ["mask-integ=", "input-hdf5",
                                                                               "update-thres=","update-sigma",
                                                                               "normalize","bg=","difkerwidth=",
                                                                               "drift-middle","include-contaminants",
                                                                               "update-period=","spr_tot","bsres=",
                                                                                   "bsktype=","add_chromatic_abberation"])

    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    thresh = None
    naming = "id"
    idir = "."
    odir = "."
    Gsigma = -1.
    masks = True
    nbrs = True
    Mpx = 6
    DataDir = ''
    TabPath = ''
    maskinteg = None
    table = None
    h5in = False
    normalize = False
    MaskType = 1
    CentroidType = 0
    maskUpdateThresh = np.inf
    maskZeroThresh = 0.
    maskUpdateSigma = False
    maskUpdatePeriod = 24
    bg = -1.
    DifKerWidth = 0.4
    DriftMiddle = False
    IncludeContaminants = False
    compute_sprtot = False
    PSFbsres = -1
    PSFbsktype = 1
    PSFFile = None
    PSFsubres = None
    add_chromatic_abberation = False
    for o, a in opts:
        if o == "-h" :
            usage()
            sys.exit()
        elif o == "-T" :
            MaskType = int(a)
        elif o == '-x' :
            maskZeroThresh = float(a)
        elif o == "-C" :
            CentroidType = int(a)
        elif o == "-M" :
            masks = True
        elif o == "-m" :
            thresh = float(a)
        elif o == "-N" :
            nbrs = True
        elif o == "-n" :
            Mpx = int(a)
        elif o == "--normalize" :
            normalize = True
        elif o == "--drift-middle" :
            DriftMiddle = True
        elif o == "-o" :
            odir = a
        elif o == "-p" :
            naming = "pixel"
        elif o == "-a" :
            naming = "astro"
        elif o == "-r" :
            radius = float(a)
        elif o == "-s" :
            Gsigma = float(a)
        elif o == "-B" :
            DataDir = a.rstrip('/')
        elif o == "-t" :
            TabPath = a.rstrip('/')
        elif o == "--mask-integ":
            maskinteg = float(a)
        elif o == "--input-hdf5":
            h5in = True
        elif o == "--update-thres":
            maskUpdateThresh = float(a)
        elif o == "--update-sigma":
            maskUpdateSigma = True
        elif o == "--update-period":
            maskUpdatePeriod = int(a)
        elif o == "--spr_tot":
            compute_sprtot = True
        elif o == "--bg":
            bg = float(a)
        elif o == "--difkerwidth":
            DifKerWidth = float(a)
        elif o == "--include-contaminants":
            IncludeContaminants = True    
        elif o == "--bsres":
            PSFbsres = int(a)
        elif o == "--bsktype":
            PSFbsktype = int(a)
        elif o == "-I" :
            PSFFile = a
        elif o == "-R" :
            PSFsubres = int(a)
        elif o == "--add_chromatic_abberation":
            add_chromatic_abberation = True
        else:
            print("unhandled option %s" % (o))
            sys.exit(1) 

    if (TabPath != '') and (Gsigma > 0.) :
        print("only one of -s and -t may be specified")
        usage()
        sys.exit()

    nargs = len(args)

    if nargs > 1 :
        print("too many arguments")
        usage()
        sys.exit()

    if nargs == 1 :
        idir = args[0]

    if nargs == 0 :
        print("imagette directory must be specified")
        usage()
        sys.exit()

    pis.trace("This is photometry")
    pis.trace("input directory : %s" % (idir))
    pis.trace("output directory : %s" % (odir))

    if thresh :
        pis.trace("magnitude threshold {0:.3f}".format(thresh))
    else :
        pis.trace("no magnitude threshold")

    if naming == "pixel" :
        pis.trace("pixel naming scheme")
    elif naming == "astro" :
        pis.trace("astronomical naming scheme")
    else:
        pis.trace("ID naming scheme")

    if masks :
        pis.trace("creating mask files")
    if nbrs :
        pis.trace("creating neighbour files")

    if (TabPath != ''):
        pis.trace("reading sigma table:" + TabPath)
        table = h5py.File(TabPath, 'r')    
    
    if h5in:
        pis.trace("Using HDF5 input file")

    if( (CentroidType<0) | (CentroidType>3) ):
        pis.trace("unkown centroid method")
        sys.exit(1) 
    
    if( DriftMiddle & (maskUpdateThresh != np.inf) ):
        pis.trace("options --update-thres and are --drift-middle incompatible")
        sys.exit(1) 
        
    
    if(MaskType ==0): # weighted Gaussian Mask
        MaskParameters = {'Size': Mpx, 'MaskType': MaskType, 'Gsigma': Gsigma,'Table': table, 'Integral': maskinteg , 'UpdateThresh': maskUpdateThresh,'UpdateSigma':maskUpdateSigma, 'maskZeroThresh': maskZeroThresh, 'DriftMiddle': DriftMiddle, "maskUpdatePeriod": maskUpdatePeriod}
    elif( (MaskType ==1)| (MaskType==4) ): # Optimal  or Extended Binary Mask
        MaskParameters = {'Size': Mpx, 'MaskType': MaskType, 'UpdateThresh': maskUpdateThresh,'UpdateSigma':False, 'DriftMiddle': DriftMiddle, 'IncludeContaminants': IncludeContaminants, "maskUpdatePeriod": maskUpdatePeriod}
    elif( MaskType ==2) : # Optimal  Weighted Mask
        MaskParameters = {'Size': Mpx, 'MaskType': MaskType, 'UpdateThresh': maskUpdateThresh,'UpdateSigma':False, 'maskZeroThresh': maskZeroThresh, 'DriftMiddle': DriftMiddle, "maskUpdatePeriod": maskUpdatePeriod}
    elif( MaskType ==3): # hybrid optimal   Mask
        MaskParameters = {'Size': Mpx, 'MaskType': MaskType, 'UpdateThresh': maskUpdateThresh,'UpdateSigma':False, 'Width': Gsigma, 'DriftMiddle': DriftMiddle, "maskUpdatePeriod": maskUpdatePeriod}
    elif( MaskType ==5): # circular binary   Mask
        MaskParameters = {'Size': Mpx, 'MaskType': MaskType, 'UpdateThresh': maskUpdateThresh,'UpdateSigma':False, 'Width': Gsigma, 'DriftMiddle': DriftMiddle, "maskUpdatePeriod": maskUpdatePeriod}
    else:
        pis.trace("unkown mask type")
        sys.exit(1) 

    # Run the extraction
    run(idir, odir, thresh, naming, MaskParameters, masks,
        nbrs, DataDir, h5in, normalize, bg, CentroidType, compute_sprtot,
        PSFFile, PSFsubres,PSFbsktype, PSFbsres,
        add_chromatic_abberation)

    pis.trace("Done.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3

"""
This file contains tools to generate and anslyse noise signals.
"""

# Python standard
import os
import sys
import datetime

# PlatoSim standard
import scipy
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import periodogram
from pathlib import Path
from numba import njit

# PlatoSim imports
import platosim.plot            as pt
import platosim.utilities       as ut
import platosim.referenceFrames as rf
from platosim.utilities    import errorcode

#--------------------------------------------------------------#
#                        FREQUENCY DOMAIN                      #
#--------------------------------------------------------------#

def numpyFFT(signal, timestep):
    """Computes power spectrum of an equidistant time series

    The length of the time series need not be a power of 2 (zero padding is done
    automatically). Normalisation is such that a signal A*sin(2*pi*nu_0*t)
    gives power A^2 at nu=nu_0  (IF nu_0 is in the 'freq' array)
    
    @param signal: the time series [0..Ntime-1]
    @type signal: ndarray
    @param timestep: time step fo the equidistant time series
    @type timestep: float
    @return: frequencies and the power spectrum
    @rtype: array,array
    """
    # Compute the FFT of a real-valued signal. If N is the number
    # of points of the original signal, 'Nfreq' is (N/2+1).
    fourier = np.fft.rfft(signal)
    Ntime = len(signal)
    Nfreq = len(fourier)

    # Compute the power
    power = np.abs(fourier)**2 * 4.0 / Ntime**2

    # Compute the frequency array.
    # First compute an equidistant array that goes from 0 to 1 (included),
    # with in total as many points as in the 'fourier' array.
    # Then rescale the array that it goes from 0 to the Nyquist frequency
    # which is 0.5/timestep
    freq = np.arange(float(Nfreq)) / (Nfreq-1) * 0.5 / timestep

    # That's it!
    return freq, power


def DFTpower(time, signal, f0=None, fn=None, df=None, full_output=False):
    """Computes the modulus square of the fourier transform.

    Unit: square of the unit of signal. Time points need not be equidistant.
    The normalisation is such that a signal A*sin(2*pi*nu_0*t)
    gives power A^2 at nu=nu_0

    Parameters
    ----------
    time : ndarray 
        Time points [0..Ntime-1]
    signal : ndarray
        Signal [0..Ntime-1]
    f0, fn, df : float
        The power is computed for the frequencies freq = np.arange(f0,fn,df)
        f0 : frequency of the lower boundary 
        fn : frequency of the upper boundary
        df : frequency sampling of the lower boundary  
    
    Returns
    -------
    freq : ndarray
        Frequencies of DFT
    power : ndarray
        Amplitudes of DFT power spectrum
    """
    freqs = np.arange(f0, fn, df)
    Ntime = len(time)
    Nfreq = int(np.ceil((fn-f0)/df))

    A = np.exp(1j * 2 * np.pi * f0 * time) * signal
    B = np.exp(1j * 2 * np.pi * df * time)
    ft = np.zeros(Nfreq, complex)
    ft[0] = A.sum()
    for k in range(1,Nfreq):
        A *= B
        ft[k] = np.sum(A)

    if full_output:
        return freqs, ft**2*4.0/Ntime**2
    else:
        return freqs, (ft.real**2 + ft.imag**2) * 4.0 / Ntime**2


def DFTpower2(time, signal, freqs):
    """Computes the power spectrum of a signal using a discrete Fourier transform.

    The main difference between DFTpower and DFTpower2, is that the latter allows
    for non-equidistant frequencies for which the power spectrum will be computed.

    @param time: time points, not necessarily equidistant
    @type time: ndarray
    @param signal: signal corresponding to the given time points
    @type signal: ndarray
    @param freqs: frequencies for which the power spectrum will be computed. Unit: inverse of 'time'.
    @type freqs: ndarray
    @return: power spectrum. Unit: square of unit of 'signal'
    @rtype: ndarray
    """
    powerSpectrum = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        arg = 2.0 * np.pi * freq * time
        powerSpectrum[i] = np.sum(signal * np.cos(arg))**2 + np.sum(signal * np.sin(arg))**2

    return powerSpectrum * 4.0 / len(time)**2


def scipyPSD(signal, time_interval=25, window='boxcar', nfft=None, one_sided=False,
             scaling='density'):
    """Compute Double Sided Power Spectral Density (PSD).

    PSD is simply a PS (Power Spectrum) divided by ENBW (Effective Noise Bandwidth).
    This function computes a double sided PSD, so the negative frequencies are not
    removed. Because of this the PS is not multiplied by a factor of 2 when scaling
    it. The scaling only takes into account the sampling frequency and the the loss
    of energy because of applying a window function. As the window is a boxcar,
    we can simply use the length of the time series instead.

    Args:
        time_series ([np.ndarray]): time series data
        time_interval (int, optional): Time interval of the time series in seconds. Defaults to 25.
        detrend (bool or str or int, optional): Whether to apply de-trending or not.
        scipy (bool, optional): Whether to use SciPy periodogram or NumPy FFT. Defaults to True.

    Returns:
        [tuple]: tuple containing frequency array, PSD array, and time series tuple
    """
    time = np.linspace(0, len(signal), len(signal)) * time_interval
    freq, psd = periodogram(signal,
                            fs=1/time_interval,
                            window=window,
                            nfft=nfft,
                            return_onesided=one_sided,
                            scaling=scaling,
                            axis=-1)
    idx = np.argsort(freq)
    psd = psd[idx]
    freq = freq[idx]

    return freq, psd



def astropyLombScargle(times, signal, f0=0, fn=0, df=0, norm='amplitude'):

    import astropy.timeseries as apy

    # times and signal are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(times)
    mean_s = np.mean(signal)
    times_ms = times - mean_t
    signal_ms = signal - mean_s

    # setup
    n = len(signal)
    t_tot = np.ptp(times_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if (df == 0):
        df = 0.1 / t_tot
    if (fn == 0):
        fn = 1 / (2 * np.min(times_ms[1:] - times_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    f1 = f0 + np.arange(nf) * df

    # use the astropy fast algorithm and normalise afterward
    ls = apy.LombScargle(times_ms, signal_ms, fit_mean=False, center_data=False)
    s1 = ls.power(f1, normalization='psd', method='fast', assume_regular_frequency=True)

    # replace negative by zero (just in case - have seen it happen)
    s1[s1 < 0] = 0

    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(signal_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
        
    return f1, s1

#--------------------------------------------------------------#
#                          TIME DOMAIN                         #
#--------------------------------------------------------------#

def timeSeriesFromFourier(time, freq, ampl, phase, power=1, plot=False, title=False):

    """Generate light curve from Fourier info.

    Paramters
    ---------
    time : ndarray, pdframe
        Time points of which light curve will be generated [d]
    freq : ndarray, pdframe
        Frequencies of sinusoids [c/d]
    ampl : ndarray, pdframe
        Amplitudes of sinusoids [mag]
    phase : ndarray, pdframe
        Phases of sinusoids [rad]

    Returns
    -------
    signal : ndarray
        Signal for each time point [as ampl]

    Notes
    -----
    Conversion: 1 ppt (ppm) = 1.0863 mmag (mumag)
    Following: m1-m2 = -2.5 log(f2/f1) => dm = -2.5 log(1-df)
    """

    # Number of modes
    N = len(freq)

    # Compute signal from a sum of all the modes
    signal = np.zeros_like(time)
    for i in range(N):
        signal += ampl[i] * np.sin((2*np.pi * freq[i]) * time + phase[i])

    # Normalize the magnitude so its values is in [-1, 1]
    # Then add 1, such that the roots are not undefined
    # Raise the power
    # Normalise the signal [0, 1]
    # Subtract the mean and multiply with overall amplityde
    A = np.max(np.abs(signal))
    signal = (1 + signal / A)**power
    signal = signal / np.max(signal)
    signal = - A * (signal - np.mean(signal))

    # If requested, plot model 
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))

        # Plot time series
        ax[0].plot(time, signal*1e3, 'k-', lw=0.4)
        ax[0].set_xlabel(r'Time [d]')
        ax[0].set_ylabel(r'Signal [mmag]')
        ax[0].set_xlim(time.min(), time.max())
        if title: ax[0].set_title(str(title))
        
        # Generate DFT for regular sampling
        fn = np.max(freq) + 0.1
        df = np.diff(time)[0] * 2
        freq0, ampl0 = DFTpower(time, signal*1e3, f0=0, fn=fn, df=df)
        amax = np.max(ampl0)
        for i in range(N):
            if i == 0:
                ax[1].vlines(x=freq[i], ymin=-0.1*amax, ymax=0, colors='b', alpha=0.3,
                         label='Input freq.')
            else:
                ax[1].vlines(x=freq[i], ymin=-0.1*amax, ymax=0, colors='b', alpha=0.3)       
        ax[1].plot(freq0, ampl0, '-', c='deeppink', lw=1, label='DFT')
        ax[1].set_ylabel(r'Amplitude [mmag]')
        ax[1].set_xlabel(r'Frequency [c/d]')
        ax[1].set_xlim(0, fn)
        ax[1].set_ylim(-0.1*amax, amax+0.1*amax)
        ax[1].legend()

        # Settings
        plt.tight_layout()
        plt.show()

    # Return signal
    return signal





def timeSeriesFromMeanPSD(freq, psd):

    """
    PURPOSE: Given the average noise profile in the power spectral density, compute the 
             corresponding noisy time series. 

    INPUT: freq: Array containing {n / N / deltat } with n in [0,..,Ntime/2],
                 where N is the number of time points, and deltat the time step.
                 Unit: [Hz | microHz | mHz]
           psd:  Power spectral density: abs(fourier)**2 / Ntime * timestep
                 Unit: [ppm^2/Hz | ppm^2/microHz | ppm^2/mHz]

    OUTPUT: time:    time points [s | Ms | KHz]
            signal:  [ppm]

    EXAMPLE: 
        >>> timeStep = 25.0e-6    # in Ms
        >>> Nfreq = 15001
        >>> freq = arange(float(Nfreq)) / (Nfreq-1) * 0.5 / timeStep
        >>> omega = 2*np.pi*freq
        >>> omegaMin  = 2. * np.pi *  20.0
        >>> omegaKnee = 2. * np.pi * 200.0
        >>> meanPSD = (omega**2 + omegaKnee**2)/(omega**2 + omegaMin**2) *  timeStep
        >>> time, signal = timeSeriesFromMeanPSD(freq, meanPSD)
        >>> nu, noisyPSD = FFTpowerdensity(signal, timeStep)
        >>> plt.loglog(nu, noisyPSD, c="b")
        >>> plt.loglog(freq, meanPSD, c="r")

    NOTE: - N points in the time domain correspond to N/2+1 points in the frequency 
            domain (not N/2), where the division is an _integer_ division (rounded down).

    """

    # Determine the number of time points. 

    Nfreq = len(freq)
    Ntime = 2*(Nfreq-1)

    # Determine the frequency resolution
    # Note: N points in the time domain corresponds to N/2+1 points in the frequency domain (not N/2).

    freqStep = freq[1] - freq[0]
    timeStep = 1./(freqStep*Ntime)

    # Generate the time points of the signal

    time = np.arange(Ntime) * timeStep

    # Generate the real and imaginary parts of the fourier transform with gaussian noise

    realPart = normal(0., 1., Nfreq)
    imagPart = normal(0., 1., Nfreq)

    # Construct the full fourier spectrum, using the proper scale factor
    
    fourier = np.sqrt(psd * Nfreq / timeStep) * (realPart + imagPart * 1j)

    # Inverse-FFT the fourier transform to generate the time series [ppm]

    signal = np.real(np.fft.irfft(fourier)) 

    # That's it!

    return time, signal






def apodization_correction(nu, dt):

    """Amplitude correction due to apodization.

    This function calculates the amplitude correction needed to account
    for apodization (c.f. Hekker & Christensen-Dalsgaard, 2017) for a
    amplitude spectrum (hence now a power spectrum as in the paper).

    NOTE: NumPy defined the normalized sinc() function, but apodization
          is defined by the unnormalized sinc() function. Hence, we have
          removed the normalization factor of pi in the original Equation.
  
    Parameters
    ----------
    nu : float, ndarray
        Frequency of pulsations amplitudes [c/d]
    dt : float
        Sampling rate (or cadence) of instruement [s]

    Returns
    -------
    Apodization factor to be multiplied with observed amplitudes 
    c.f. Bowman & Kurtz (2018).
    """

    # Nyquist frequency of instrument [c/d]
    nu_Nyquist = 86400 / (2 * dt)

    return np.sqrt(np.sinc(nu/nu_Nyquist)**(-2))


#--------------------------------------------------------------#
#                     OPTIMUM SNR CRITERION                    #
#--------------------------------------------------------------#


def getNoisePeakSNR(cadence, quarters=1, N=1000, odir=None):

    """Find SNR of largest noise peak.

    This function generates a white noise PLATO light curve with
    a duration equal to the number of mission quarters parsed.
    Note that it is assumed that two days are lost overall for
    every mission quarter.

    Parameters
    ----------
    cadence : int
        Cadence (exposure + readout time) for observation [s]
    quarters : int
        Number of mission quarters (e.g. 1, 2, ...)
    N : int 
        Number of iterative calculations to make
    odir : str
        Output directory to save file (handy for large N)

    Returns
    -------
    Data frame with N number of SNR values. 

    NOTE this function needs platonium packages.
    """
        
    from tqdm import tqdm

    # Prepare for calculation
    
    texp = cadence / 86400.
    tdur = quarters * (quarter() - 2)
    time = np.arange(0, tdur, texp)
    df   = 0.1 / np.ptp(time)
    snr  = np.zeros(N)

    # Loop over N iterations
    
    for i in tqdm(range(N), bar_format=tqdmBar()):
        
        noise = np.random.normal(0, 1, len(time))
        freq, ampl = astropyLombScargle(time, noise, df=df)        
        ampl /= np.median(ampl)
        snr[i] = np.max(ampl) / np.median(ampl)

    dx = pd.DataFrame({'snr':snr})

    # Save output file
    
    if odir:
        filename = f'{odir}/snr_quarters{quarters}_cadence{cadence}.ftr'
        dx.to_feather(filename)

    return dx





def plotNoisePeakSNR(path, cadence, quarters, fap=0.1, bins=50, figsize=(8,5)):

    """Plot SNR histogram of largest noise peak and FAP.

    Parameters
    ----------
    path : str
        Directory where SNR file are stored 
    cadence : int
        Cadence (exposure + readout time) for observation [s]
    quarters : int
        Number of mission quarters (e.g. 1, 2, ...)
    fap : float 
        False Alarm Probability (FAP) matching SNR criterion
    bins : int
        Number of histogram bins to plot
    figsize : mpl object
        Matplotlib figure object to alter figure dimentions

    Returns
    -------
    fig, ax : mpl figure objects

    NOTE this function needs platonium packages.
    """

    import platosim.statistics as st

    # Load file with SNR values    
    
    filename = f'{path}/snr_quarters{quarters}.ftr'
    dx = pd.read_feather(filename)
    dx = dx[f'snr{cadence}']
    
    # Calculate requested FAP
    snr_fap = st.hist_fap(dx, fap=fap)[0]

    # Start plotting

    fig, ax = plt.subplots(1,1, figsize=figsize)
    # Plots
    ax.hist(dx, bins=bins, histtype='step', label=r'$\Delta t = $'+f' {cadence}s',
            fc='b', ec='b', fill=True, alpha=0.3)
    ax.axvline(x=snr_fap, c="b", ls="--", lw=1.5, zorder=2)
    # Settings
    ax.set_xlabel(r'SNR amplitude')
    ax.set_ylabel('Number of stars')
    # Settings
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Print best SNR criterions
    print(snr_fap)
    
    return fig, ax






def getMultiCadenceNoisePeakSNR(quarters=1, N=1000, odir=None):

    """Find SNR of largest noise peak.

    This is a custom function to calculate the SNR for
    the three cadences {25 50, 600} seconds.

    Parameters
    ----------
    quarters : int
        Number of mission quarters (e.g. 1, 2, ...)
    N : int 
        Number of iterative calculations to make
    odir : str
        Output directory to save file (handy for large N)

    Returns
    -------
    Data frame with N number of SNR values. 

    NOTE this function needs platonium packages.
    """
    
    from tqdm import tqdm

    # Prepare for calculation
    
    tdur = quarters * (quarter() - 2)
    cadence = np.array([25, 50, 600]) / 86400
    time0 = np.arange(0, tdur, cadence[0])
    time1 = np.arange(0, tdur, cadence[1])
    time2 = np.arange(0, tdur, cadence[2])
    df0 = 0.1 / np.ptp(time0)
    df1 = 0.1 / np.ptp(time1)
    df2 = 0.1 / np.ptp(time2)
    snr = np.zeros((N, 3))

    # Loop over N iterations
    
    for i in tqdm(range(N), bar_format=tqdmBar()):
    
        noise0 = np.random.normal(0, 1, len(time0))
        noise1 = np.random.normal(0, 1, len(time1))
        noise2 = np.random.normal(0, 1, len(time2))

        # Introduce 2 day gaps (do not work with simple DFT)
        # for i in range(1, quarters+1):
        #     t0 = (i * quarter()) - 2
        #     t1 =  i * quarter()
        #     dex0 = np.where((time0 > t0) & (time0 < t1))
        #     dex1 = np.where((time1 > t0) & (time1 < t1))
        #     dex2 = np.where((time2 > t0) & (time2 < t1))
        #     time0  = np.delete(time0, dex0)
        #     time1  = np.delete(time1, dex1)
        #     time2  = np.delete(time2, dex2)
        #     noise0 = np.delete(noise0, dex0)
        #     noise1 = np.delete(noise1, dex1)
        #     noise2 = np.delete(noise2, dex2)
        
        freq0, ampl0 = astropyLombScargle(time0, noise0, df=df0)
        freq1, ampl1 = astropyLombScargle(time1, noise1, df=df1)
        freq2, ampl2 = astropyLombScargle(time2, noise2, df=df2)
        
        ampl0 /= np.median(ampl0)
        ampl1 /= np.median(ampl1)
        ampl2 /= np.median(ampl2)
    
        snr[i, 0] = np.max(ampl0) / np.median(ampl0)
        snr[i, 1] = np.max(ampl1) / np.median(ampl1)
        snr[i, 2] = np.max(ampl2) / np.median(ampl2)

    dx = pd.DataFrame({'snr25':snr[:,0], 'snr50':snr[:,1], 'snr600':snr[:,2]})

    # Save output to file
    
    if odir:
        filename = f'{path}/snr_quarters{quarters}.ftr'
        dx.to_feather(filename)

    return dx
    




def plotMultiCadenceNoisePeakSNR(odir, quarters=1, fap=0.1, bins=50,
                                 show_snr=False, figsize=(8,5)):

    """Plot SNR histogram of largest noise peak and FAP.
    
    This is a custom function to plot the SNR for
    the three cadences {25 50, 600} seconds.

    Parameters
    ----------
    path : str
        Directory where SNR file are stored 
    quarters : int
        Number of mission quarters (e.g. 1, 2, ...)
    fap : float 
        False Alarm Probability (FAP) matching SNR criterion
    bins : int
        Number of histogram bins to plot
    figsize : mpl object
        Matplotlib figure object to alter figure dimentions

    Returns
    -------
    fig, ax : mpl figure objects

    NOTE this function needs platonium packages.
    """
    
    import platosim.statistics as st

    # Load file with SNR values
    
    filename = f'{odir}/snr_quarters{quarters}.ftr'
    dx = pd.read_feather(filename)

    # Calculate requested FAP
    
    snr_fap0 = st.hist_fap(dx.iloc[:,0], fap=fap)[0]
    snr_fap1 = st.hist_fap(dx.iloc[:,1], fap=fap)[0]
    snr_fap2 = st.hist_fap(dx.iloc[:,2], fap=fap)[0]

    # Start plotting
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    # Plots
    aa = 0.4
    c = ['darkblue', 'orange', 'm']
    ax.hist(dx.iloc[:,0], bins=bins, histtype='step', label=r'$\Delta t = 25$s',
            fc=c[0], ec=c[0], fill=True, alpha=aa)
    ax.hist(dx.iloc[:,1], bins=bins, histtype='step', label=r'$\Delta t = 50$s',
            fc=c[1], ec=c[1], fill=True, alpha=0.5)
    ax.hist(dx.iloc[:,2], bins=bins, histtype='step', label=r'$\Delta t = 600$s',
            fc=c[2], ec=c[2], fill=True, alpha=aa)
    ax.axvline(x=snr_fap0, c=c[0], ls="--", lw=1.5, zorder=2)
    ax.axvline(x=snr_fap1, c=c[1], ls="--", lw=1.5, zorder=2)
    ax.axvline(x=snr_fap2, c=c[2], ls="--", lw=1.5, zorder=2)
    # Labels
    ax.set_xlabel(r'SNR')
    ax.set_ylabel('Number of stars')
    # Settings
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Print best SNR criterions
    if show_snr:
        print([snr_fap0, snr_fap1, snr_fap2])
    
    return fig, ax





#--------------------------------------------------------------#
#                    STOCASHIC NOISE SOURCES                   #
#--------------------------------------------------------------#


@njit
def getRedNoise(time, currenttime, kicktimestep, Ntime,
                timescale, varscale, noise, mu, sigma,
                rng):
        
    signal = np.zeros(Ntime)
    
    for i in range(Ntime):

        # Compute the contribution of each component separately.
        # First advance the time series right *before* the time point i,

        while ((currenttime + kicktimestep) < time[i]):
            noise = noise * (1.0 - kicktimestep/timescale) + rng.normal(mu[0], sigma[0])
            currenttime = currenttime + kicktimestep

        # Then advance the time series with a small time step right *on* time[i]

        delta  = time[i] - currenttime
 
        # Correction factor to have varscale in RMS arcsec

        #sigma1 = np.sqrt(delta/timescale)*varscale
        sigma1 = varscale * 0.66
        noise  = noise * (1.0 - delta/timescale) + rng.normal(mu[0], sigma1[0])
        currenttime = time[i]

        # Add the different components to the signal. 

        signal[i] = np.sum(noise)

    return signal





def modelRedNoise(time, timescale, varscale, kickscale=100, n_warmup=2000, seed=None):

    """Function to generate a red noise time series.
    
    Parameters
    ----------
    time : ndarray
        Time points: time[0..Ntime-1]
    timescale : ndarray
        Time scale tau of each red noise component: timescale[0..Ncomp-1]
    varscale : ndarray
        Variation scale of each red noise component: varscale[0..Ncomp-1]
            
    Returns
    -------
    signal : ndarray
        Signal containing all red noise components: signal[0..Ntime-1]
    """
    
    # Initialise random generator
    rng = ut.rng(seed=seed)

    Ntime = len(time)
    Ncomp = len(timescale)

    # Set the kick (= excitation) timestep to be one 100th of the
    # shortest noise time scale (i.e. kick often enough).

    timescale = np.sqrt(timescale)
    
    kicktimestep = min(timescale) / kickscale
    currenttime  = time[0] - kicktimestep
    
    # Predefine some arrays

    delta = 0.0
    noise = np.zeros(Ncomp)
    mu    = np.zeros(Ncomp)
    sigma = np.sqrt(kicktimestep / timescale) * varscale

    # Warm up the first-order autoregressive process

    for i in range(n_warmup):
        noise = noise * (1 - kicktimestep / timescale) + rng.normal(mu, sigma)

    # Start simulating the granulation time series
    signal_red = getRedNoise(time, currenttime, kicktimestep, Ntime,
                             timescale, varscale, noise, mu, sigma, rng)
    return signal_red
    




def modelRedNoisePSD(freq, timescale, varscale):

    """Function to generate a red noise model from the PSD.

    Compute the mean power spectral density (PSD) corresponding to the 
    red noise time series that is generated by modelRedNoise().

    INPUT: freq:       frequency points of the PSD  [microHz | mHz | Hz]
           timescale:  see the function rednoise()
           varscale:   see the function rednoise()

    OUTPUT: psd:  power spectral density   [ppm^2/microHz | ppm^2/microHz | ppm^2/microHz]
    """

    psd = np.zeros_like(freq)

    for n in range(len(timescale)):
        sigma = varscale[n]
        tau = timescale[n]
        psd += sigma * sigma * tau / (1.0 + (2.0*np.pi*freq*tau)**2)
    
    return psd





#--------------------------------------------------------------#
#                   SYSTEMATIC NOISE SOURCES                   #
#--------------------------------------------------------------#


def getPRE(alpha, delta, kappa, quarter, sigma=3,
           seed=None, ofile=False, table=False, plot=False):

    """Pointing Reproducibility Error (PRE) in PLM reference frame.
    
    Paramters
    ---------
    alpha : float
        Equtorial right ascension [rad]
    delta : float
        Equatorial declination [rad]
    kappa : float
        Orientation of the solar panel [rad]
        This corresponds to (Q1, Q2, Q3, Q4) = (0, pi/2, pi, 3pi/2)
    quarter : range(int, int)
        Mission quarter range, e.g. [Q1, Q8] = range(1, 9)
    sigma : float
        Standard deviation to draw within
    ofile : str
        Output filename pointing to storage directory
    table : bool
        Option to print the table produced
    plot : bool
        Option to show the plot of the PRE data

    Return
    ------
    - A pandas dataframe with PRE data
    - Optionally a feather output file
    """

    # Random number generator
    rng = ut.rng(seed)
    
    # Sort input quarters    
    n = len(quarter)
        
    # PRE in the PLM reference frame (yaw, pitch, roll)
    # Here t stands for transverse direction and [deg]
    # NOTE: Performance values "as required"    
    t = 3.0/3600 
    b = 6.0/3600

    # Find distribution within 3 sigma of req.
    tt = np.array([rng.normal(0, t/sigma) for i in range(n)])
    bb = np.array([rng.normal(0, b/sigma) for i in range(n)])

    # Corresponding yaw, pitch, roll
    # y = tt
    # z = 3 * y
    # x = bb - z
    y = tt
    z = bb - y
    x = tt

    # ICRS pointing angles
    phi   = np.deg2rad(alpha)
    theta = np.deg2rad(delta)

    # Find change of pointing for each quarter
    PRE = np.zeros((n, 4))
    for i in range(n):
        data = rf.perturbPlatformPointing(x[i], y[i], z[i], phi, theta)[0]
        PRE[i,:] = np.append(quarter[i], data)
    df = pd.DataFrame(PRE, columns=["quarter", "yaw", "pitch", "roll"])
    df = df.astype({"quarter":int, "yaw":np.float64, "pitch":np.float64, "roll":np.float64})
    df0 = df.copy()
    df0.iloc[:,1:] = df0.iloc[:,1:] * 3600
    df1 = df.copy()
    df1.rename(columns={"yaw":"alpha", "pitch":"delta", "roll":"kappa"}, inplace=True)
    df1.iloc[:,1] = df1.iloc[:,1] + alpha
    df1.iloc[:,2] = df1.iloc[:,2] + delta
    df1.iloc[:,3] = df1.iloc[:,3] + kappa

    # Print generated values
    if table:
        print('\nChange of coordinates [arcsec]')
        print(df0)
        print('\nNew ICRS coordinates [deg]')
        print(df1)

    # Plot distributions
    t *= 3600
    b *= 3600
    y = t/sigma
    z = 3 * y
    x = np.abs(b/sigma - z)
    xx = np.linspace(-10*x, 10*x, 1000)
    alpha0 = (df1.alpha - alpha) * 3600 / 18
    delta0 = (df1.delta - delta) * 3600 / 18

    fig, ax = plt.subplots(1, 2, figsize=(9,4))

    # Plot PDF
    ax[0].set_title(f'PRE distributions at {sigma}$\sigma$')        
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, x)*100, '-', c='b', label='Trans.')
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, z)*100, '-', c='m', label='Rot.')
    ax[0].set_xlabel('Platform pointing errors in FPA [pixel]')
    ax[0].set_ylabel('Probability (PDF) [\%]')
    ax[0].set_xlim(xx[0], xx[-1])
    ax[0].legend()

    # Show distribution on sky 
    ax[1].grid(zorder=0)
    ax[1].plot(0, 0, 'k*', ms=10, zorder=2)
    for i in range(len(alpha0)):
        ax[1].scatter(alpha0[i], delta0[i], marker=f'${i+1}$', s=50, alpha=0.8, zorder=2)
    ax[1].set_title('Distribution on Sky')
    ax[1].set_xlabel('RA [pixel]')
    ax[1].set_ylabel('Dec [pixel]')
    ax[1].set_aspect('equal', adjustable='box')

    # Settings
    lim = np.max([np.max(np.abs(alpha0)), np.max(np.abs(delta0))])
    lim += lim/10.
    ax[1].set_xlim(-lim, +lim)
    ax[1].set_ylim(-lim, +lim)    
    plt.tight_layout()
        
    # Plot figure above
    if plot: plt.show()
    
    # Save file with relative pointing errors [deg]
    if ofile:
        df.to_csv(ofile, sep=" ", header=False, index=False)
        fig.savefig(f"{ofile[:-4]}.png", bbox_inches='tight', dpi=200)

    return PRE





def getAPE(alpha, delta, kappa, sigma=3,
           seed=None, ofile=False, table=False, plot=False):

    """Pointing Reproducibility Error (PRE) in P/L reference frame.

    The mission reuquirements are 4.5 arcmin in X and Y, and 9 arcmin
    for the rotation. This gives a RMS value of 1.5 arcsec (6 pixel)
    
    Paramters
    ---------
    alpha : float
        Equtorial right ascension [rad]
    delta : float
        Equatorial declination [rad]
    kappa : float
        Orientation of the solar panel [rad]
        This corresponds to (Q1, Q2, Q3, Q4) = (0, pi/2, pi, 3pi/2)
    sigma : float
        Standard deviation to draw within
    ofile : str
        Output filename pointing to storage directory
    table : bool
        Option to print the table produced
    plot : bool
        Option to show the plot of the APE data

    Return
    ------
    - A pandas dataframe with APE data
    - Optionally a feather output file
    """

    # APE in the PLM reference frame (yaw, pitch, roll)
    # Here t stands for transverse direction and [deg]    
    # NOTE: Performance values "as required"

    # Random number generator
    rng = ut.rng(seed)
    
    t = 4.5/60  # [deg]
    b = 9.0/60  # [deg]
        
    # Find distribution within 3 sigma of req.
    tt = np.array([rng.normal(0, t/sigma) for i in range(26)])
    bb = np.array([rng.normal(0, t/sigma) for i in range(26)])

    # Corresponding yaw, pitch, roll
    dy = tt
    dz = 3 * dy
    dx = bb - dz

    # Store APE
    APE = np.transpose([tt, bb])
    df  = pd.DataFrame(APE, columns=["tilt", "azimuth"])

    # Print distributions to bash
    if table:
        print(f'\nCamera alignment errors for all 26 cameras [pixel]')
        APE0 = np.transpose([tt, bb, dx, dy, dz]) * 3600 / 15
        df0  = pd.DataFrame(APE0, columns=["Alt", "Az", "Yaw", "Pitch", "Roll"])
        print(df0)
        
    # Create figure object
    t *= 3600 / ( 15 * sigma * (sigma-1))
    b *= 3600 / ( 15 * sigma * (sigma-1))
    xx = np.linspace(-10*t, 10*t, 1000)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    # Plot PDF
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, t)*100, '-', c='b', label='Trans.')
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, b)*100, '-', c='m', label='Rot.')
    ax[0].set_title(f'APE distributions at {sigma}$\sigma$')
    ax[0].set_xlabel('Camera misalignment in FPA [pixel]')
    ax[0].set_ylabel('Probability (PDF) [\%]')
    ax[0].set_xlim(xx[0], xx[-1])
    ax[0].legend()

    # Plot distribution on sky
    azim = df.azimuth*3600/15
    tilt = df.tilt*3600/15
    ax[1].grid(zorder=0)
    ax[1].plot(0, 0, 'k*', ms=10, zorder=2)
    for i in range(24):
        if i < 6:
            ax[1].scatter(azim[i], tilt[i], marker=rf'${i+1}$', s=50, c='g', alpha=0.8,zorder=2)
        else:
            ax[1].scatter(azim[i], tilt[i], marker=rf'${i+1}$', s=70, c='g', alpha=0.8,zorder=2)
    ax[1].scatter(azim[24], tilt[24], marker=r'$1$', s=50, c='b', alpha=0.8, zorder=3)
    ax[1].scatter(azim[25], tilt[25], marker=r'$2$', s=50, c='r', alpha=0.8, zorder=3)
    ax[1].set_title('Relative offset')
    ax[1].set_xlabel('Azimuth [pixel]')
    ax[1].set_ylabel('Tilt [pixel]')
    ax[1].set_aspect('equal', adjustable='box')

    # Settings
    lim = np.max([np.max(np.abs(azim)), np.max(np.abs(tilt))])
    lim += lim/10.
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    plt.tight_layout()

    # Plot figure above 
    if plot: plt.show()
    
    # Save APE camera misalignments
    if ofile:
        df.to_csv(ofile, sep=" ", header=False, index=False)
        fig.savefig(f"{ofile[:-4]}.png", bbox_inches='tight', dpi=200)

    return APE





def getTED(quarter, model="poly", wheel_offloading=False, ampl=2,
           ofile=False, seed=None, table=False, plot=False):

    """Generate a Themo-Elastic Drift (TED) file.
   
    This function generates a complete TED model returned in euler angles.

    TODO Wheel offloading events needs more investigations before usage.

    Paramters
    ---------
    quarter : range
        Range of quarters (e.g. range(1,8) for Q1-Q8).
    model : str
        Model to produce TED: 'linear' or 'poly'.
    ofile : bool, str
        Parse string to save the model to a ascii file.
    plot : bool
        True will make a plot of the models.
        
    Return
    ------
    Output file if requested.
    """

    # Random number generator
    rng = ut.rng(seed)

    # Constants
    time0 = np.arange(0, ut.year()/4, 25)
    cols  = ["yaw", "pitch", "roll"]
    N     = len(quarter)
    n     = len(time0)

    # Create data frame and store default time0 for fit
    df_1  = pd.DataFrame(); df1 = pd.DataFrame()
    df_2  = pd.DataFrame(); df2 = pd.DataFrame()
    df_3  = pd.DataFrame(); df3 = pd.DataFrame()
    df_4  = pd.DataFrame(); df4 = pd.DataFrame()
    A = np.zeros((len(quarter), 4))

    # Load wheel offloadings
    if wheel_offloading:
        idir = Path(os.getenv("PLATO_PROJECT_HOME")) / 'inputfiles'
        filename_dir = idir / 'TED_dir_prime_2021jan.ftr'
        filename_rot = idir / 'TED_rot_prime_2021jan.ftr'
        # Check if they should be downloaded
        if not filename_dir.is_file() or not filename_rot.is_file():
            print(f'Downloading Prime reaction wheel offloading models..\n')
            ut.downloadFromFTP(filename_dir.name, idir, 'plato')
            ut.downloadFromFTP(filename_rot.name, idir, 'plato')
        df_dir = pd.read_feather(filename_dir)
        df_rot = pd.read_feather(filename_rot)
    
    # Loop over each quarter

    for Q in range(quarter[0]-1, quarter[-1]):

        # Time column
        t0 = round(ut.year()/4 * Q)
        time = t0 + np.arange(0, n) * 25
        df1["time"] = time
        df2["time"] = time
        df3["time"] = time
        df4["time"] = time

        # Create model for each camera group

        for col in cols:

            # Generate linear model
            if model == 'linear':
                a = 1.3 * 15      
                if col == "roll":
                    df1[col] = np.zeros(n)
                else:
                    df1[col] = np.linspace(0, a, n)

            # Generate a random 2nd order polynomial
            else:
                # NOTE these parameters has been compared to Prime TED
                a = rng.uniform(-10, 10) * 1e-14 * ampl / 12
                b = rng.uniform(-15, 15) * 1e-7  * ampl / 12
                # Secure that c (the y offset) is always zero
                c = 0
                # Make sure that a and b always has opposite signs
                if np.sign(a) == np.sign(b): b *= -1
                # Get model fit
                aa = np.abs(a/5)
                bb = np.abs(b/5)
                poly1 = np.array([a, b, c])
                poly2 = np.array([a+rng.uniform(-aa, aa), b+rng.uniform(-bb, bb), c])
                poly3 = np.array([a+rng.uniform(-aa, aa), b+rng.uniform(-bb, bb), c])
                poly4 = np.array([a+rng.uniform(-aa, aa), b+rng.uniform(-bb, bb), c])
                df1[col] = np.polyval(poly1, time0)
                df2[col] = np.polyval(poly2, time0)
                df3[col] = np.polyval(poly3, time0)
                df4[col] = np.polyval(poly4, time0)

            # Add reaction wheel offloadings
            if wheel_offloading:
                dex = rng.integers(1, 24, 1)[0]
                t = np.linspace(time0[0], time0[-1], len(df_dir.time))
                # Apply a small random amplitude (+-10%) for variation
                ampl += ampl * rng.uniform(-0.1, 0.1)
                # Directional (yaw, picth)
                if col in ['yaw', 'pitch']:
                    a = df_dir[f'ncam{dex}'] * ampl / 1000
                else:
                    a = df_rot[f'ncam{dex}'] * ampl / 1000
                spline = make_interp_spline(t, a, k=2)
                wheel  = spline(time0)            
                df1[col] += wheel
                df2[col] += wheel
                df3[col] += wheel
                df4[col] += wheel
                
        # File to save
        df_1 = pd.concat([df_1, df1])
        df_2 = pd.concat([df_2, df2])
        df_3 = pd.concat([df_3, df3])
        df_4 = pd.concat([df_4, df4])
        
        # Array with all amplitudes
        A[Q-quarter[0],0] = Q+1
        A[Q-quarter[0],1] = df1.yaw.max()   - df1.yaw.min()
        A[Q-quarter[0],2] = df1.pitch.max() - df1.pitch.min()
        A[Q-quarter[0],3] = df1.roll.max()  - df1.roll.min()

    # Show amplitudes
    
    if table:
        print('TED model amplitudes [arcsec]')
        names = ['Quarter', 'A_yaw', 'A_pitch', 'A_roll']
        da = pd.DataFrame(A, columns=names)
        da = da.sort_values(['Quarter'])
        da = da.astype({'Quarter':np.int32})
        da = da.reset_index(drop=True)
        print(da)
        print('\nMaximum TED amplitudes [arcsec]')
        print(da.max()[1:])
        
    # Plot model
    
    fig, ax = plt.subplots(3,1,figsize=(9, 6))

    # Plots
    for i, col in zip(range(3), cols):
        ax[i].plot(df_1["time"]/ut.day(), df_1[col], '-', c='b')
        ax[i].plot(df_2["time"]/ut.day(), df_2[col], '-', c='g')
        ax[i].plot(df_3["time"]/ut.day(), df_3[col], '-', c='orange')
        ax[i].plot(df_4["time"]/ut.day(), df_4[col], '-', c='r')
        ax[i].axhline(y=0, linestyle=':', color='k')
        Qday = ut.year()/86400/4
        for k in range(N-1):
            ax[i].axvline(x=quarter[k]*Qday, linestyle='--', color='k')

    # Settings
    ax[2].set_xlabel("Time [days]")
    ax[0].set_ylabel("Yaw [arcsec]")
    ax[1].set_ylabel("Pitch [arcsec]")
    ax[2].set_ylabel("Roll [arcsec]")
    for i in range(3):
        ax[i].set_xlim(df_1.time.min()/ut.day(), df_1.time.max()/ut.day())

    # Layout
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    plt.tight_layout(h_pad=0.2, w_pad=0)
        
    # Plot figure above     
    if plot: plt.show()

    # Save data in one big drift ascii file for PlatoSim
    if ofile:
        df_1.to_csv(f'{ofile[:-4]}_group1.csv', sep=" ", header=False, index=False)
        df_2.to_csv(f'{ofile[:-4]}_group2.csv', sep=" ", header=False, index=False)
        df_2.to_csv(f'{ofile[:-4]}_group3.csv', sep=" ", header=False, index=False)
        df_4.to_csv(f'{ofile[:-4]}_group4.csv', sep=" ", header=False, index=False)
        fig.savefig(f'{ofile[:-4]}.png', bbox_inches='tight', dpi=200)




            
def getACS(time, rms=[0.038, 0.038, 0.040], ofile=False, plot=False):

    """Generate a Attitude and orbit Control System (ACS) jitter file.
   
    This function generates a complete ACS model returned in euler angles.
    
    Paramters
    ---------
    quarter : range
        Range of quarters (e.g. range(1,8) for Q1-Q8).
    ofile : bool, str
        Parse string to save the model to a ascii file.
    plot : bool
        True will make a plot of the models.
        
    Return
    ------
    Output file if requested.
    """

    # Generate red noise components

    tau = time[1] - time[0]
    df = pd.DataFrame()
    df['t'] = time
    for i, n, in zip(range(3), ['x', 'y', 'z']):
        print(f'Generating {n} red noise component using {rms[i]}')
        df[n] = modelRedNoise(time, timescale=np.array([tau]), varscale=np.array(rms[i]))

    # Plot time series and PSD

    fig1, ax = pt.plotYawPitchRollTimeSeries(df.t/3600, np.array([df.x, df.y, df.z]),
                                             units=["hours", "arcsec"])
    fig2, ax = pt.plotYawPitchRollPSD(df.t, np.array([df.x, df.y, df.z]))
    if plot: plt.show()
        
    # Save data in one jitter file
    
    if ofile:
        df.to_csv(ofile, sep=" ", header=False, index=False)
        fig1.savefig(f"{ofile[:-4]}_timeseries.png", bbox_inches='tight', dpi=200)
        fig2.savefig(f"{ofile[:-4]}_psd.png", bbox_inches='tight', dpi=200)
    
    



def getDataGaps(time, quarter=range(1,9), seed=None, ofile=False, plot=False):

    """Function to create data gaps in time series.

    All time gaps are based on knowledge from the Kepler mission.
    The following time gaps are considered:

      - Quarterly rolls
      - Monthly data donlinks
      - Loss of fine guidance
      - Save mode events

    The Quarterly rolls introduce a time gap of 1-2 days happening every
    three months (hence four times in a year). These are are needed in 
    order to align the spacecraft's solar panels towards the Sun. Note that
    the PES called PRE takes into account for the error in every pointing
    from quarter to quarter. 

    Montly data downlinks is needed during the mission. During these events
    the spacecraft changed its orientation once a month to point at Earth
    and downlink the last month of data. This caused a gap in the time series
    data, as the spacecraft can not collect while it was downlinking. When 
    the spacecraft returns to the regular pointing, its motion induced what
    is known as a “thermal transient.” This means that the telescope components
    and detector electronics underwent a temperature change, and the electron
    count reading was temporarily increased. A change in temperature can slightly
    change the camera focus. This manifests in the data as a downward slope 
    caused by “reheating,” while the flux returns to its previous level. 

    Loss of fine guidance and coarse pointing events which results in
    a lower photometric precision. Typically these cadences are not
    suggested for use in photometry. Since these event do not introduce
    thermal transients in the light curve, they can simply be removed 
    after the simulations have been generated by the user. 

    Safe modes (Kepler Data Characteristics Handbook, Section 4.2)
    are another type of thermal transient that appears in Kepler data. 
    A safe mode occured when the telescope temporarily shut off operation
    due to an unexpected event, usually caused by an issue with the 
    detector electronics. Typically a positive flux jump happens each event
    (probably due to an increased gain) an exponentially decrease to the
    count level before the event over a period of a few days.
    """

    # Random number generator
    rng = ut.rng(seed)

    # Storage arrays
    roll    = np.zeros_like(time, dtype=bool)
    station = np.zeros_like(time, dtype=bool)
    wheel   = np.zeros_like(time, dtype=bool)
    jitter  = np.zeros_like(time, dtype=bool)
    safe    = np.zeros_like(time, dtype=bool)

    # QUARTERLY ROLLS (~91 days)

    quarter       = np.array(quarter)
    roll_period   = ut.quarter()
    roll_duration = 1.5
    roll_anomaly  = 0.5
    roll_events   = quarter
    n_roll        = len(roll_events)
    roll_event0   = np.zeros(n_roll)
    roll_event1   = np.zeros(n_roll)

    for i, Q in zip(range(n_roll), roll_events):
        roll_gap = roll_duration + rng.uniform(-roll_anomaly, roll_anomaly)       # [d]
        roll_event0[i] = (roll_period * Q - roll_gap/2) * ut.day()                 # [s]
        roll_event1[i] = (roll_period * Q + roll_gap/2) * ut.day()                 # [s]
        roll_dex       = np.where((time>=roll_event0[i]) & (time<=roll_event1[i]))[0]
        roll[roll_dex] = True

    # STATION KEEPING MANOEUVRES (~30 days)

    station_period   = ut.quarter()/3.
    station_duration = 2/24
    station_anomaly  = 0.5/24
    # Remove times during quarter rolls
    x = np.arange(station_period, roll_period*quarter[-1], station_period)
    station_events   = np.take(x, np.setdiff1d(np.arange(len(x)), np.arange(2, len(x), 3)))
    n_station        = len(station_events)
    station_event0   = np.zeros(n_station)
    station_event1   = np.zeros(n_station)

    for i, S in zip(range(n_station), station_events):
        station_gap = station_duration + rng.uniform(-station_anomaly, station_anomaly)
        station_event0[i] = (S - station_gap/2) * ut.day()
        station_event1[i] = (S + station_gap/2) * ut.day()
        station_dex       = np.where((time>=station_event0[i]) & (time<=station_event1[i]))[0]
        station[station_dex] = True

    # REACTION WHEEL OFFLOADINGS (~3 days)

    # Period is from DFT of Prime data
    # wheel_period   = 1/0.334
    # wheel_duration = 0.05/24.
    # wheel_anomaly  = 0.01/24.
    # wheel_events   = np.arange(wheel_period, roll_period*quarter[-1], wheel_period)
    # wheel_events = np.array(wheel_events)
    # n_wheel        = len(wheel_events)
    # wheel_event0   = np.zeros(n_wheel)
    # wheel_event1   = np.zeros(n_wheel)
    
    # for i, W in zip(range(n_wheel), wheel_events):
    #     wheel_gap = wheel_duration + rng.uniform(-wheel_anomaly, wheel_anomaly)
    #     wheel_event0[i] = (W - wheel_gap/2) * ut.day()
    #     wheel_event1[i] = (W + wheel_gap/2) * ut.day()
    #     wheel_dex       = np.where((time>=wheel_event0[i]) & (time<=wheel_event1[i]))[0]
    #     wheel[wheel_dex] = True
        
    # LOSS OF FINE GUIDANCE

    jitter_freq = 120
    jitter_offset = rng.uniform(0, jitter_freq)
    t = (quarter[0] - 1) * roll_period - jitter_offset
    jitter_duration = 0.2/24.
    jitter_anomaly  = 0.1/24.
    jitter_events   = []

    while t < quarter[-1] * roll_period:            
        jitter_event = rng.poisson(lam=jitter_freq)
        t += jitter_event
        jitter_events.append(t)

    n_jitter = len(jitter_events)
    jitter_event0 = np.zeros(n_jitter)
    jitter_event1 = np.zeros(n_jitter)
    event0 = roll_event0 # np.concatenate((roll_event0, link_event0))
    event1 = roll_event1 # np.concatenate((roll_event1, link_event1))

    for i, J in zip(range(n_jitter), jitter_events):
        jitter_gap = jitter_duration + rng.uniform(-jitter_anomaly, jitter_anomaly)
        jitter_event0[i] = (J - jitter_gap/2) * ut.day()
        jitter_event1[i] = (J + jitter_gap/2) * ut.day()
        jitter_dex       = np.where((time>=jitter_event0[i]) & (time<=jitter_event1[i]))[0]
        jitter[jitter_dex] = True

        for j in range(len(event0)):
            if ((jitter_event0[i] >= event0[j]) and (jitter_event1[i] <= event1[j]) or
                (jitter_event0[i] <= event0[j]) and (jitter_event1[i] >= event1[j]) or
                (jitter_event0[i] <= event0[j]) and (jitter_event1[i] >= event0[j]) or
                (jitter_event0[i] <= event1[j]) and (jitter_event1[i] >= event1[j])):
                jitter_event0[i] += event1[j] - event0[j]
                jitter_event1[i] += event1[j] - event0[j]

    # SAFE MODE EVENTS

    safe_freq = 270
    safe_offset = rng.uniform(0, safe_freq)
    t = (quarter[0] - 1) * roll_period - safe_offset
    safe_duration = 1
    safe_anomaly  = 12/24.
    safe_events   = []

    while t < quarter[-1] * roll_period:            
        safe_event = rng.poisson(lam=safe_freq)
        t += safe_event
        safe_events.append(t)
        
    n_safe = len(safe_events)
    safe_event0 = np.zeros(n_safe)
    safe_event1 = np.zeros(n_safe)
    event0 = np.concatenate((roll_event0, jitter_event0))
    event1 = np.concatenate((roll_event1, jitter_event1))        

    for i, S in zip(range(n_safe), safe_events):
        safe_gap = safe_duration + rng.uniform(-safe_anomaly, safe_anomaly)
        safe_event0[i] = (S - safe_gap/2) * ut.day()
        safe_event1[i] = (S + safe_gap/2) * ut.day()
        safe_dex       = np.where((time>=safe_event0[i]) & (time<=safe_event1[i]))[0]
        safe[safe_dex] = True
        
        for j in range(len(event0)):
            if ((safe_event0[i] >= event0[j]) and (safe_event1[i] <= event1[j]) or
                (safe_event0[i] <= event0[j]) and (safe_event1[i] >= event1[j]) or
                (safe_event0[i] <= event0[j]) and (safe_event1[i] >= event0[j]) or
                (safe_event0[i] <= event1[j]) and (safe_event1[i] >= event1[j])):
                safe_event0[i] += 2*(event1[j] - event0[j])
                safe_event1[i] += 2*(event1[j] - event0[j])
        
    # Show figure

    if n_roll != 0:

        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.axhline(y=0, linestyle=':', color='k')

        for i in range(n_roll):
            ax_roll = ax.axvspan(roll_event0[i]/ut.day(), roll_event1[i]/ut.day(),
                                 color='b', alpha=0.5)
            if i == n_roll-1: ax_roll.set_label('Quarterly rolls')

        for i in range(n_station):
            ax_station = ax.axvspan(station_event0[i]/ut.day(), station_event1[i]/ut.day(),
                                    color='m', alpha=0.5)
            if i == n_station-1: ax_station.set_label('Station keeping')

        # for i in range(n_wheel):
        #     ax_wheel = ax.axvspan(wheel_event0[i]/ut.day(), wheel_event1[i]/ut.day(),
        #                           color='g', alpha=0.5)
        #     if i == n_wheel-1: ax_wheel.set_label('Wheel offloadings')
            
        for i in range(n_jitter):
            ax_jitter = ax.axvspan(jitter_event0[i]/ut.day(), jitter_event1[i]/ut.day(),
                                   color='orange', alpha=0.5)
            if i == n_jitter-1: ax_jitter.set_label('Loss of fine guidance')

        for i in range(n_safe):
            ax_safe = ax.axvspan(safe_event0[i]/ut.day(), safe_event1[i]/ut.day(),
                                 color='r', alpha=0.5)
            if i == n_safe-1: ax_safe.set_label('Safe mode events')

        # Labels
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Arb.")
        ax.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 1.4))

        # Layout
        ax.set_yticklabels([])
        ax.set_xlim(time[0]/ut.day(), time[-1]/ut.day()+2)
        ax.set_ylim(-1,1)
        plt.tight_layout()

    # Show plot
    if plot: plt.show()

    # Create pandas data frame for different flags    

    df = pd.DataFrame()
    df["time"]    = time
    df["roll"]    = roll
    df["station"] = station
    #df["wheel"]   = wheel
    df["jitter"]  = jitter
    df["safe"]    = safe
    df["all"]     = roll + station + jitter + safe
        
    # Compute event times

    t0 = np.concatenate((roll_event0, station_event0, jitter_event0, safe_event0))
    t1 = np.concatenate((roll_event1, station_event1, jitter_event1, safe_event1))
    dt = pd.DataFrame({'t0':t0, 'td':t1-t0})
    
    # Save file (and plot) if requested
    
    if ofile:
        df.reset_index(drop=True, inplace=True)
        df.to_feather(ofile)
        dt.to_feather(f"{ofile[:-4]}.tab")
        fig.savefig(f"{ofile[:-4]}.png", bbox_inches='tight', dpi=200)

    # That's it!
    
    return df, t0, t1-t0





def temperatureTransients(time, t0, td, tempCCD=203.15, tempConst=10, gapSize=0.1,timeSpan=30,
                          timeScale=False, amplitude=False, ofile=False, plot=False):

    """Function to model detector temperature transients.

    Parameters
    ----------
    time : ndarray
        Time points with gaps [s]
    tempCCD : float
        Average CCD temperature [K]
    gapSize : float, ndarray
        Minimum duration to detect the gaps [days]
    timeSpan : float, ndarray
        Time duration to model each transient [days]
        Parameter used to minimize run time.
    timeScale : float, ndarray
        Time scale duration of the flare(s) [days]
        Linear model is used if False.
    amplitude : float, ndarray
        Temperature amplitude of the transient(s) [K]
        Linear model is used if False.
    plot : bool
        Flag to plot the generated model.

    NOTE: The following code snippet removes the data gaps from the df:
    >> df0 = df.drop(df[(df.roll == True)   | (df.link == True) |
                        (df.jitter == True) | (df.safe == True)].index)
    """
    
    # Secure numpy syntax
    if isinstance(time, pd.Series):
        time = time.to_numpy()
    if isinstance(t0, list):
        t0 = np.array(t0)
    if isinstance(td, list):
        td = np.array(td)
        
    # Create temperature array to write to    
    time = time / 86400.
    temp = np.zeros_like(time)

    # Convert to days
    timeGap0 = t0   / 86400.
    tdurGap  = td   / 86400.
    timeGap1 = timeGap0 + tdurGap
    
    # Unit parameters
    cadence = np.diff(time)[0]
    ndex    = int(timeSpan / cadence)
    n       = len(t0)

    # Indices for start and end of each event
    timeDex0 = [np.argmin(np.absolute(time - timeGap0[i])) for i in range(n)]
    timeDex1 = [np.argmin(np.absolute(time - timeGap1[i])) for i in range(n)]
    
    # Linear CCD temperature dependece with gap size
    if not timeScale:
        timeScale = 1/tempConst * tdurGap
    
    if not amplitude:
        amplitude = tempConst * tdurGap

    # Secure that a single event works
    try: len(timeGap1)
    except: timeGap1 = [timeGap1]
    try: len(timeScale)
    except: timeScale = [timeScale]
    try: len(amplitude)
    except: amplitude = [amplitude]

    # Loop over each transient event
    for i in range(n):

        # Time array during transient event
        tn = time / timeScale[i]

        # Model parameters of transients
        a0 = 0.689
        a1 = -1.6
        b0 = 1 - a0
        b1 = -0.2783
        
        # Secure last event is within time series
        if timeDex1[i]+ndex > len(time):
            timeDex2 = len(time)-1
        else:
            timeDex2 = timeDex1[i] + ndex

        # Loop over every time-step in the transient time interval
        for j,k in zip(range(timeDex1[i], timeDex2), range(len(tn))):
            #temp[j] += (a0 * np.exp(a1 * tn[k]) + b0 * np.exp(b1 * tn[k]) * amplitude[i])
            temp[j] += (b0 * np.exp(b1 * tn[k]) * amplitude[i])

            # Add amplitude for overlapping events
            if (temp[timeDex0[i]] > tempCCD+0.001) and (j==timeDex1[i]):
               amplitude[i] += temp[timeDex0[i]]

    # Add CCD zero point temperature
    temp += np.ones_like(temp) * tempCCD
               
    # Plot if requested
    
    fig, ax = plt.subplots(figsize=(9, 3.5))

    for i in range(n):
        ax.axvspan(timeGap0[i], timeGap1[i], color='b', alpha=0.2)
    ax.plot(time, temp, '-', lw=1, c='deeppink')
    ax.set_xlabel('Time [d]')
    ax.set_ylabel('CCD temperature [K]')
    ax.set_xlim(np.min(time), np.max(time))
    plt.tight_layout()

    # Plot figure above
    if plot: plt.show()        

    # Save data if requested
    if ofile:
        np.savetxt(ofile, np.transpose([time*ut.day(), temp]), fmt=['%.1f', '%.6f'])
        fig.savefig(f"{ofile[:-4]}.png", bbox_inches='tight', dpi=200)
        
    return temp






def getGain(sigma=3, gain0CCD=False, seed=None, ofile=False, plot=False):

    """ointing Reproducibility Error (PRE) in PLM reference frame.
    
    TODO under construction!

    Paramters
    ---------

    Return
    ------
    """

    
    # Initialise random generator
    rng = np.random.default_rng()
    
    # Total number of CCD x 2 halves

    nCCD = 104
    
    # Find distribution within 3 sigma of req.

    if not gain0CCD: gain0CCD = 1.8
    gainRef = gain0CCD / sigma
    deltaGainF = np.array([rng.normal(0, gainRef) for i in range(nCCD)])
    deltaGainE = np.array([rng.normal(0, gainRef) for i in range(nCCD)])
    
    # Plot distributions
    
    t *= 3600
    b *= 3600
    y = t/sigma
    z = 3 * y
    x = np.abs(b/sigma - z)
    xx = np.linspace(-10*x, 10*x, 1000)

    ra0  = (df1.RA-ra)   * 3600 / 18
    dec0 = (df1.Dec-dec) * 3600 / 18

    fig, ax = plt.subplots(1, 2, figsize=(9,4))

    ax[0].set_title(f'PRE distributions at {sigma}$\sigma$')        
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, x)*100, '-', c='b', label='Trans.')
    ax[0].plot(xx, scipy.stats.norm.pdf(xx, 0, z)*100, '-', c='m', label='Rot.')
    ax[0].set_xlabel('Platform pointing errors in FPA [pixel]')
    ax[0].set_ylabel('Probability (PDF) [\%]')
    ax[0].set_xlim(xx[0], xx[-1])
    ax[0].legend()

    ax[1].grid(zorder=0)
    ax[1].plot(0, 0, 'k*', ms=10, zorder=2)
    for i in range(len(ra0)):
        ax[1].scatter(ra0[i], dec0[i], marker=f'${i+1}$', s=50, alpha=0.8, zorder=2)
    ax[1].set_title('Distribution on Sky')
    ax[1].set_xlabel('RA [pixel]')
    ax[1].set_ylabel('Dec [pixel]')
    ax[1].set_aspect('equal', adjustable='box')

    lim = np.max([np.max(np.abs(ra0)), np.max(np.abs(dec0))])
    lim += lim/10.
    ax[1].set_xlim(-lim, +lim)
    ax[1].set_ylim(-lim, +lim)        
    plt.tight_layout()

    # Plot figure above 
    
    if plot: plt.show()
        
    # Save file with relative pointing errors [deg]
    
    if ofile:
        df.to_csv(ofile, sep=" ", header=False, index=False)
        fig.savefig(f"{ofile[:-4]}.png", bbox_inches='tight', dpi=200)

    # That's it!
    
    return PRE

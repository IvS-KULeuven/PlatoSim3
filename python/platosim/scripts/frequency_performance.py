"""
Visualise Frequency Performance
=============================

A simple script to create some visualisations for the
PLATO Mission Performance Report.
"""
import os
import re
from typing import Union
from itertools import islice

from scipy import stats
from scipy.stats import binned_statistic
from scipy.signal import periodogram, resample

import matplotlib.pyplot as plt

import h5py
import numpy as np

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

__author__ = 'Sami Niemi (sami.matias.niemi@esa.int)'
__version__ = 0.1


def compute_double_sided_PSD(time_series: np.ndarray,
                             time_interval: Union[int, float] = 25,
                             detrend: Union[bool, str, int] = False,
                             scipy: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
    time = np.linspace(start=0,
                       stop=len(time_series),
                       num=len(time_series)) * time_interval

    if scipy:
        frequencies, psd = periodogram(time_series,
                                       fs=1 / time_interval,
                                       window='boxcar',
                                       nfft=None,
                                       detrend=detrend,
                                       return_onesided=False,
                                       scaling='density',
                                       axis=-1)
    else:
        if detrend:
            not_nan_ind = ~np.isnan(time_series)
            if isinstance(detrend, str) and 'linear' in detrend:
                print('Applying linear detrending...')

                m, b, _, _, _ = stats.linregress(
                    time[not_nan_ind], time_series[not_nan_ind])

                time_series = time_series - (m * time + b)
            elif isinstance(detrend, int):
                print(f'Applying polymial order {detrend} detreding...')
                model = np.polyfit(time[not_nan_ind],
                                   time_series[not_nan_ind],
                                   detrend)
                predicted = np.polyval(model, time)

                time_series = time_series - predicted
            else:
                raise NotImplementedError

        fft_of_time_series = np.fft.fft(time_series)

        # scaling - normally this would be 2 / (frequency * window_compensation)
        # here we compute a double sided PSD, so we drop the factor 2
        # instead of frequency we have time interval i.e. 1/frequency
        # because we do not use a window function, instead of having
        # a sum of squared samples of window function, we simply use the
        # lenght of the input time series. Thus, the scaling here is
        # time_interval / length of the data
        # for more details, see
        # https://dsp.stackexchange.com/questions/32187/what-should-be-the-correct-scaling-for-psd-calculation-using-tt-fft
        # https://stackoverflow.com/questions/22338415/scipy-periodogram-terminology-confusion
        scale = time_interval / time_series.shape[-1]

        # psd is the absolute ps (fft of time series) squared
        psd = scale * np.abs(fft_of_time_series)**2

        frequencies = np.fft.fftfreq(time_series.shape[-1], time_interval)

    idx = np.argsort(frequencies)
    psd = psd[idx]
    frequencies = frequencies[idx]

    return frequencies, psd


def read_data_from_file(filename, exptime=21):
    with h5py.File(filename, "r") as f:
        print(f)
        # handle L1 output
        if "LIGHTCURVE" in filename:
            # L1 gives e-/exposure
            # ignore the first, it's alwasy bad..
            lc = f['FLUX_TS']['FLUX'][1:]
            status = f['FLUX_TS']['STATUS'][1:]
        else:
            print('star ids: %s' % list(f['Photometry']['Lightcurves']))
            # make e-/exposure, scale by exptime
            lc = f['Photometry']['Lightcurves']['starID1']['estimatedFlux'][()]*exptime
            status = np.array([0]*len(lc))
    return lc, status


def equal_obs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def rebin1d(array, n):
    nr = int(float(array.shape[0]) / float(n))
    return (np.reshape(array, (n, nr))).sum(1)


def create_performance_figure(data, EoL,
                              freq_break=20e-6,
                              min_freq=3e-6,
                              max_freq=40e-3,
                              residual_noise_floor=0.68e-6,
                              random_noise_level=3.0e-6,
                              residual_noise_top=50e-6):
    """

    Args:
        data ([np.asarray]):
        EoL Bool
        freq_break ([float], optional): [description]. Defaults to 20e-6.
        min_freq ([float], optional): [description]. Defaults to 3e-6.
        max_freq ([type], optional): [description]. Defaults to 40e-3.
        residual_noise_floor ([float], optional): [description]. Defaults to 0.68e-6.
        random_noise_level ([float], optional): [description]. Defaults to 3.0e-6.
        residual_noise_top ([float], optional): [description]. Defaults to 50e-6.
        output_file ([str], optional): name of the output file.
    """
    if EoL:
        output_file='mission_performance_asd_EoL.png'
    else:
        output_file='mission_performance_asd.png'

    freq, psd = data

    fig = plt.figure(figsize=(12, 8))

    # plot the ASD
    plt.plot(freq, psd, 'gray', alpha=0.5)

    # plot binned ASD
    m = 10
    p = int(freq.size / m)
    num = rebin1d(freq[0:p*m], p) / float(m)
    binned = rebin1d(psd[0:p*m], p) / float(m)

    plt.plot(num[1:], binned[1:], 'black',  lw=2)

    # residual error line
    plt.hlines(y=residual_noise_floor, xmin=freq_break, xmax=max_freq,
               colors='red', linestyles='-')
    # random noise line
    plt.hlines(y=random_noise_level, xmin=min_freq, xmax=max_freq,
               colors='magenta', linestyles='-')
    # slope line from residual to random top level
    x_values = np.linspace(min_freq, freq_break, 2)
    y_values = np.linspace(residual_noise_top, residual_noise_floor, 2)
    plt.plot(x_values, y_values, color='red', linestyle='-')

    # dashed  guide lines
    plt.vlines(x=freq_break, ymin=1e-8, ymax=residual_noise_floor,
               linestyles='dashed', colors='blue')
    plt.vlines(x=max_freq, ymin=1e-8, ymax=residual_noise_floor,
               linestyles='dashed', colors='blue')
    plt.vlines(x=min_freq, ymin=1e-8, ymax=residual_noise_top,
               linestyles='dashed', colors='blue')
    plt.hlines(y=residual_noise_floor, xmin=1e-10, xmax=freq_break,
               linestyles='dashed', colors='blue')
    plt.hlines(y=residual_noise_top, xmin=1e-8, xmax=min_freq,
               linestyles='dashed', colors='blue')

    # texts
    freq_units = r' $\frac{\mathrm{ppm}}{\sqrt{\mu\mathrm{Hz}}}$'
    top_text = f'{int(residual_noise_top*1e6)}' + freq_units
    plt.text(x=min_freq, y=residual_noise_top*1.25,
             s=top_text,
             ha='center')
    random_noise_text = f'Random Noise\n(incl. photonic stellar reference noise)\n{random_noise_level*1e6}' + freq_units
    plt.text(x=5e-4, y=random_noise_level*1.25,
             s=random_noise_text,
             ha='center')
    residual_error_text = f'Residual Errors\n{round(residual_noise_floor*1e6, 2)}' + \
        freq_units
    plt.text(x=5e-4, y=residual_noise_floor*1.25,
             s=residual_error_text,
             ha='center')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1e-6, 1e-1)
    plt.ylim(1e-7, 1e-4)

    # modify x tick points
    ticks = [1e-6, min_freq, 1e-5, freq_break,
             1e-4, 1e-3, 1e-2, max_freq, 1e-1]
    labels = ['$10^{-6}$', str(int(min_freq*1e6)) + '$\mu$Hz', '$10^{-5}$',
              str(int(freq_break*1e6)) + '$\mu$Hz',
              '$10^{-4}$', '$10^{-3}$', '$10^{-2}$',
              str(int(max_freq*1e3)) + 'mHz', '$10^{-1}$']
    plt.xticks(ticks=ticks, labels=labels)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Amplitude Spectral Density $(\mu\mathrm{Hz})^{-\frac{1}{2}}$')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)


def find_all_files(parent_folder):
    """
    _summary_

    Args:
        parent_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_of_files = {}

    pattern = r"Ncam\d+\.\d+"

    for (dirpath, _, filenames) in os.walk(parent_folder):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                list_of_files[filename] = {}
                list_of_files[filename]['file'] = os.sep.join(
                    [dirpath, filename])

                match = re.search(pattern, filename)
                if match:
                    list_of_files[filename]['camera'] = (match.group())

    return list_of_files


def visualise_light_curves(time, data):
    plt.plot(time, data)


def combine_light_curves(all_data, normalize=True):
    """
    _summary_

    Args:
        all_data (_type_): _description_
        normalize (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    lcs = []
    for key in all_data:
        lc = all_data[key]['lc']

        if normalize:
            lc /= np.nanmean(lc)

        lcs.append(lc)

    mean_lc = np.nanmean(lcs, axis=0)

    if normalize:
        mean_lc /= np.nanmean(mean_lc)

    return mean_lc


def bin_1d(data, dt=25., per_hour=3600):
    """
    _summary_

    Args:
        data (_type_): _description_
        dt (_type_, optional): _description_. Defaults to 25..
        per_hour (int, optional): _description_. Defaults to 3600.

    Returns:
        _type_: _description_
    """
    time = np.linspace(0, len(data)*dt, len(data))
    bin_means, bin_edges, _ = binned_statistic(time, data,
                                               statistic='mean',
                                               bins=int(time[-1] / per_hour))
    bin_edges /= per_hour

    return bin_means, bin_edges


def main(parent_folder, EoL=True):
    return

if __name__ == "__main__":
    parent_folder = '/Users/jmcc/Dropbox/data/plato/PlatoSim3_simulations/cam24_test/000001245'
    os.chdir(parent_folder)

    #main(parent_folder)
    cont = 0
    EoL = True

    files = find_all_files('.')

    for key in files:
        lc_in, status_in = read_data_from_file(files[key]['file'])

        # apply masking
        mask = status_in > 0
        lc_in[mask] = np.nan
        files[key]['lc'] = lc_in

        time = np.linspace(
            0, len(files[key]['lc'])*25, len(files[key]['lc']))

        visualise_light_curves(time/3600, files[key]['lc'])

    plt.ylabel('flux [electrons]')
    plt.xlabel('time [hours]')
    plt.tight_layout()
    plt.show()

    print(f'{len(files.keys())} N-CAMs')
    if EoL:
        # Getting the first 22 key-value pairs
        files = dict(islice(files.items(), 22))
    print(f'selected {len(files.keys())} N-CAMs')

    mean_lc = combine_light_curves(files)

    bin_means, bin_edges = bin_1d(mean_lc)

    plt.plot(time/3600, mean_lc, 'k.', label='mean lc')
    #plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=2,
    #           label='binned statistic of data')
    plt.plot(bin_edges[:-1]+0.5, bin_means, 'r.', label='bin 1h')

    # do a linear detrend before calculating NSR
    time = np.linspace(start=0,
                       stop=len(bin_means),
                       num=len(bin_means))
    not_nan_ind = ~np.isnan(bin_means)
    print('Applying linear detrending...')
    m, b, _, _, _ = stats.linregress(time[not_nan_ind], bin_means[not_nan_ind])
    bin_means_detrend = bin_means / (m * time + b)

    nsr = bin_means_detrend.std() * 1e6
    print(f'NSR in 1h is {round(nsr, 2)}')

    plt.plot(time, bin_means_detrend, 'b.', label='linear detrend bin 1h')
    plt.legend()
    plt.show()

    freq, psd = compute_double_sided_PSD(
        mean_lc, time_interval=25, scipy=False, detrend='linear')

    # systematic noise is about 9ppm, all random is about 40ppm and total about 42ppm
    # currently fudgeted, should compute from the actual noise
    psd *= 0.05

    create_performance_figure([freq, psd], EoL,)

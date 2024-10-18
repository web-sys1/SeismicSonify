#!/usr/bin/env python

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import MethodType

import colorcet as cc
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obspy

from matplotlib import font_manager, mlab
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import ScalarFormatter
from obspy.signal.util import next_pow_2
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy import signal
from scipy.signal import ShortTimeFFT, windows, iirnotch, filtfilt
import scipy.integrate as integrate 
from scipy.interpolate import make_interp_spline
from scipy.fftpack import fft, ifft
from scipy.linalg import svd


from . import __version__
    

# Add Tex Gyre Heros and JetBrains Mono to Matplotlib
for font_path in font_manager.findSystemFonts(
    str(Path(__file__).resolve().parent / 'fonts')
):
    font_manager.fontManager.addfont(font_path)

LOWEST_AUDIBLE_FREQUENCY = 20  # [Hz]
HIGHEST_AUDIBLE_FREQUENCY = 20000  # [Hz]

AUDIO_SAMPLE_RATE = 44100  # [Hz]

PAD = 60  # [s] Extra data to download on either side of requested time slice

# [px] Output video resolution options (width, height)
RESOLUTIONS = {
    'crude': (640, 360),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '2K': (2560, 1440),
    '4K': (3840, 2160),
}

FIGURE_WIDTH = 7.7  # [in] Sets effective font size, basically

# For spectrograms
REFERENCE_PRESSURE = 20e-6  # [Pa]
REFERENCE_VELOCITY = 1  # [m/s]
REFERENCE_ACCELERATION = 9.806 # g=gm/r^2 | [m/s**2] or [m/s/s]: Peak acceleration due to gravity

MS_PER_S = 1000  # [ms/s]

# Colorbar extension triangle height as proportion of colorbar length
EXTENDFRAC = 0.0000

def seisPlotter(
    type='local',
    file='',
    fdsn_client='IRIS',
    network='',
    station='',
    location='*',
    channel='',
    inv_file=None,
    starttime=None,
    endtime=None,
    remove_response=None,
    pre_filt=None,
    filter_type='bandpass',
    h_freq=0.88,
    l_freq=None,
    wf_freq=None,
    freqmin=None,
    freqmax=None,
    speed_up_factor=200,
    fps=1,
    rescale=1e6,
    resolution='4K',
    output_dir=None,
    spec_win_dur=5,
    db_lim='smart',
    unit_scale='auto',
    spectral_scaling='density',
    cmap='inferno',
    switchaxes=False,
    num=None,
    log=False,
    specOpts=None,
    fName='',
    utc_offset=None,
):
    r"""
    Produce an animated spectrogram with a soundtrack derived from sped-up
    seismic or infrasound data.
    Args:
        network (str): SEED network code
        station (str): SEED station code
        channel (str): SEED channel code
        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time of
            animation (UTC)
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time of
            animation (UTC)
        location (str): SEED location code
        freqmin (int or float): Lower bandpass corner [Hz] (defaults to 20 Hz /
            `speed_up_factor`)
        freqmax (int or float): Upper bandpass corner [Hz] (defaults to 20,000
            Hz / `speed_up_factor` or the `Nyquist frequency`_, whichever is
            smaller)
        speed_up_factor (int): Factor by which to speed up the waveform data
            (higher values = higher pitches)
        fps (int): Frames per second of output video
        resolution (str): Resolution of output video; one of `'crude'` (640
            :math:`\times` 360), `'720p'` (1280 :math:`\times` 720), `'1080p'`
            (1920 :math:`\times` 1080), `'2K'` (2560 :math:`\times` 1440), or
            `'4K'` (3840 :math:`\times` 2160)
        output_dir (str or :class:`~pathlib.Path`): Directory where output video
            should be saved (defaults to :meth:`~pathlib.Path.cwd`)
        spec_win_dur (int or float): Duration of spectrogram window [s]
        db_lim (tuple or str): Tuple defining min and max colormap cutoffs [dB],
            `'smart'` for a sensible automatic choice, or `None` for no clipping
        log (bool): If `True`, use log scaling for :math:`y`-axis of spectrogram
        utc_offset (int or float): If not `None`, convert UTC time to local time
            using this offset [hours] before plotting
    .. _Nyquist frequency: https://en.wikipedia.org/wiki/Nyquist_frequency
    """
    # Capture args and format as string to store in movie metadata
    #key_value_pairs = [f'{k}={repr(v)}' for k, v in locals().items()]
    #call_str = 'sonify({})'.format(', '.join(key_value_pairs))
    
    # Use current working directory if none provided
    if not output_dir:
        output_dir = Path().cwd()
    output_dir = Path(str(output_dir)).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f'Directory {output_dir} does not exist!')
    
    if type=='fdsn':
     print('Retrieving data...')
     client = Client(fdsn_client, debug=True, timeout=300)
     st = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime - PAD,
        endtime=endtime + PAD,
    )
    elif type=='file':
     st = obspy.read(file, starttime=starttime, endtime=endtime)
     if network is not None and station is not None and channel is not None:
        print('Selecting..')
        st.select(network=network, station=station, channel=channel)
    print('Done')

    # Merge Traces with the same IDs
    st.merge(fill_value='interpolate')

    if st.count() != 1:
        warnings.warn('Stream contains more than one Trace. Using first entry!')
        for tr in st:
          print(tr.id)
    tr = st[0]
    
    # Now that we have just one Trace, get inventory (which has response info)
    if not inv_file:
     inv = client.get_stations(
        network=tr.stats.network,
        station=tr.stats.station,
        location=tr.stats.location,
        channel=tr.stats.channel,
        starttime=tr.stats.starttime,
        endtime=tr.stats.endtime,
        level='response',
     )
    elif inv_file:
     inv = obspy.read_inventory(inv_file, level="response")
    # Adjust starttime so we have nice numbers in time box (carefully!)
    offset = np.abs(tr.stats.starttime - (starttime - PAD))  # [s]
    if offset > tr.stats.delta:
        warnings.warn(
            f'Difference between requested and actual starttime is {offset} s, '
            f'which is larger than the data sample interval ({tr.stats.delta} s). '
            'Not adjusting starttime of downloaded data; beware of inaccurate timing!'
        )
    else:
        tr.stats.starttime = starttime - PAD

    # Apply UTC offset if provided
    if utc_offset is not None:
        signed_offset = f'{utc_offset:{"+" if utc_offset else ""}g}'
        print(f'Converting to local time using UTC offset of {signed_offset} hours')
        utc_offset_sec = utc_offset * mdates.SEC_PER_HOUR
        starttime += utc_offset_sec
        endtime += utc_offset_sec
        tr.stats.starttime += utc_offset_sec

    # All infrasound sensors have a "?DF" channel pattern
    if tr.stats.channel[1:3] == 'DF':
        is_infrasound = True
        acceleration = False
        rescale = 1  # No conversion
    # All high-gain seismometers have a "?H?" channel pattern
    elif tr.stats.channel[1] == 'H':
        acceleration = None
        is_infrasound = False
        rescale = rescale # Convert m to lower (can be *µm per second)
    # Include accelerometer functionality that have "?N?" channel pattern
    elif tr.stats.channel[1] == 'N':
        is_infrasound = None
        acceleration = True
        rescale = 1e2 # Convert m to cm/[s]**2 (per square)
    # We can't figure out what type of sensor this is...
    else:
        raise ValueError(
            f'Channel {tr.stats.channel} is not an infrasound or seismic channel!'
        )

    if not freqmax:
        freqmax = np.min(
            [tr.stats.sampling_rate / 2, HIGHEST_AUDIBLE_FREQUENCY / speed_up_factor]
        )
    if not freqmin:
        freqmin = LOWEST_AUDIBLE_FREQUENCY / speed_up_factor
     
    if not pre_filt:
       freq_average = (tr.stats.sampling_rate / 2.05) - 2
       freq_peak = tr.stats.sampling_rate / 2
       pre_filt = (
                  0.004,
                  0.08,
                  freq_average,
                  freq_peak
                  )
    # If you do not specify 'remove_response', then: set the parameter to 0 (int zero) to skip removal.

    if remove_response == 1:
     tr.remove_response(inventory=inv, pre_filt=pre_filt)  # Units are m/s OR Pa after response removal. 
     # By default, when remove_response set to 1, then result is equivalent to what the scale is.
     tr.detrend('demean')
     tr.taper(max_percentage=None, max_length=PAD / 2)  # Taper away some of PAD
     
    elif remove_response == 2: # These output types are 'ACC', 'DEF', 'DISP'.
       tr.remove_response(inventory=inv, output='ACC', pre_filt=pre_filt)
    elif remove_response == 3:
       tr.remove_response(inventory=inv, output='DEF', pre_filt=pre_filt)       
    elif remove_response == 4:
       tr.remove_response(inventory=inv, output='DISP', pre_filt=pre_filt)
    elif remove_response == 5: # It removes sensitivity. Recommended for strong-motion measurements.
       tr.remove_sensitivity(inventory=inv)
     
    elif remove_response==0:
     print("Skipping response removal...")
     
    else: 
     print("Parameter 'remove_response' doesn't have exact value. Skipping process...")
     sys.exit(-1)

    if filter_type == 'bandpass':
     #tr.detrend("linear")
     print(f'Applying {freqmin:g}-{freqmax:g} Hz bandpass, filtering waveform {wf_freq:g}.')
     tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=False)
    elif filter_type == 'highpass':
     print(f'Applying {h_freq:g} Hz highpass')
     tr.filter('highpass', freq=h_freq-.46, corners=2, zerophase=True)
    elif filter_type == 'lowpass':
     print(f'Applying {l_freq:g} Hz lowpass')
     tr.filter('lowpass', freq=l_freq, corners=2, zerophase=True)
         
    # Make trimmed version
    tr_trim = tr.copy()
    tr_trim.trim(starttime, endtime)

    # We don't need an anti-aliasing filter here since we never use the values,
    # just the timestamps
    timing_tr = tr_trim.copy().interpolate(sampling_rate=fps / speed_up_factor)
    times = timing_tr.times('UTCDateTime')[:-1]  # Remove extra frame

    # Store user's rc settings, then update font stuff
    original_params = mpl.rcParams.copy()
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['font.sans-serif'] = 'Tex Gyre Heros'
    mpl.rcParams['mathtext.fontset'] = 'custom'

    fig, tr_w, wf_ax, spec_ax, *fargs = _spectrogram(
        tr,
        starttime,
        endtime,
        is_infrasound,
        acceleration,
        rescale,
        spec_win_dur,
        db_lim,
        unit_scale,
        num,
        spectral_scaling,
        cmap,
        wf_freq,
        (freqmin, freqmax),
        filter_type,
        (l_freq, h_freq),
        log,
        utc_offset is not None,
        resolution,
        specOpts=specOpts,
        fileName=fName if fName else f"{tr.id}_spec+{int(float(endtime.timestamp) * 1000)}",
        switchaxes=switchaxes,
    )
    
    # Clean up temporary directory, just to be safe
    #temp_dir.cleanup()
    return fig, tr, wf_ax, spec_ax, starttime, endtime


def _spectrogram(
    tr,
    starttime,
    endtime,
    is_infrasound,
    acceleration,
    rescale,
    spec_win_dur,
    db_lim,
    unit_scale,
    num,
    spectral_scaling,
    colormap,
    wf_freq,
    freq_lim,
    filter_type,
    lowhig_freqs,
    log,
    is_local_time,
    resolution,
    specOpts='',
    fileName='',
    switchaxes=False,
    ):
    
    """
    Make a combination waveform and spectrogram plot for an infrasound or
    seismic signal.
    Args:
        tr (:class:`~obspy.core.trace.Trace`): Input data, usually starts
            before `starttime` and ends after `endtime` (this function expects
            the response to be removed!)
        starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time
        endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time
        is_infrasound (bool): `True` if infrasound, `False` if seismic
        rescale (int or float): Scale waveforms by this factor for plotting
        spec_win_dur (int or float): See docstring for :func:`~sonify.sonify`
        db_lim (tuple or str): See docstring for :func:`~sonify.sonify`
        wf_freq (int or float): Filter waveform frequencies
            (followed by filter_type="bandpass")
        freq_lim (tuple): Tuple defining frequency limits for spectrogram plot
        filter_type (str): Choose what waveform data you want to filter out
             (can be 'bandpass', 'lowpass' or 'highpass')
        lowhig_freqs (tuple): Lowpass or highpass. Tuple argument to adjust the 
              frequency differences resulted from retrieving data
              from the server in MSEED
        log (bool): See docstring for :func:`~sonify.sonify`
        is_local_time (bool): Passed to :class:`_UTCDateFormatter`
        resolution (str): See docstring for :func:`~sonify.sonify`
    Returns:
        Tuple of (`fig`, `spec_line`, `wf_line`, `time_box`, `wf_progress`)
    """

    if is_infrasound:
        ylab = 'Pressure (Pa)'
        clab = f'Power (dB rel. [{REFERENCE_PRESSURE * 1e6:g} µPa]$^2$ Hz$^{{-1}}$)'
        ref_val = REFERENCE_PRESSURE
    elif acceleration and not is_infrasound:
        ylab = 'Acceleration\n (m/s**2 | cm s$^{-2}$)'
        clab = f'Power (dB rel. [{REFERENCE_ACCELERATION:g} m s$^{{-2}}$]$^2$ Hz$^{{-2}}$)'
        ref_val = REFERENCE_ACCELERATION
    else:
        ylab = 'Velocity (µm s$^{-1}$)'
        if REFERENCE_VELOCITY == 1:
            clab = (
                f'Power (dB rel. {REFERENCE_VELOCITY:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            )
        else:
            clab = (
                f'Power (dB rel. [{REFERENCE_VELOCITY:g} m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            )
        ref_val = REFERENCE_VELOCITY
        
    
    #print('Spectral scaling:', spectral_scaling)
    
    fs = tr.stats.sampling_rate
    nperseg = int(spec_win_dur * fs)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad fft with zeroes

    print('Sample buffering: {} .. {} per {}Hz (fs)'.format(nperseg, nfft, fs))

    M = 1000 # window size 
    w = windows.tukey(M)
    hop = int(M*0.1)

    #spt = ShortTimeFFT(w, hop , fs , mfft=nfft, scale_to='psd') #
    spt = ShortTimeFFT.from_window('hann', fs, nperseg=nperseg,
                               noverlap=nperseg // 2, mfft=nfft, scale_to='psd')
    sxx = spt.spectrogram(tr.data)  # perform the spectrogram

    t = spt.t(len(tr.data))
    f = spt.f

    # f, t, sxx = signal.spectrogram(
    #    tr.data, fs, nperseg=nperseg, window='hann', noverlap=nperseg // 2, scaling=spectral_scaling, nfft=nfft
    # )

    """
    plt.psd(tr.data, NFFT=nfft, Fs=fs,
            pad_to = 512,
            noverlap=nperseg // 2,
            scale_by_freq = True)
    """

    # [dB rel. (ref_val <ref_val_unit>)^2 Hz^-1]
    sxx_db = 10 * np.log10(sxx / (ref_val**2))

    t_mpl = tr.stats.starttime.matplotlib_date + (t / mdates.SEC_PER_DAY)
    
    # Ensure a 16:9 aspect ratio
    fig = plt.figure(figsize=(FIGURE_WIDTH, (9 / 16) * FIGURE_WIDTH))
    # set color
    fig.set_facecolor('#fcfcfc')
    # width_ratios effectively controls the colorbar width
    if switchaxes==True: # Switch position of axes between waveform and spectrogram.
      gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[40, 1])
      spec_ax = fig.add_subplot(gs[1, 0])
      wf_ax = fig.add_subplot(gs[0, 0], sharex=spec_ax)  # Share x-axis with spec
      cax = fig.add_subplot(gs[1, 1])
      wf_t = tr.copy()
    elif switchaxes==False:
      gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 1], width_ratios=[40, 1])
      spec_ax = fig.add_subplot(gs[0, 0])
      wf_ax = fig.add_subplot(gs[1, 0], sharex=spec_ax)  # Share x-axis with spec
      cax = fig.add_subplot(gs[0, 1])
      wf_t = tr.copy()    
    
    if filter_type == 'bandpass':
      wf_t.filter('bandpass', freqmin=wf_freq, freqmax=freq_lim[1], corners=2, zerophase=False)
    elif filter_type == 'lowpass':
      wf_t.filter('lowpass', freq=lowhig_freqs[0], corners=2, zerophase=False)
    elif filter_type == 'highpass':
      wf_t.filter('highpass', freq=lowhig_freqs[1], corners=2, zerophase=True)
    else:
      raise TypeError('Invalid frequency output')
    
    wf_lw = 0.5
    wf_ax.plot(wf_t.times('matplotlib'), wf_t.data * rescale, 'b', linewidth=wf_lw) #wf_t.data * rescale
    spec_ax.set_facecolor('#a3c8d4')
    cax.patch.set_facecolor('#f5f5f5')
    wf_ax.patch.set_facecolor('#c0c2fc')
    #wf_ax.patch.set_alpha(0.7)
    wf_progress = wf_ax.plot(np.nan, np.nan, 'black', linewidth=wf_lw)[0]
    spec_ax.set_alpha(0.2)
    wf_ax.set_ylabel(ylab, fontsize=8)
    wf_ax.grid(linestyle=':')
    spec_ax.yaxis.set_tick_params(labelsize=8)
    wf_ax.yaxis.set_tick_params(labelsize=8)
    max_value = np.abs(wf_t.copy().trim(starttime, endtime).data).max() * rescale
    
    if unit_scale == 'auto':
      print("Max value: ", round(max_value, 2))
      wf_ax.set_ylim(-max_value, max_value)
    elif unit_scale == 'other':
     if num:
      wf_ax.set_ylim(-num, num)
     else:
      num = (max_value * tr.count() / tr.stats.npts) ** .25
      print("Max value: ", num)
      wf_ax.set_ylim(-num, num)
    im = spec_ax.pcolormesh(
        t_mpl, f, sxx_db, cmap=colormap, shading='nearest', rasterized=True
    )

    spec_ax.set_ylabel('Frequency (Hz)', fontsize=8)
    spec_ax.grid(linestyle=':')
    spec_ax.set_ylim(freq_lim)
    if log:
        spec_ax.set_yscale('log')

    # Tick locating and formatting
    locator = mdates.AutoDateLocator()
    wf_ax.xaxis.set_major_locator(locator)
    wf_ax.xaxis.set_major_formatter(_UTCDateFormatter(locator, is_local_time))
    #fig.autofmt_xdate()

    # "Crop" x-axis!
    wf_ax.set_xlim(starttime.matplotlib_date, endtime.matplotlib_date)

    # Initialize animated stuff
    line_kwargs = dict(x=starttime.matplotlib_date, color='forestgreen', linewidth=1)
    spec_line = spec_ax.axvline(**line_kwargs)
    wf_line = wf_ax.axvline(**line_kwargs) #ymin=0.01, clip_on=False, zorder=10
    time_box = AnchoredText(
        s=starttime.strftime('%H:%M:%S'),
        pad=0.2,
        loc='lower right',
        bbox_to_anchor=[1, 1],
        bbox_transform=wf_ax.transAxes,
        borderpad=0,
        prop=dict(color='forestgreen'),
    )
    offset_px = -0.0025 * RESOLUTIONS[resolution][1]  # Resolution-independent!
    time_box.txt._text.set_y(offset_px)  # [pixels] Vertically center text
    time_box.zorder = 12  # This should place it on the very top; see below
    #time_box.patch.set_linewidth(mpl.rcParams['axes.linewidth'])
    #wf_ax.add_artist(time_box)

    # Adjustments to ensure time marker line is zordered properly
    # 9 is below marker; 11 is above marker
    
    spec_ax.spines['bottom'].set_zorder(9)
    
    wf_ax.spines['top'].set_zorder(9)
    for side in 'bottom', 'left', 'right':
        wf_ax.spines[side].set_zorder(11)
    
    # Pick smart limits rounded to nearest 10
    if db_lim == 'smart':
        db_min = np.percentile(sxx_db, 20)
        db_max = sxx_db.max()
        db_lim = (np.ceil(db_min / 10) * 10, np.floor(db_max / 10) * 10)

    # Clip image to db_lim if provided (doesn't clip if db_lim=None)
    im.set_clim(db_lim)

    # Automatically determine whether to show triangle extensions on colorbar
    # (kind of adopted from xarray)
    if db_lim:
        min_extend = sxx_db.min() < db_lim[0]
        max_extend = sxx_db.max() > db_lim[1]
    else:
        min_extend = False
        max_extend = False
    if min_extend and max_extend:
        extend = 'both'
    elif min_extend:
        extend = 'min'
    elif max_extend:
        extend = 'max'
    else:
        extend = 'neither'
        
    plt.colorbar(im, cax, extend=extend, extendfrac=EXTENDFRAC, ax=fig.get_axes(), label=clab)
    
    if switchaxes==False:
           spec_ax.set_title(f'{tr.id} - {tr.stats.endtime}', family='JetBrains Mono')
    elif switchaxes==True:
           wf_ax.set_title(f'{tr.id} - {tr.stats.endtime}', family='JetBrains Mono')
           
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02, wspace=0.05)
    
    
    if switchaxes == True:
      fig.autofmt_xdate()
      spec_ax.xaxis.set_major_locator(locator)
      spec_ax.xaxis.set_major_formatter(_UTCDateFormatter(locator, is_local_time))
      plt.setp(wf_ax.get_xticklabels(), visible=False)
      wf_ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
      plt.setp(spec_ax.xaxis.get_majorticklabels(), rotation = 45)
      #fig.delaxes(wf_ax)
    elif switchaxes == False:
      fig.autofmt_xdate()
      plt.setp(spec_ax.get_xticklabels(), visible=False)
      spec_ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
     
     
    # Finnicky formatting to get extension triangles (if they exist) to extend
    # above and below the vertical extent of the spectrogram axes
    pos = cax.get_position()
    triangle_height = EXTENDFRAC * pos.height
    ymin = pos.ymin
    height = pos.height
    if min_extend and max_extend:
        ymin -= triangle_height
        height += 2 * triangle_height
    elif min_extend and not max_extend:
        ymin -= triangle_height
        height += triangle_height
    elif max_extend and not min_extend:
        height += triangle_height
    else:
        pass
    cax.set_position([pos.xmin, ymin, pos.width, height])
    # Move offset text around and format it more nicely, see
    # https://github.com/matplotlib/matplotlib/blob/710fce3df95e22701bd68bf6af2c8adbc9d67a79/lib/matplotlib/ticker.py#L677
    magnitude = wf_ax.yaxis.get_major_formatter().orderOfMagnitude
    if magnitude:  # I.e., if offset text is present
        wf_ax.yaxis.get_offset_text().set_visible(False)  # Remove original text
        sf = ScalarFormatter(useMathText=True)
        sf.orderOfMagnitude = magnitude  # Formatter needs to know this!
        sf.locs = [47]  # Can't be an empty list
        wf_ax.text(
            0.002,
            0.95,
            sf.get_offset(),  # Let the ScalarFormatter do the formatting work
            transform=wf_ax.transAxes,
            ha='left',
            va='top',
        )
        
    if specOpts == "saveAsPngFile":
     from matplotlib.artist import Artist
     try:
       Artist.remove(time_box)
     except:
       pass
     print("Resolution", plt.gcf().get_size_inches(), fig.get_dpi())
     fig.savefig(fileName, format='png', dpi=RESOLUTIONS[resolution][0] / FIGURE_WIDTH)
     print('Fig. size:', plt.gcf().get_size_inches())
     print('DPI:', fig.get_dpi())
     print('File Saved as PNG')
    elif specOpts == "saveAsPDF":
     from matplotlib.artist import Artist
     spec_ax.set_rasterized(True)
     #outFile=str(fileName)
     fig.savefig(f"{fileName}" + ".pdf", format='pdf', dpi=RESOLUTIONS[resolution][0] / FIGURE_WIDTH, bbox_inches='tight')
     print('File Saved as PDF.') 
    elif specOpts == "interactive": # mode [Preview] : preview as interactive
     #plt.ion()
     fig.set_size_inches(10, 8)
     print('Fig. size:', plt.gcf().get_size_inches())
     print('DPI', fig.get_dpi())
     plt.show(block=True)
    elif specOpts is None:
     print("DPI:", fig.get_dpi())
     #pass
    else:
     raise ValueError(f'specOpts argument "{specOpts}" has unexpected non-valid choice. Cannot proceed.')
     
    return fig, tr, wf_ax, spec_ax, spec_line, wf_line, time_box, wf_progress

def maximize():
    plot_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if plot_backend == 'TkAgg':
        mng.resize(*mng.window.maxsize())
    elif plot_backend == 'wxAgg':
        mng.frame.Maximize(True)
    elif plot_backend == 'Qt4Agg':
        mng.window.showMaximized()

# Subclass ConciseDateFormatter (modifies __init__() and set_axis() methods)
class _UTCDateFormatter(mdates.ConciseDateFormatter):
    def __init__(self, locator, is_local_time):
        super().__init__(locator)

        # Determine proper time label (local time or UTC)
        if is_local_time:
            time_type = 'Local'
        else:
            time_type = 'UTC'

        # Re-format datetimes
        self.formats[1] = '%B'
        self.zero_formats[2:4] = ['%B', '%B %d']
        self.offset_formats = [
            f'{time_type} time',
            f'{time_type} time in %Y',
            f'{time_type} time in %B %Y',
            f'{time_type} time on %B %d, %Y',
            f'{time_type} time on %B %d, %Y',
            f'{time_type} time on %B %d, %Y at %H:%M',
        ]

    def set_axis(self, axis):
        self.axis = axis

        # If this is an x-axis (usually is!) then center the offset text
        if self.axis.axis_name == 'x':
            offset = self.axis.get_offset_text()
            offset.set_horizontalalignment('center')
            offset.set_x(0.5)

class FigureWrapper(object):
    '''Frees underlying figure when it goes out of scope.'''            
    def __init__(self, figure):
        self._figure = figure
    def __enter__(self):
        plt.close(self._figure)
    def __del__(self):
        self._figure.clf()
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('Figure removed')

def main():
    """
    This function is run when ``sonify.py`` is called as a script. It's also set
    up as an entry point.
    """

    parser = argparse.ArgumentParser(
        description='Produce an animated spectrogram with a soundtrack derived from sped-up seismic or infrasound data.',
        allow_abbrev=False,
    )

    # Hack the printing function of the parser to fix --db_lim option formatting
    def _print_message_replace(self, message, file=None):
        if message:
            if file is None:
                file = _sys.stderr
            file.write(message.replace('[DB_LIM ...]', '[DB_LIM]'))

    parser._print_message = MethodType(_print_message_replace, parser)

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'{parser.prog}, rev. {__version__}',
        help=f'show revision number and exit',
    )

    parser.add_argument('network', help='SEED network code')
    parser.add_argument('station', help='SEED station code')
    parser.add_argument('channel', help='SEED channel code')
    parser.add_argument(
        'starttime',
        type=UTCDateTime,
        help='start time of animation (UTC), format yyyy-mm-ddThh:mm:ss',
    )
    parser.add_argument(
        'endtime',
        type=UTCDateTime,
        help='end time of animation (UTC), format yyyy-mm-ddThh:mm:ss',
    )
    parser.add_argument('--location', default='*', help='SEED location code')
    parser.add_argument(
        '--freqmin',
        default=None,
        type=float,
        help='lower bandpass corner [Hz] (defaults to 20 Hz / "SPEED_UP_FACTOR")',
    )
    parser.add_argument(
        '--freqmax',
        default=None,
        type=float,
        help='upper bandpass corner [Hz] (defaults to 20,000 Hz / "SPEED_UP_FACTOR" or the Nyquist frequency, whichever is smaller)',
    )
    parser.add_argument(
        '--speed_up_factor',
        default=200,
        type=int,
        help='factor by which to speed up the waveform data (higher values = higher pitches)',
    )
    parser.add_argument(
        '--fps', default=1, type=int, help='frames per second of output video'
    )
    parser.add_argument(
        '--resolution',
        default='4K',
        choices=RESOLUTIONS.keys(),
        help='resolution of output video; one of "crude" (640 x 360), "720p" (1280 x 720), "1080p" (1920 x 1080), "2K" (2560 x 1440), or "4K" (3840 x 2160)',
    )
    parser.add_argument(
        '--spec_win_dur',
        default=5,
        type=float,
        help='duration of spectrogram window [s]',
    )
    parser.add_argument(
        '--db_lim',
        default='smart',
        nargs='+',
        help='numbers "<min>" "<max>" defining min and max colormap cutoffs [dB], "smart" for a sensible automatic choice, or "None" for no clipping',
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='use log scaling for y-axis of spectrogram',
    )
    parser.add_argument(
        '--utc_offset',
        default=None,
        type=float,
        help='if provided, convert UTC time to local time using this offset [hours] before plotting',
    )

    input_args = parser.parse_args()

    # Extra type check for db_lim kwarg
    db_lim_error = False
    db_lim = np.atleast_1d(input_args.db_lim)
    if db_lim.size == 1:
        db_lim = db_lim[0]
        if db_lim == 'smart':
            pass
        elif db_lim == 'None':
            db_lim = None
        else:
            db_lim_error = True
    elif db_lim.size == 2:
        try:
            db_lim = tuple(float(s) for s in db_lim)
        except ValueError:
            db_lim_error = True
    else:  # User provided more than 2 args
        db_lim_error = True
    if db_lim_error:
        parser.error(
            'dB Limit <argument --db_lim>: must be one of "smart", "None", or two numeric values "<min>" "<max>"'
        )

    sonify(
        input_args.network,
        input_args.station,
        input_args.channel,
        input_args.starttime,
        input_args.endtime,
        input_args.location,
        input_args.freqmin,
        input_args.freqmax,
        input_args.speed_up_factor,
        input_args.fps,
        input_args.resolution,
        input_args.spec_win_dur,
        db_lim,
        input_args.log,
        input_args.utc_offset,
    )


if __name__ == '__main__':
    main()

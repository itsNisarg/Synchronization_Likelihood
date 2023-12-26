import os
import mne
import numpy as np
from eeglib.preprocessing import bandPassFilter


def load_data(file):
    """Load data from a BrainVision file and preprocess it.
    Parameters
    ----------
    file : str
        Path to the BrainVision file.
    Returns
    -------
    raw : mne.io.Raw
        The raw data.
    """
    raw = mne.io.read_raw_brainvision(file, preload=True)
    raw.set_channel_types({'EOG': 'eog', 'ECG': 'ecg'})
    raw.set_eeg_reference(['Cz'])
    # raw.info['bads'] = ['Cz']
    raw.filter(l_freq=0.5, h_freq=250, n_jobs=4)
    raw.notch_filter(50)
    ds_raw = raw.copy().resample(sfreq=500)
    return ds_raw


def filter(rawdata: mne.io.Raw, info: mne.io.Info) -> None:
    """Function to filter the raw data into different frequency bands and save them as fif files."""

    if not os.path.exists('./filtered'):
        os.makedirs('./filtered')
    os.chdir('./filtered')

    data = rawdata.get_data()
    sfreq = info['sfreq']

    delta = bandPassFilter(data, sampleRate=sfreq,
                           highpass=0.5, lowpass=4, order=2)
    theta = bandPassFilter(data, sampleRate=sfreq,
                           highpass=4, lowpass=8, order=2)
    alpha = bandPassFilter(data, sampleRate=sfreq,
                           highpass=8, lowpass=13, order=2)
    beta = bandPassFilter(data, sampleRate=sfreq,
                          highpass=13, lowpass=30, order=2)
    gamma = bandPassFilter(data, sampleRate=sfreq,
                           highpass=30, lowpass=45, order=2)

    mne.io.RawArray(delta, info).save('delta_raw.fif.gz', overwrite=True)
    mne.io.RawArray(theta, info).save('theta_raw.fif.gz', overwrite=True)
    mne.io.RawArray(alpha, info).save('alpha_raw.fif.gz', overwrite=True)
    mne.io.RawArray(beta, info).save('beta_raw.fif.gz', overwrite=True)
    mne.io.RawArray(gamma, info).save('gamma_raw.fif.gz', overwrite=True)

    os.chdir("..")


if __name__ == '__main__':
    path = input("Enter the path of the file: ")
    if (not os.path.exists(path)):
        print("File does not exist")
    elif (not path.endswith(".vhdr")):
        print("Invalid file format")
    else:
        # raw = mne.io.read_raw_edf(path, preload=True)
        raw = load_data(path)
        filter(raw, raw.info)

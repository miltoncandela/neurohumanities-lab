import os
import pandas as pd
from brainflow.data_filter import DataFilter
import numpy as np
from copy import deepcopy
from scipy.signal import welch
from scipy.integrate import simps
from statistics import median, mean, stdev

def norm_metric(df_name, df_name_calib):
    # name = corr_coef_nolast5min

    df = pd.read_csv(df_name + '.csv', index_col=0)
    df_calib = pd.read_csv(df_name_calib + '.csv', index_col=0)
    df_calib = df_calib[df_calib.Band.isin(df.Band.unique())]

    a = np.zeros(shape=(df.shape[0],))
    c = np.zeros(shape=(df.shape[0],))
    l_missing = []

    for subject in df.Subject.unique():
        l = []
        l.append(subject)
        if set(l) <= set(df_calib.Subject.unique()):
            df_calib_temp = df_calib[df_calib.Subject == subject]
            b = np.array(df_calib_temp.Power)
            d = np.array(df_calib_temp.SPower)
            y = np.array(df_calib_temp.APower)
            z = np.array(df_calib_temp.IPower)
            for scene in df.Scene.unique():
                df_temp = df[(df.Subject == subject) & (df.Scene == scene)]
                c[df_temp.index[0]:(df_temp.index[-1] + 1)] = b
                a[df_temp.index[0]:(df_temp.index[-1] + 1)] = (df_temp.Power - b)/d
                # a[df_temp.index[0]:(df_temp.index[-1] + 1)] = (df_temp.Power - z) / (y - z)
        else:
            print(subject, 'not found in Calib!')
            l_missing.append(subject)
            continue

    df['Power'] = a
    df_calib = deepcopy(df)
    df_calib['Power'] = c

    print(df)

    df = df[~df.Subject.isin(l_missing)].reset_index(drop=True)
    df_calib = df_calib[~df_calib.Subject.isin(l_missing)].reset_index(drop=True)

    df_calib.to_csv('pross/' + df_name_calib + '_pross.csv', index=False)
    df.to_csv('pross/' + df_name + '_norm.csv', index=False)
    exit()




def calc_bands2(s, band, seconds):
    sampling_rate = 250

    def calc_band_power(data):
        # Calculate PSD
        freqs, psd = welch(data, fs=sampling_rate, nperseg=window_size, axis=0)
        # psd = 10 * np.log10(psd)

        freq_res = freqs[1] - freqs[0]
        low, high = freq_bands[band]
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        return simps(psd[idx_band], dx=freq_res)
        # band_indices = np.where((psd >= low) & (psd < high))[0]
        # return np.sum(psd[band_indices])

    def calc_PSD(data):
        sr = 250  # Sampling frequency
        wf = 3  # Windows function (0: No Window, 1: Hanning, 2: Hamming, 3: Blackman Harris)
        nfft = DataFilter.get_nearest_power_of_two(sr)
        over = DataFilter.get_nearest_power_of_two(sr) // 2

        # Calculate the PSD using the Welch method with specified window parameters
        psd = DataFilter.get_psd_welch(data=np.array(data).astype(float), nfft=nfft,
                                       overlap=over, sampling_rate=sr, window=wf)

        # Calculate the average alpha power (e.g., for alpha frequency range of 8-13 Hz)
        power = DataFilter.get_band_power(psd, freq_bands[band][0], freq_bands[band][1])
        return power

    # Calculate the number of windows
    window_size = seconds * sampling_rate  # 4 seconds * sampling rate
    num_windows = len(s) // window_size

    a = []
    # Iterate over 4-second windows and calculate PSD and Band Power
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size

        # Extract data for the current window
        window_data = s[start_index:end_index]

        # Calculate PSD and Band Power for the window
        # bp = calc_band_power(window_data)
        bp = calc_PSD(window_data)
        a.append(bp)

    return median(a), stdev(a), max(a), min(a)



# norm_metric('PSD_TAB', 'PSD_TAB_calib')

# freq_bands = {'Theta': [4, 8], 'AlphaL': [8, 10], 'AlphaH': [10, 12], 'Alpha': [8, 12],
#                                 'BetaL': [12, 20], 'BetaH': [20, 30], 'Beta': [12, 30]}
freq_bands = {'Theta': [4, 8], 'Alpha': [8, 12], 'Beta': [12, 30]}
channels = ['FP1', 'FP2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2']
# samples = 250*60*4  # Removing the last 1 minute of the take
centralp = 'C:/Users/Milton/PycharmProjects/neurohumanities-lab/IEEE2026/data/csv'
fs = 250

df_PSD = pd.DataFrame(columns=['Subject', 'Scene', 'Ch', 'Band', 'Power', 'SPower', 'APower', 'IPower'])
for file in os.listdir(centralp): # used_children: #   #   # os.listdir('data/EEG_Frontiers'):
    subject = file[1:4]
    scene = file[5:8]
    if scene == '000':
        continue
    df = pd.read_csv('{}/{}'.format(centralp, file), header=None).drop(0, axis=0)
    # df = df.iloc[(fs*30):(df.shape[0]-fs*30), :]  # Removing the first minute of data
    # df = df.iloc[:250*60*1, :]  # Only gathering the first minute of data (calibration)
    # df = df.iloc[(250*60*4):(250*60*19), :] # Removing the first 4 minutes until 19 minutes (15 min)
    # df = df.iloc[samples:(df.shape[0] - samples + 250 * 60 * 1), :]  # Dropping the last 1 minute
    df.columns = channels
    for band in freq_bands.keys():
        combinations = []
        for channel_i in channels:
            # val, valsd = calc_PSD(df[channel_i], band)
            val, valsd, valmax, valmin = calc_bands2(df[channel_i], band, 2)

            data_dict = {'Subject': subject, 'Scene': scene, 'Ch': channel_i, 'Band': band, 'Power': val,
                         'SPower': valsd, 'APower': valmax, 'IPower': valmin}
            df_PSD = pd.concat([df_PSD, pd.DataFrame(data_dict, index=[0])], axis=0)
    print()

print(df_PSD)
df_PSD.reset_index(drop=True).to_csv('pross/PSD_TAB.csv')
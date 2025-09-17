import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from brainflow import DataFilter
import numpy as np
import os
from math import floor


def eeg_analysis():
    # Task: From 2-Second windows to 1-Second windows
    spectral_signals = {'Theta': [4, 8], 'Alpha': [8, 12], 'Beta': [12, 30]}

    # EEG
    sr = 250  # Sampling frequency
    ft = 0  # Filter type (0: Butter, 1: Chev, 2: Bessel)
    eta = 4  # Order number of filter
    det = 2  # De-trend operation (0: None, 1: Constant, 2: Linear)
    ws = sr*2  # 2-second windows
    over = floor(ws*19/20)  # 95% overlap
    nfft = DataFilter.get_nearest_power_of_two(sr)
    wf = 3  # Windows function (0: No Window, 1: Hanning, 2: Hamming, 3: Blackman Harris)
    channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    n_channels = len(channels)  # Number of channels in EEG
    # channel_to_position = {'A2': ['Right Cushion'], 'A1': ['Left Cushion'], 'C4': ['Top Right'], 'C3': ['Top Left']}
    # channel_to_position = {'C4': ['Top Right'], 'C3': ['Top Left']}
    columns_signals = [s + '_' + str(c) for c in channels for s in spectral_signals.keys()]
    df_total = pd.DataFrame()

    for file in os.listdir('data/csv_raw'):
        df_raw = (pd.read_csv('data/csv_raw/{}'.format(file), index_col=0))
        df_raw = df_raw.drop(['Timestamp', 'Acc_1', 'Acc_2', 'Acc_3'], axis=1)
        # print(df_raw)
        # print(df_raw.columns)
        df_eeg = pd.DataFrame(columns=columns_signals)

        # Assuming 250 values per second, then
        # 2-second windows with a second overlap
        for n in range(df_raw.shape[0] // ws):
            l_signals = []

            # Average referencing
            df_unpross = np.array(df_raw.iloc[(n * ws):((n + 1) * ws), :])
            reference = np.average(df_unpross, axis=1, keepdims=True)
            # print(reference.shape)
            df_unpross = df_unpross - reference
            # print(df_unpross.shape)

            for n_channel in range(n_channels):
                df_unpross_channel = df_unpross[:, n_channel]

                # 60 Hz Notch filter
                DataFilter.perform_bandstop(data=df_unpross_channel, sampling_rate=sr, start_freq=59,
                                            stop_freq=61, order=eta, filter_type=ft, ripple=0)

                # 4–50 Hz 4th order Butterworth bandpass filter
                DataFilter.perform_lowpass(data=df_unpross_channel, sampling_rate=sr, cutoff=50,
                                           order=eta, filter_type=ft, ripple=0)
                DataFilter.perform_highpass(data=df_unpross_channel, sampling_rate=sr, cutoff=4,
                                            order=eta, filter_type=ft, ripple=0)

                # Linear de-trend
                DataFilter.detrend(df_unpross_channel, detrend_operation=det)

                psd_data = DataFilter.get_psd_welch(df_unpross_channel, nfft=nfft, overlap=nfft//2,
                                                    sampling_rate=sr, window=wf)

                for spectral_signal in spectral_signals.keys():
                    l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                               spectral_signals[spectral_signal][1]))
            data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[n], columns=columns_signals)
            df_eeg = pd.concat([df_eeg, data_signals])

        # n = df_eeg.shape[0]
        # if n in [321, 322, 323]:
        # print([x for x in range(df_eeg.shape[0])])
        # print(len([x for x in range(2, df_eeg.shape[0], 2)]))
        # print(df_eeg.shape)
        df_eeg['Subject'], df_eeg['Scene'], df_eeg['Second'] = file[1:4], file[5:8], [x*2 for x in range(1, df_eeg.shape[0]+1)]
        df_total = pd.concat([df_total, df_eeg], axis=0, ignore_index=True)

    # df_fatigue = get_fatigue().drop(['Name', 'FAS_Cat_2', 'FAS_Cat_3'], axis=1)
    # Subject columns are transformed to a numerical feature.
    df_total['Subject'] = pd.to_numeric(df_total.Subject)
    df_total = df_total.dropna(axis=0, how='any')
    # df_fatigue['Subject'] = pd.to_numeric(df_fatigue.Subject)

    # Both fatigue features and biometric features are joined using Subject's ID as the primary key.
    # df_total = pd.merge(df_total, df_fatigue, on='Subject')

    # channel_to_position = {'A2': ['Right Cushion'], 'A1': ['Left Cushion']}
    # channel_to_position = channel_to_position = {'C4': ['Top Right'], 'C3': ['Top Left']}
    # columns_signals = [s + '_' + str(c) for c in channel_to_position.keys() for s in spectral_signals.keys()]
    # df_total = df_total.drop(columns_signals, axis=1)

    # Subject ID 99 is removed because it is only used for testing, on the other hand, subject ID 19 is removed due to
    # pain medication, which could disturb biometrics.
    # df_total = df_total.drop(df_total[df_total.Subject.isin(['99', '19', 99, 19])].index, axis=0)

    return df_total

df = eeg_analysis()
df.to_csv('pross/eeg.csv')
# df = pd.read_csv('pross/PSD_TAB_ICA_QZ_norm.csv')
# create_model(df)

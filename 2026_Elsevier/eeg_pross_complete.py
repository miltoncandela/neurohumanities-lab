import os
import pandas as pd
from brainflow.data_filter import DataFilter
import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def norm_metric(df_name, df_name_calib, outl='quantile', norm='zscore'):
    df = pd.read_csv('pross/' + df_name + '.csv', index_col=0).drop(['Second'], axis=1)
    df_calib = pd.read_csv('pross/' + df_name_calib + '.csv', index_col=0).drop(['Scene', 'Second'], axis=1)

    def outlier_removal(data, df_calibration, method='quantile'):
        if method == 'quantile':
            q1 = df_calibration.quantile(q=.05)
            q3 = df_calibration.quantile(q=.95)
            iqr = df_calibration.apply(stats.iqr)
            data = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]
        elif method == 'zscore':
            data = data[np.abs(data - df_calibration.mean()) <= (3 * df_calibration.std())].dropna(axis=0, how='any')
        return data

    def normalization(data, df_calibration, method='zscore'):
        column_names = data.columns
        if method == 'minmax':
            scaler = MinMaxScaler().fit(df_calibration)
        elif method == 'zscore':
            scaler = StandardScaler().fit(df_calibration)
        data = pd.DataFrame(scaler.transform(data), columns=column_names)
        return data

    df_norm = pd.DataFrame(columns=df.columns)
    for subject in df.Subject.unique():
        df_temp = df[df.Subject == subject].drop('Subject', axis=1)
        df_calib_temp = df_calib[df_calib.Subject == subject].drop('Subject', axis=1)
        for scene in df_temp.Scene.unique():
            df_temp_temp = df_temp[df_temp.Scene == scene].drop('Scene', axis=1)
            a = df_temp_temp.shape[0]
            df_temp_temp = outlier_removal(df_temp_temp, df_calib_temp, outl)
            b = df_temp_temp.shape[0]
            print(subject, scene, a, b, round(b/a*100, 2))
            if b == 0:
                continue

            df_temp_temp = normalization(df_temp_temp, df_calib_temp, norm)
            df_temp_temp[['Subject', 'Scene']] = subject, scene
            df_norm = pd.concat([df_norm, df_temp_temp], axis=0)

    print(df_norm)

    # df_calib.to_csv('pross/' + df_name_calib + '_pross.csv', index=False)
    df_norm.to_csv('pross/{}_{}{}_norm.csv'.format(df_name, outl[0].capitalize(), norm[0].capitalize()), index=False)


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
    window_size = seconds * sampling_rate  # 2 seconds * sampling rate
    num_windows = len(s) // window_size

    a = []
    # Iterate over 2-second windows and calculate PSD and Band Power
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size

        # Extract data for the current window
        window_data = s[start_index:end_index]

        # Calculate PSD and Band Power for the window
        # bp = calc_band_power(window_data)
        bp = calc_PSD(window_data)
        a.append(bp)

    return a

    # return median(a), stdev(a), max(a), min(a)


def dim_reduction(df, freq_bands, channel_groups):
    init_columns = df.columns
    for cg in channel_groups.keys():
        for band in freq_bands.keys():
            sub = ['{}_{}'.format(x, band) for x in channel_groups[cg]]
            df['{}_{}'.format(cg, band)] = np.mean(df[sub], axis=1)
    df = df.drop(init_columns, axis=1)

    # Average pooled
    # subt = np.array(df[['{}_{}'.format(x, 'Theta') for x in channel_groups.keys()]])
    # suba = np.array(df[['{}_{}'.format(x, 'Alpha') for x in channel_groups.keys()]])
    # subb = np.array(df[['{}_{}'.format(x, 'Beta') for x in channel_groups.keys()]])

    # Targeted pooled
    subt = np.mean(np.array(df[['{}_Theta'.format(x) for x in ['Fro', 'Cen']]]), 1)
    suba = np.array(df['Par_Alpha'])
    subb = np.array(df['Fro_Beta'])

    # df['Engagement'] = np.mean(np.divide(subb, np.sum([subt, suba], axis=0)), axis=1)
    # df['Fatigue'] = np.mean(np.divide(suba, subt), axis=1)
    # df['Excitement'] = np.mean(np.divide(subb, suba), axis=1)

    df['Engagement'] = np.divide(subb, np.sum([subt, suba], axis=0))
    # df['Fatigue'] = np.divide(suba, subt)
    df['Fatigue'] = np.divide(subt, suba)
    df['Excitement'] = np.divide(subb, suba)

    return df


for outlier_method in ['quantile', 'zscore']:
    for norm_method in ['minmax', 'zscore']:
        # pass
        norm_metric('PSD_TABC', 'PSD_TABC_calib', outlier_method, norm_method)

# freq_bands = {'Theta': [4, 8], 'AlphaL': [8, 10], 'AlphaH': [10, 12], 'Alpha': [8, 12],
#                                 'BetaL': [12, 20], 'BetaH': [20, 30], 'Beta': [12, 30]}
freq_bands = {'Theta': [4, 8], 'Alpha': [8, 12], 'Beta': [12, 30]}
channels = ['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
channel_groups = {'Fro': ['FP1', 'FP2'], 'Cen': ['C3', 'C4'], 'Par': ['P7', 'P8'], 'Occ': ['O1', 'O2']}
# 4 areas by 3 frequency bands = 12 features + 3 indices = 15 features

# samples = 250*60*4  # Removing the last 1 minute of the take
centralp = 'C:/Users/Milton/PycharmProjects/neurohumanities-lab/2026_Elsevier/raw/csv'
fs = 250

channels_bands = ['{}_{}'.format(channel, band) for channel in channels for band in freq_bands.keys()]
groups_bands = ['{}_{}'.format(group, band) for group in channel_groups.keys() for band in freq_bands.keys()]

# FOR GROUP VALUES:
# df_PSD = pd.DataFrame(columns=groups_bands + ['Subject', 'Scene', 'Second'])
# FOR CHANNEL VALUES:
df_PSD = pd.DataFrame(columns=channels_bands + ['Subject', 'Scene', 'Second'])

for file in os.listdir(centralp): # used_children: #   #   # os.listdir('data/EEG_Frontiers'):
    subject, scene = file[1:4], file[5:8]
    if scene != '000':
        continue
    df = pd.read_csv('{}/{}'.format(centralp, file), header=None).drop(0, axis=0)
    # df = df.iloc[(fs*30):(df.shape[0]-fs*30), :]  # Removing the first minute of data
    # df = df.iloc[:250*60*1, :]  # Only gathering the first minute of data (calibration)
    # df = df.iloc[(250*60*4):(250*60*19), :] # Removing the first 4 minutes until 19 minutes (15 min)
    # df = df.iloc[samples:(df.shape[0] - samples + 250 * 60 * 1), :]  # Dropping the last 1 minute
    df.columns = channels
    df_temp = pd.DataFrame(columns=channels_bands + ['Subject', 'Scene', 'Second'] )

    for channel in channels:
        for band in freq_bands.keys():
            df_temp['{}_{}'.format(channel, band)] = calc_bands2(df[channel], band, 2)
    # FOR GROUP VALUES:
    # df_temp = dim_reduction(df_temp, freq_bands, channel_groups)

    df_temp[['Subject', 'Scene']] = [subject, scene]
    df_temp['Second'] = range(2, df_temp.shape[0]*2+1, 2)
    df_PSD = pd.concat([df_PSD, df_temp], axis=0)
    print()

print(df_PSD)
df_PSD.reset_index(drop=True).to_csv('pross/PSD_TABC_calib.csv')
import pandas as pd
import numpy as np
from scipy.signal import welch, butter, lfilter
import os
from brainflow.data_filter import DataFilter


def compute_psd_bands(data, fs):
    f, psd = welch(data, fs=fs, nperseg=250)

    # Define the frequency ranges for each band
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }

    # Compute the PSD for each frequency band
    psd_bands = {}
    for band, (f_min, f_max) in bands.items():
        idx = np.where((f >= f_min) & (f < f_max))[0]
        psd_bands[band] = np.mean(psd[idx])

    return psd_bands


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


sampling_rate = 128
channels = list(range(9))

lowcut = 4  # Lower cutoff frequency in Hz
highcut = 45  # Upper cutoff frequency in Hz
fs = 128  # Sampling rate in Hz


def get_df(device):
    """
    The current function receives a device and a folder as a parameters, this determines which CSV files would be read
    in order to create a DataFrame based on all the CSV files on that sub-folder.

    :param string device: Biometric devices from which the data would be gathered.
    :param string folder: Sub-folder of the take, based on the biometric device selected.
    :return pd.DataFrame: A DataFrame with all the CSV observations found on the sub-folder.
    """

    # The CSV files are listed, and so it creates a list of files from which a blank dataframe is initialized.
    file_list = os.listdir(device)

    df_file = pd.read_csv(device + '/' + file_list[0] + '/' + os.listdir(device + '/' + file_list[0])[0], index_col=0, nrows=0)
    df_file = df_file.drop(['Timestamp', 'Acc_1', 'Acc_2', 'Acc_3'], axis=1).dropna()
    columns = list(df_file.columns) + ['ID', 'Take', 'Group']
    df_file = pd.DataFrame(columns=columns)

    # The following for loop iterates over all the CSV files, concatenating a temporal DataFrame with the main, blank
    # DataFrame. It is worth noting that the DataFrame keeps track of the CSV file which is read, such as: the kid's ID
    # kid; Take; and Session, this is important because the target variables would be joined using these data.
    for i, file in enumerate(file_list):
        df_file_temp = pd.read_csv(device + '/' + file_list[i] + '/' + os.listdir(device + '/' + file_list[i])[0], index_col=0)
        df_file_temp = df_file_temp.drop(['Timestamp', 'Acc_1', 'Acc_2', 'Acc_3'], axis=1).dropna()

        df_file_temp['ID'], df_file_temp['Take'] = file[1:4], file[5:8]
        df_file_temp['Group'] = 'Happiness' if int(file[1:4]) in [2, 4, 6] else 'Sadness'

        print(df_file_temp.shape)

        df_file = pd.concat([df_file, df_file_temp], ignore_index=True)
    # df_file = df_file.dropna(axis=1, how='any').reset_index(drop=True)
    return df_file


# df = pd.read_csv('EEG/S001R001_17112023_1256/S001R001_Complete.csv', index_col=0).drop(['Timestamp', 'Acc_1', 'Acc_2', 'Acc_3'], axis=1).dropna()
df = get_df('EEG')
print(df)
print(df.shape)

CANALES = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
SPECTRALS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
cana_spec = ['{}_{}'.format(c, s) for c in CANALES for s in SPECTRALS]


def pross_eeg(df_sub):
    lowcut = 0.4  # Lower cutoff frequency in Hz
    highcut = 45  # Upper cutoff frequency in Hz
    fs = 250  # Sampling rate in Hz
    ratio = 128 / 250
    print('*')
    print(df_sub)

    df_sub = df_sub.apply(pd.to_numeric, errors='coerce')
    for col in df_sub.columns:
        x = np.array(df_sub.loc[:, col])
        DataFilter.detrend(x, detrend_operation=2)
        df_sub[col] = x
    df_sub = df_sub.apply(lambda col: butter_bandpass_filter(col, lowcut, highcut, fs))
    df_sub = df_sub.sub(df_sub.mean(axis=1), axis=0)
    print('*')
    print(df_sub)
    #exit()

    def psd_calc(df5):
        filtered_df = df5.apply(pd.to_numeric, errors='coerce').iloc[::int(1 / ratio)].interpolate()

        # Apply the bandpass filter to each column

        # filtered_df = df5.apply(lambda col: butter_bandpass_filter(col, lowcut, highcut, fs))

        # Create an empty DataFrame to store the PSD results
        psd_df = pd.DataFrame()

        # Iterate over each column in your DataFrame
        for column in filtered_df.columns:
            # Compute the PSD for the column data and frequency bands
            psd_bands = compute_psd_bands(filtered_df[column].values, fs=128)

            # Add the PSD values to the DataFrame
            psd_df = pd.concat([psd_df, pd.DataFrame(psd_bands, index=[0])], axis=0, ignore_index=True)

        df_t = psd_df.transpose()
        df_t.columns = CANALES
        df_t = df_t.reset_index()

        # Use the melt function to reshape the DataFrame
        melted_df = pd.melt(df_t, id_vars='index', var_name='channel', value_name='value')

        # Convert channel numbers to strings
        melted_df['channel'] = melted_df['channel'].astype(str)

        # Create a new 'channel_band' column by combining 'channel' and 'index' columns
        melted_df['channel_band'] = melted_df['channel'] + '_' + melted_df['index']

        # Pivot the DataFrame to get the desired format
        new_df = melted_df.pivot(index='index', columns='channel_band', values='value')

        series = new_df.stack()

        # Convert the Series back to a DataFrame with a single row
        filter_df = pd.DataFrame(series)

        valo = filter_df[0]
        valores = valo.reset_index(drop=True)
        df_modelo = pd.DataFrame(valores).transpose()

        df_modelo.columns = cana_spec
        df_pred = df_modelo.reset_index(drop=True)

        channels_indices = {'Engagement': ['Fp1', 'Fp2'],
                            'Fatigue': ['Fp1', 'Fp2'],
                            'Excitement': ['Fp1', 'Fp2'],
                            'Relaxation': ['C3', 'C4', 'O1', 'O2']}

        def calc_index(x, channel, name_index):
            if name_index == 'Engagement':
                return x[f'{channel}_Beta'] / (x[f'{channel}_Theta'] + x[f'{channel}_Alpha'])
            elif name_index == 'Fatigue':
                return x[f'{channel}_Alpha'] / x[f'{channel}_Theta']
            elif name_index == 'Excitement':
                return x[f'{channel}_Beta'] / x[f'{channel}_Alpha']
            elif name_index == 'Relaxation':
                return x[f'{channel}_Theta'] / x[f'{channel}_Delta']

        df_pred['Theta'] = (df_pred['Fp1_Theta'] + df_pred['Fp2_Theta'])/4
        df_pred['Alpha'] = (df_pred['Fp1_Alpha'] + df_pred['Fp2_Alpha'])/4
        df_pred['Beta'] = (df_pred['Fp1_Beta'] + df_pred['Fp2_Beta']) / 4

        for index, channels in channels_indices.items():
            a = np.zeros((df_pred.shape[0], len(channels)))
            for i, channel in enumerate(channels):
                a[:, i] = calc_index(df_pred, channel, index)
            df_pred[index] = np.mean(a, axis=1)

        return df_pred

    df_all = pd.DataFrame(columns=cana_spec)
    for n in range(df_sub.shape[0] // fs):
        df_all = pd.concat([df_all, psd_calc(df_sub.iloc[(n * fs):((n + 1) * fs), :])], axis=0)
    df_all = df_all.reset_index(drop=True)
    return df_all


info_cols = ['ID', 'Take', 'Group']
df_eeg = pd.DataFrame(columns=cana_spec + info_cols)
for subject in df.ID.unique():
    for take in df.Take.unique():
        df_temp = pross_eeg(df_sub=df[(df.ID == subject) & (df.Take == take)].drop(info_cols, axis=1))
        df_temp['ID'] = subject
        df_temp['Take'] = take
        df_temp['Group'] = 'Happiness' if int(subject) in [2, 4, 6] else 'Sadness'
        df_eeg = pd.concat([df_eeg, df_temp], axis=0, ignore_index=True)

df_eeg.reset_index(drop=True).to_csv('df_eeg.csv', index=False)
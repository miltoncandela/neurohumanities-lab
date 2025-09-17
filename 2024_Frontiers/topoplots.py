
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
data_point = np.array([[1.0, 0.5, 1.0]])  # Replace with your data values

# Create an info object with channel information
ch_names = ['EEG 1', 'T8', 'Cz']  # Replace with your channel names
ch_types = ['eeg', 'eeg', 'eeg']  # Replace with your channel types
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)

print(data_point)
print(info)

# Create an Evoked object for the custom data point
evoked = mne.EvokedArray(data_point.T, info)

montage = mne.channels.make_dig_montage(ch_pos={})

# Set the montage for the Evoked object
evoked.set_montage(montage)

# Specify the time point for plotting (in seconds)
time_point = 0.0  # Replace with your desired time point

# Plot topographical maps at the specified time point
#mne.viz.plot_topomap(evoked.data, evoked.info, show=False)
evoked.plot_topomap(times=[time_point], ch_type='eeg', time_unit='s')
plt.show()
exit()
'''


channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

data = pd.read_csv('datos/df_imp_3.csv', index_col=0)
data.index = [x.split('_')[0] for x in data.index]
print(data)
data = data.rename(columns={'Domain': 'Dominance'})

for channel in set(channels).difference(set(data.index)):
    data = pd.concat([data, pd.DataFrame(dict(zip(data.columns, [0, 0, 0])), index=[channel])], axis=0)
print(data)

montage = mne.channels.make_standard_montage("biosemi32")
n_channels = len(montage.ch_names)
data = data.reindex(montage.ch_names)
data['EII'] = data.sum(axis=1)/3
print(data)
exit()
# data = data.sort_values('EII', ascending=False)
data_sorted = data.sort_values('EII', ascending=False)
print(data_sorted)

for i in range(data_sorted.shape[0]//2):
    z = i + data_sorted.shape[0]//2
    l = ([data_sorted.index[i]] + [round(data_sorted.iloc[i, j], 4) for j in range(data_sorted.shape[1])] +
         [data_sorted.index[z]] + [round(data_sorted.iloc[z, j], 4) for j in range(data_sorted.shape[1])])
    print('{} & ${}$ & ${}$ & ${}$ & ${}$ & {} & ${}$ & ${}$ & ${}$ & ${}$ \\\\'.format(*l))
# exit()

info = mne.create_info(ch_names=montage.ch_names, sfreq=250, ch_types='eeg')
evoked = mne.EvokedArray(np.array(data), info)
evoked.set_montage(montage)

minmax = (0, data.max().max())
print(minmax)

fig, axes = plt.subplots(1, 4)
for i, ax in enumerate(axes):
    mne.viz.plot_topomap(evoked.data[:, i], evoked.info, cmap='viridis', axes=ax, show=False, vlim=minmax)
    ax.set_title(list(data.columns)[i])

cbar = fig.colorbar(axes[0].images[0], ax=axes, orientation='horizontal', shrink=0.5, pad=0.05)
cbar.set_label('Gini importance')

# plt.tight_layout()
# plt.show()
plt.savefig('GI_Gamma_Topoplot2.pdf', bbox_inches='tight')

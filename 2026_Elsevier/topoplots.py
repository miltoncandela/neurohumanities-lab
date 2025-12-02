
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
bands = ['Theta', 'Alpha', 'Beta']

df = pd.read_csv('pross/PSD_TABC_QZ_norm.csv')
print(df.shape)
# df = df[df.Subject.isin([3, 4, 6, 9])]
df = df[df.Subject != 1]
print(df.shape)
df = df.drop('Subject', axis=1)
mdf = df.groupby('Scene').median().T
print(mdf)

# minmax = (mdf.min().min(), mdf.max().max())
minmax = (-1, 1)
fig, axes = plt.subplots(3, 4, dpi=300)

for j, band in enumerate(bands):
    df = mdf.iloc[range(j, mdf.shape[0], 3), :]
    df.columns = ['Express', 'Name', 'Read', 'Fear']
    df.index = channels

    montage = mne.channels.make_standard_montage("biosemi128")
    n_channels = len(montage.ch_names)
    data = df.reindex(montage.ch_names)

    info = mne.create_info(ch_names=montage.ch_names, sfreq=250, ch_types='eeg')
    evoked = mne.EvokedArray(np.array(data), info)
    evoked.set_montage(montage)

    for i in range(df.shape[1]):
        mne.viz.plot_topomap(evoked.data[:, i], evoked.info, cmap='viridis', axes=axes[j][i], show=False, vlim=minmax)
        if j == 0:
            axes[j][i].set_title(list(data.columns)[i])
        if i == 0:
            axes[j][i].set_ylabel(band, rotation=90, labelpad=5, fontsize=12)

cbar = fig.colorbar(axes[0][0].images[0], ax=axes, orientation='vertical', shrink=0.8, pad=0.05)
plt.savefig('figs/power_Topoplot_zscore2.png', dpi=300, bbox_inches='tight')

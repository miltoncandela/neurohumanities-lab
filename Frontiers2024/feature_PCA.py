from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def create_csv():
    df = pd.read_csv('datos/df_raw.csv')
    info_columns = ['Arousal', 'Valence', 'Domain', 'Liking'] + ['Subject', 'Video']
    df_information = df[info_columns]
    df = df.drop(info_columns, axis=1)

    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Lobules and their corresponding electrode regions
    lobules = {'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'AF3', 'AF4'],
               'Temporal': ['T7', 'T8', 'TP7', 'TP8'],
               'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz', 'PO3', 'PO4'],
               'Occipital': ['O1', 'O2', 'Oz', 'O9', 'O10'],
               'Central': ['FC5', 'FC1', 'C3', 'C4', 'FC2', 'FC6', 'Cz'],
               'Posterior Cingulate': ['CP5', 'CP1', 'CP2', 'CP6']}

    short_lobules = dict(zip([''.join(lob[0] for lob in lobule.split()) for lobule in lobules.keys()], lobules.keys()))
    electrode_to_lobule = {electrode: lobule for lobule, electrodes in lobules.items() for electrode in electrodes}
    organized_channels = {lobule: [electrode for electrode in channels if electrode_to_lobule[electrode] == lobule] for lobule in lobules}

    df_PCA = pd.DataFrame(columns=[lobule + '_' + band for lobule in short_lobules.keys() for band in bands])
    subjects = df_information.Subject.unique()

    # Have 1 component for each zone
    for col in df_PCA.columns:
        s_lobule, band = col.split('_')
        curr_channs = [channel + '_' + band for channel in organized_channels[short_lobules[s_lobule]]]

        a = np.zeros(shape=df.shape[0], )
        for subject in subjects:
            curr_df = df[df_information.Subject == subject]

            a[curr_df.index[0]:curr_df.index[-1] + 1] = np.squeeze(PCA(1).fit_transform(curr_df[curr_channs]))

        df_PCA[col] = a

    df_PCA = pd.concat([df_PCA, df_information], axis=1)
    df_PCA.to_csv('datos/df_PCA_2.csv', index=False)

    return 0


# create_csv()

df_PCA = pd.read_csv('datos/df_PCA_2.csv')
print(df_PCA)
# exit()
# df_PCA = df_PCA[df_PCA.Subject == 1]
# df_PCA = df_PCA[df_PCA.Video < 5]
# channels_bands = ['{}_{}'.format(c, b) for c in channels for b in bands]
target_emotions = ['Arousal', 'Valence', 'Domain', 'Liking']
data_cols = [col for col in df_PCA.columns if col not in target_emotions + ['Subject', 'Video']]

df_corr = pd.DataFrame(columns=target_emotions)
df_ps = pd.DataFrame(columns=target_emotions)
subjects = df_PCA.Subject.unique()

for emotion in target_emotions:
    a = np.zeros(shape=(len(data_cols), ))
    b = np.zeros(shape=(len(data_cols), ))

    for subject in subjects:
        curr_df = df_PCA[df_PCA.Subject == subject]
        a += np.array([abs(pearsonr(curr_df[emotion], curr_df[col])[0]) for col in data_cols])
        b += np.array([1 if pearsonr(curr_df[emotion], curr_df[col])[1] < 0.05 else 0 for col in data_cols])

    df_corr[emotion] = np.round(a/len(subjects), 4)
    df_ps[emotion] = np.round(b/len(subjects), 2)

df_corr.index = data_cols
df_ps.index = data_cols

df_corr = df_corr.drop('Liking', axis=1)
df_ps = df_ps.drop('Liking', axis=1)


df_ps = df_ps[df_ps >= 0.95]
# df_corr.index = set([col.split('_')[0] for col in data_cols])

print(df_corr)
print(df_ps)
#exit()


# LaTeX table
lobules = ['F', 'T', 'P', 'O', 'C', 'PC']
print()

for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
    print("\multirow{6}{*}{$\\" + band.lower() + "$}")
    curr_cols = [col for col in df_corr.index if col.split('_')[1] == band]
    df_curr = df_corr.loc[curr_cols, :].reset_index(drop=True)

    for row in range(df_curr.shape[0]):
        data = [lobules[row]] + list(df_curr.iloc[row, :]) + [round(sum(df_curr.iloc[row, :]), 4)]
        print('& {} & ${}$ & ${}$ & ${}$ & ${}$ \\\\'.format(*data))

        if row < df_curr.shape[0] - 1:
            print('\cline{3-6}')
        else:
            print('\hline')

    print()

# df_corr.to_csv('datos/df_corr_3.csv')
# print(df_corr.sum(axis=1))

sns.heatmap(df_corr)
plt.show()

'''
rs = []
for short_lobule in set([col.split('_')[0] for col in data_cols]):
    curr_cols = [col for col in data_cols if col.split('_')[0] == short_lobule]

    lob_pca = np.squeeze(PCA(1).fit_transform(df_PCA[curr_cols]))
    rs.append(abs(pearsonr(df_PCA[emotion], lob_pca)[0]))
'''



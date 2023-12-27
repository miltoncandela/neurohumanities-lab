import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Gamma: Parietal, Frontal, Central
# Which Channels from the given lobes (Gamma) are more related to the three emotions?


def create_csv():
    desired_bands = ['Gamma']

    df = pd.read_csv('datos/df_raw.csv')
    info_columns = ['Arousal', 'Valence', 'Domain', 'Liking'] + ['Subject', 'Video']
    df_information = df[info_columns]
    df = df.drop(info_columns, axis=1)

    # Lobules and their corresponding electrode regions
    lobules = {'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'AF3', 'AF4'],
               'Temporal': ['T7', 'T8'],
               'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz', 'PO3', 'PO4'],
               'Occipital': ['O1', 'O2', 'Oz'],
               'Central': ['FC5', 'FC1', 'C3', 'C4', 'FC2', 'FC6', 'Cz'],
               'Posterior Cingulate': ['CP5', 'CP1', 'CP2', 'CP6']}

    desired_lobes = ['Temporal', 'Parietal', 'Central']

    desired_columns = [ch + '_' + ban for lob in desired_lobes for ch in lobules[lob] for ban in desired_bands]
    df = df[desired_columns]

    df = pd.concat([df, df_information], axis=1)
    df.to_csv('datos/df_TPC.csv', index=False)

    return 0

# create_csv()


def define_imp():
    df = pd.read_csv('datos/df_Gamma.csv')
    n_fold = 10
    # df = df[df.Video <= 5]
    # Proposal if don't work: Iterate through a set of videos (0, 5), (5, 10), (10, 15) and average per subject

    target_emotions = ['Arousal', 'Valence', 'Domain']
    info_columns = ['Arousal', 'Valence', 'Domain', 'Liking'] + ['Subject', 'Video']

    df_imp = pd.DataFrame(columns=target_emotions)
    n_data_cols = df.shape[1] - len(info_columns)
    subjects = df.Subject.unique()

    for emotion in target_emotions:
        print(emotion)

        a = np.zeros(shape=(n_data_cols,))

        for subject in subjects:
            print(subject, end=', ')

            curr_df = df[df.Subject == subject]
            y = curr_df[emotion]
            x = curr_df.drop(info_columns, axis=1)

            b = np.zeros(shape=(n_data_cols,))
            for fold in range(1, n_fold + 1):
                b += np.array(RandomForestRegressor(random_state=fold, n_jobs=1).fit(x, y).feature_importances_)
            b /= n_fold

            a += b

        df_imp[emotion] = np.round(a / len(subjects), 4)
        print()

    print()
    df_imp.index = df.columns[:n_data_cols]
    df_imp.to_csv('datos/df_imp_true.csv')
    print(df_imp)

    return 0


define_imp()

df = pd.read_csv('datos/df_imp_true.csv', index_col=0)
print(df.sum(axis=1))

print([x.split('_')[0] for x in df.index])


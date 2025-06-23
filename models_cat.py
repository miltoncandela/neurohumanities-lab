import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, accuracy_score
from pickle import dump
from scipy.stats import pearsonr
from copy import deepcopy

# numpy 1.19.5
# scipy 1.23.5

chan = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
schan = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'] #, 'Eng', 'Fat', 'Exc', 'Rel']
spec_chan = ['{}_{}'.format(c, s) for c in chan for s in bands]
spec_schan = ['{}_{}'.format(c, s) for c in schan for s in bands]
path_pross = 'data/pross/10s_norm/'
max_feat = 8

def conv_class(x):
    a = np.empty(x.shape, dtype='<U4')
    a[x <= 3] = 'Low'
    a[(x > 3) & (x <= 6)] = 'Med'
    a[x > 6] = 'Hig'
    return a

def feature_generation(df):
    """
    This function creates a variety of combined features using the provided features, it is only used in EEG and PPG
    features, because they are non-normalized and will further be normalized, and so their range varies between 0 and 1.

    :param pd.DataFrame df: Non-normalized DataFrame with continuous values, from PPG and EEG.
    :return pd.DataFrame: Returns a pandas Dataframe with combined features in addition to the previous features.
    """

    # The following variables are created:
    # df_features would be our pandas DataFrame where the normal and combined features are placed.
    # Epsilon is a constant to avoid dividing by 0.
    # Names and combinations would track the names of the combined features created.
    df = df.astype('float32')
    df_features = deepcopy(df)
    print(df_features.shape)
    eps = 1e-6 # epsilon
    names = list(df_features.columns)
    combinations = []

    # The following for loop creates a set of combined features based on the spectral signals that were generated.
    # It iterates over all the features on a separate DataFrame, and it applies a function. The result is further
    # saved on a column with the following encoding:

    # Name_i-I : Inverse on ith feature
    # Name_i-L : Logarithm on ith feature
    # Name_i-M-Name_j : Multiplication of ith feature with feature jth
    # Name_i-D-Name_j : Division of ith feature with feature jth

    # A small number on the form of a epsilon is being used to avoid NANs because some functions are 0 sensitive,
    # such as the natural logarithm and the division by 0. Moreover, a separate list "combinations" is used to keep
    # track the combinations of ith and jth features, and so not to generate duplicate features when multiplying
    # ith feature with jth feature and vice versa (as they are the same number).
    for i in range(len(df.columns)):
        names.append(df.columns[i] + '-I')
        df_features = pd.concat([df_features, np.divide(np.ones(df.shape[0]), df.loc[:, df.columns[i]])],
                                axis=1, ignore_index=True)

        names.append(df.columns[i] + '-L')
        df_features = pd.concat([df_features, pd.Series(np.log(np.abs(np.array(df.loc[:, df.columns[i]])) + 1))],
                                axis=1, ignore_index=True)

        # for j in range(len(df.columns)):
        #     if i != j:
        #         names.append(df.columns[i] + '-D-' + df.columns[j])
        #        df_features = pd.concat([df_features,
        #                                 pd.Series(np.divide(df.loc[:, df.columns[i]],
        #                                                     np.array(df.loc[:, df.columns[j]]) + eps))],
        #                                 axis=1, ignore_index=True)
        #         if [i, j] not in combinations and [j, i] not in combinations:
        #             combinations.append([i, j])
        #            combinations.append([j, i])
        #             names.append(df.columns[i] + '-M-' + df.columns[j])
        #            df_features = pd.concat([df_features,
        #                                     np.multiply(df.loc[:, df.columns[i]], df.loc[:, df.columns[j]])],
        #                                     axis=1, ignore_index=True)


    # The generated feature names are placed, infinity values from columns are removed, and the DF is returned.
    df_features.columns = names
    print(df_features.shape)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna(axis='columns', how='any')
    print(df_features.shape)
    return df_features

with open(path_pross + 'data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain, allow_pickle=True)

with open(path_pross + 'label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL, allow_pickle=True)

def add_index(x):
    eps = 1e-6  # epsilon

    # Engagement: avg bandpowers over Fp1, Fp2, C3, C4
    eng_channels = ['Fp1', 'Fp2', 'C3', 'C4']
    t = np.mean([x[ch+'_Theta'] for ch in eng_channels], axis=0)
    a = np.mean([x[ch+'_Alpha'] for ch in eng_channels], axis=0)
    b = np.mean([x[ch+'_Beta'] for ch in eng_channels], axis=0)
    x['Eng'] = b / (a + t + eps)

    # Fatigue: P7, P8, O1, O2
    fat_channels = ['P7', 'P8', 'O1', 'O2']
    t = np.mean([x[ch+'_Theta'] for ch in fat_channels], axis=0)
    a = np.mean([x[ch+'_Alpha'] for ch in fat_channels], axis=0)
    x['Fat'] = t / (a + eps)

    # Excitement: Fp1, Fp2, C3, C4, P7, P8
    exc_channels = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8']
    b = np.mean([x[ch+'_Beta'] for ch in exc_channels], axis=0)
    a = np.mean([x[ch+'_Alpha'] for ch in exc_channels], axis=0)
    x['Exc'] = b / (a + eps)

    # Relaxation: Fp1, Fp2, P7, P8, O1, O2
    exc_channels = ['Fp1', 'Fp2', 'P7', 'P8', 'O1', 'O2']
    t = np.mean([x[ch+'_Theta'] for ch in exc_channels], axis=0)
    d = np.mean([x[ch+'_Delta'] for ch in exc_channels], axis=0)
    x['Rel'] = t / (d + eps)

    return x


def pross_X(x):
    # Select the 8 OpenBCI channels
    x = pd.DataFrame(x, columns=spec_chan)[spec_schan]
    x = add_index(x)
    x = feature_generation(x)
    # cols = x.columns
    # for i, col in enumerate(cols):
    #     if col != cols[-1]:
    #         x[col+'_D_'+cols[i+1]] = x[col]/x[cols[i+1]]
    #     x[col+'_I'] = 1/x[col]
     #    x[col + '_L'] = np.log(x[col] + 1)

    return x


# 12 PSD/video * 40 videos/subject * 27 subjects
# X: (27, 480) # 27 subjects, 480 PSD/subject
# X: (12960, 160) # 12960 PSDs from all subjects, 288: 32 channels * (5 frequency bands + 4 indices)
# X = np.stack([np.stack(row) for row in X]).reshape(-1, 288)
X = np.vstack(X.reshape(-1))
print(X.shape)
X = pross_X(X)
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
print(X)

# Y = np.stack([np.stack(row) for row in Y]).reshape(-1, 4)
Y = np.vstack(Y.reshape(-1)) - 1
print(Y.shape)
# Z = np.ravel(Y[:, 1])
Z = conv_class(Y)

# mask = np.all(Y >= 6, axis=1)
# X = X[mask]
# Y = Y[mask]

print(X)
print(Y)
print(X.shape)

# model = RandomForestClassifier(random_state=1, n_jobs=1).fit(X, Z[:, 0])
# model.fit(X, Z[:, 0])
# print(classification_report(model.predict(X), Z[:, 0]))

Arousal_Train = np.ravel(Z[:, 0])
Valence_Train = np.ravel(Z[:, 1])
Domain_Train = np.ravel(Z[:, 2])

s = pd.Series(data=0, index=X.columns).sort_values(ascending=True)

while len(s) > 20:
    features = list(s.index)
    print(len(features))
    print(X.shape)
    Aimp = RandomForestClassifier(n_estimators=5, random_state=100).fit(X.loc[:, features], Arousal_Train).feature_importances_
    Vimp = RandomForestClassifier(n_estimators=5, random_state=100).fit(X.loc[:, features], Valence_Train).feature_importances_
    Dimp = RandomForestClassifier(n_estimators=5, random_state=100).fit(X.loc[:, features], Domain_Train).feature_importances_

    s = pd.Series(data=(Aimp + Vimp + Dimp) / 3, index=features).sort_values(ascending=False)
    s = s.iloc[:-10]
    print(s)

print(s)
# Aimp = np.abs([pearsonr(X[col], Arousal_Train)[0] for col in X.columns])
# Vimp = np.abs([pearsonr(X[col], Valence_Train)[0] for col in X.columns])
# Dimp = np.abs([pearsonr(X[col], Domain_Train)[0] for col in X.columns])

tindices = np.argsort((Aimp + Vimp + Dimp)/3)[::-1]
print(tindices)
# print(pd.Series(data=(Aimp + Vimp + Dimp)/3, index=X.columns).sort_values(ascending=True))
# exit()
# tindices = np.argsort((Aimp + Vimp + Dimp)/3)[::-1][:max_feat]
# X = X.iloc[:, tindices]
# print(X.shape)

with open(path_pross + 'data_testing.npy', 'rb') as fileTrain:
    M = np.load(fileTrain, allow_pickle=True)

with open(path_pross + 'label_testing.npy', 'rb') as fileTrainL:
    N = np.load(fileTrainL, allow_pickle=True)

# M = pross_X(normalize(M))
# M = np.stack([np.stack(row) for row in M]).reshape(-1, 288)
M = np.vstack(M.reshape(-1))
M = pross_X(M)
M = pd.DataFrame(scaler.transform(M))
# M = M.iloc[:, tindices]
print(M.shape)

# N = np.stack([np.stack(row) for row in N]).reshape(-1, 4)
N = np.vstack(N.reshape(-1)) - 1
L = conv_class(N)

# mask = np.all(N >= 6, axis=1)
# M = M[mask]
# N = N[mask]
# print(M.shape)

Arousal_Test = np.ravel(L[:, 0])
Valence_Test = np.ravel(L[:, 1])
Domain_Test = np.ravel(L[:, 2])

def train_models(n):

    print(n)
    # tindices = np.argsort((Aimp + Vimp + Dimp) / 3)[::-1][:max_feat]
    # print(tindices[:n])
    # exit()
    # model = LinearRegression()
    model = RandomForestClassifier(n_estimators=10, random_state=1)
    Val_R = model.fit(X.iloc[:, tindices[:n]], Valence_Train)
    Aro_R = model.fit(X.iloc[:, tindices[:n]], Arousal_Train)
    Dom_R = model.fit(X.iloc[:, tindices[:n]], Domain_Train)

    val_pred = Val_R.predict(M.iloc[:, tindices[:n]])
    aro_pred = Aro_R.predict(M.iloc[:, tindices[:n]])
    dom_pred = Dom_R.predict(M.iloc[:, tindices[:n]])
    # print(round(min(val_pred), 2), round(min(aro_pred), 2), round(min(dom_pred), 2),
    #       round(max(val_pred), 2), round(max(aro_pred), 2), round(max(dom_pred), 2))

    # r2_val = 1 - np.sum((Valence_Test - val_pred) ** 2)/np.sum((Valence_Test - np.mean(Valence_Test))**2)
    # r2_aro = 1 - np.sum((Arousal_Test - aro_pred) ** 2)/np.sum((Arousal_Test - np.mean(Valence_Test))**2)
    # r2_dom = 1 - np.sum((Domain_Test - dom_pred) ** 2)/np.sum((Domain_Test - np.mean(Valence_Test))**2)
    # r2_val = r2_score(Valence_Test, val_pred)
    # r2_aro = r2_score(Arousal_Test, aro_pred)
    # r2_dom = r2_score(Domain_Test, dom_pred)
    # print("R-squared:", round(r2_val, 3), round(r2_aro, 3), round(r2_dom, 3))

    p_val = accuracy_score(Valence_Test, val_pred)
    p_aro = accuracy_score(Arousal_Test, aro_pred)
    p_dom = accuracy_score(Domain_Test, dom_pred)
    print("Acc:", round(p_val, 3), round(p_aro, 3), round(p_dom, 3))

    # print(classification_report(Valence_Test, val_pred))
    # print(classification_report(Arousal_Test, aro_pred))
    # print(classification_report(Domain_Test, dom_pred))

    val_pred = Val_R.predict(X.iloc[:, tindices[:n]])
    aro_pred = Aro_R.predict(X.iloc[:, tindices[:n]])
    dom_pred = Dom_R.predict(X.iloc[:, tindices[:n]])

    # r2_val = 1 - np.sum((Valence_Train - val_pred) ** 2) / np.sum((Valence_Train - np.mean(Valence_Train)) ** 2)
    # r2_aro = 1 - np.sum((Arousal_Train - aro_pred) ** 2) / np.sum((Arousal_Train - np.mean(Arousal_Train)) ** 2)
    # r2_dom = 1 - np.sum((Domain_Train - dom_pred) ** 2) / np.sum((Domain_Train - np.mean(Domain_Train)) ** 2)
    # print("R-squared:", round(r2_val, 3), round(r2_aro, 3), round(r2_dom, 3))

    p_val = accuracy_score(Valence_Train, val_pred)
    p_aro = accuracy_score(Arousal_Train, aro_pred)
    p_dom = accuracy_score(Domain_Train, dom_pred)
    print("Acc:", round(p_val, 3), round(p_aro, 3), round(p_dom, 3))

    # print(classification_report(Valence_Train, val_pred))
    # print(classification_report(Arousal_Train, aro_pred))
    # print(classification_report(Domain_Train, dom_pred))
    print()

    if n > 20: # == 2:
        with open('models/reg_val_model2_10s.pkl', 'wb') as file:
            dump(Val_R, file)
        with open('models/reg_aro_model2_10s.pkl', 'wb') as file:
            dump(Aro_R, file)
        with open('models/reg_dom_model2_10s.pkl', 'wb') as file:
            dump(Dom_R, file)
        exit()


for n_feat in range(1, max_feat):
    train_models(n_feat)

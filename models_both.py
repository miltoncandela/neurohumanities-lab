import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, accuracy_score
from pickle import dump
from scipy.stats import pearsonr
from copy import deepcopy

# Previous versions: numpy 1.19.5, scipy 1.23.5, scikit-learn 0.24.1
chan = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
schan = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Eng', 'Fat', 'Exc', 'Rel']
spec_chan = ['{}_{}'.format(c, s) for c in chan for s in bands]
spec_schan = ['{}_{}'.format(c, s) for c in schan for s in bands]
path_pross = 'data/pross/10s_norm/'
max_feat = 8

def conv_class(x):
    a = np.empty(x.shape, dtype='<U4')
    a[x <= 3.65] = 'Low'
    a[(x > 3.65) & (x <= 6.35)] = 'Med'
    a[x > 6.35] = 'Hig'
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
        # names.append(df.columns[i] + '-I')
        # df_features = pd.concat([df_features, np.divide(np.ones(df.shape[0]), df.loc[:, df.columns[i]])], axis=1, ignore_index=True)

        # names.append(df.columns[i] + '-L')
        # df_features = pd.concat([df_features, pd.Series(np.log(np.abs(np.array(df.loc[:, df.columns[i]])) + 1))], axis=1, ignore_index=True)

        for j in range(len(df.columns)):
            if i != j and (df.columns[i].split('_')[0] == df.columns[j].split('_')[0]):
                a = np.array(df.loc[:, df.columns[i]])
                b = np.array(df.loc[:, df.columns[j]] + eps)
                names.append(df.columns[i] + '-D-' + df.columns[j])
                df_features = pd.concat([df_features, pd.Series(np.divide(a, b))], axis=1, ignore_index=True)
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
    # x = add_index(x)
    # x = feature_generation(x)
    # x = x.drop(spec_schan, axis=1)
    return x


with open(path_pross + 'data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain, allow_pickle=True)

with open(path_pross + 'label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL, allow_pickle=True)

# 12 PSD/video * 40 videos/subject * 27 subjects
# X: (27, 480) # 27 subjects, 480 PSD/subject
# X: (12960, 160) # 12960 PSDs from all subjects, 288: 32 channels * (5 frequency bands + 4 indices)
X = np.vstack(X.reshape(-1))
X = pross_X(X)
# scaler = StandardScaler().fit(X)
# X = pd.DataFrame(scaler.transform(X), columns=X.columns)

Y = np.vstack(Y.reshape(-1))
Z = conv_class(Y)

print(X)
print(Y)
print(X.shape)

# Prior mask
# mask = np.all(Y <= 3, axis=1)
# X_low, Y_low = X[mask], Y[mask]
# mask = np.all((Y > 3) & (Y <= 6), axis=1)
# X_med, Y_med = X[mask], Y[mask]
# mask = np.all(Y > 6, axis=1)
# X_hig, Y_hig = X[mask], Y[mask]

# Emotion component-specific mask
mask = Y[:, 1] <= 3
X_lowV, Y_lowV = X[mask], Y[mask]
mask = (Y[:, 1] > 3) & (Y[:, 1] <= 6)
X_medV, Y_medV = X[mask], Y[mask]
mask = Y[:, 1] > 6
X_higV, Y_higV = X[mask], Y[mask]

mask = Y[:, 0] <= 3
X_lowA, Y_lowA = X[mask], Y[mask]
mask = (Y[:, 0] > 3) & (Y[:, 0] <= 6)
X_medA, Y_medA = X[mask], Y[mask]
mask = Y[:, 0] > 6
X_higA, Y_higA = X[mask], Y[mask]

mask = Y[:, 2] <= 3
X_lowD, Y_lowD = X[mask], Y[mask]
mask = (Y[:, 2] > 3) & (Y[:, 2] <= 6)
X_medD, Y_medD = X[mask], Y[mask]
mask = Y[:, 2] > 6
X_higD, Y_higD = X[mask], Y[mask]

# Min-max normalization
# Y = Y - 5

# Training using categorical and regression features
TRA_Cat, TRV_Cat, TRD_Cat = np.ravel(Z[:, 0]), np.ravel(Z[:, 1]), np.ravel(Z[:, 2])  # Training Arousal_Categorical
TRA_Reg, TRV_Reg, TRD_Reg = np.ravel(Y[:, 0]), np.ravel(Y[:, 1]), np.ravel(Y[:, 2])  # Training Arousal_Regression
sA = pd.Series(data=0, index=X.columns).sort_values(ascending=True)
sV = pd.Series(data=0, index=X.columns).sort_values(ascending=True)
sD = pd.Series(data=0, index=X.columns).sort_values(ascending=True)

while len(sA) > 15:
    featuresA, featuresV, featuresD = list(sA.index), list(sV.index), list(sD.index)
    Aimp = RandomForestRegressor(n_estimators=5, max_depth=10, random_state=100).fit(X.loc[:, featuresA], TRA_Reg).feature_importances_
    Vimp = RandomForestRegressor(n_estimators=5, max_depth=10, random_state=100).fit(X.loc[:, featuresV], TRV_Reg).feature_importances_
    Dimp = RandomForestRegressor(n_estimators=5, max_depth=10, random_state=100).fit(X.loc[:, featuresD], TRD_Reg).feature_importances_

    sA = pd.Series(data=Aimp, index=featuresA).sort_values(ascending=False)
    sV = pd.Series(data=Vimp, index=featuresV).sort_values(ascending=False)
    sD = pd.Series(data=Dimp, index=featuresD).sort_values(ascending=False)

    sA, sV, sD = sA.iloc[:-10], sV.iloc[:-10], sD.iloc[:-10]
    print(sA)
featuresA, featuresV, featuresD = list(sA.index), list(sV.index), list(sD.index)
print()
print(sA)
print(sV)
print(sD)
# Fp2_Theta    0.184031
# P7_Theta     0.078534
# C3_Gamma     0.070599
# O1_Gamma     0.056738
# O2_Gamma     0.051193
# Fp1_Beta     0.044235

with open(path_pross + 'data_testing.npy', 'rb') as fileTrain:
    M = np.load(fileTrain, allow_pickle=True)

with open(path_pross + 'label_testing.npy', 'rb') as fileTrainL:
    N = np.load(fileTrainL, allow_pickle=True)

M = np.vstack(M.reshape(-1))
M = pross_X(M)
# M = pd.DataFrame(scaler.transform(M), columns=M.columns)
# M = M.iloc[:, tindices]
print(M.shape)

N = np.vstack(N.reshape(-1))
L = conv_class(N)

# N = N - 5

# Testing Regression/Categorical data distribution
TEA_Cat, TEV_Cat, TED_Cat = np.ravel(L[:, 0]), np.ravel(L[:, 1]), np.ravel(L[:, 2])
TEA_Reg, TEV_Reg, TED_Reg = np.ravel(N[:, 0]), np.ravel(N[:, 1]), np.ravel(N[:, 2])


def calc_perf(true, mode, pred):
    TV, TA, TD = true
    PV, PA, PD = pred

    if mode == 'reg':
        p_val = pearsonr(TV, PV)[0]
        p_aro = pearsonr(TA, PA)[0]
        p_dom = pearsonr(TD, PD)[0]
        print('r:', round(p_val, 3), round(p_aro, 3), round(p_dom, 3))

    elif mode == 'cat':
        acc_val = accuracy_score(TV, PV)
        acc_aro = accuracy_score(TA, PA)
        acc_dom = accuracy_score(TD, PD)
        print('Acc:', round(acc_val, 3), round(acc_aro, 3), round(acc_dom, 3))


print(min(TRV_Reg), max(TRV_Reg), min(TRA_Reg), max(TRA_Reg), min(TRD_Reg), max(TRD_Reg))
print(min(TEV_Reg), max(TEV_Reg), min(TEA_Reg), max(TEA_Reg), min(TED_Reg), max(TED_Reg))


def train_models(n):

    print(n)
    Val_C = RandomForestClassifier(n_estimators=5, max_depth=7, random_state=1).fit(X.loc[:, featuresV[:n]], TRV_Cat)
    Aro_C = RandomForestClassifier(n_estimators=5, max_depth=7, random_state=1).fit(X.loc[:, featuresA[:n]], TRA_Cat)
    Dom_C = RandomForestClassifier(n_estimators=5, max_depth=7, random_state=1).fit(X.loc[:, featuresD[:n]], TRD_Cat)

    predTA_val, predTA_aro = Val_C.predict(X.loc[:, featuresV[:n]]), Aro_C.predict(X.loc[:, featuresA[:n]])
    predTA_dom = Dom_C.predict(X.loc[:, featuresD[:n]])
    print('Training ', end='')
    calc_perf([TRV_Cat, TRA_Cat, TRD_Cat], 'cat', [predTA_val, predTA_aro, predTA_dom])

    predTE_val, predTE_aro = Val_C.predict(M.loc[:, featuresV[:n]]), Aro_C.predict(M.loc[:, featuresA[:n]])
    predTE_dom = Dom_C.predict(M.loc[:, featuresD[:n]])
    print('Testing ', end='')
    calc_perf([TEV_Cat, TEA_Cat, TED_Cat], 'cat', [predTE_val, predTE_aro, predTE_dom])

    # Traditional Approach
    # Val_R = RandomForestRegressor(n_estimators=5, max_depth=6, criterion='absolute_error', random_state=1).fit(X.loc[:, features[:n]], TRV_Reg)
    # Aro_R = RandomForestRegressor(n_estimators=5, max_depth=6, criterion='absolute_error', random_state=1).fit(X.loc[:, features[:n]], TRA_Reg)
    # Dom_R = RandomForestRegressor(n_estimators=5, max_depth=6, criterion='absolute_error', random_state=1).fit(X.loc[:, features[:n]], TRD_Reg)

    Val_R = HistGradientBoostingRegressor(max_depth=7, loss='quantile', quantile=0.5, random_state=1).fit(X.loc[:, featuresV[:n]], TRV_Reg)
    Aro_R = HistGradientBoostingRegressor(max_depth=7, loss='quantile', quantile=0.5, random_state=1).fit(X.loc[:, featuresA[:n]], TRA_Reg)
    Dom_R = HistGradientBoostingRegressor(max_depth=7, loss='quantile', quantile=0.5, random_state=1).fit(X.loc[:, featuresD[:n]], TRD_Reg)

    print('Training ', end='')
    predTA_val, predTA_aro = Val_R.predict(X.loc[:, featuresV[:n]]), Aro_R.predict(X.loc[:, featuresA[:n]])
    predTA_dom = Dom_R.predict(X.loc[:, featuresD[:n]])
    calc_perf([TRV_Reg, TRA_Reg, TRD_Reg], 'reg', [predTA_val, predTA_aro, predTA_dom])

    print(round(min(predTA_val), 2), round(max(predTA_val), 2), round(min(predTA_aro), 2), round(max(predTA_aro), 2),
          round(min(predTA_dom), 2), round(max(predTA_dom), 2))

    print('Testing ', end='')
    predTE_val, predTE_aro = Val_R.predict(M.loc[:, featuresV[:n]]), Aro_R.predict(M.loc[:, featuresA[:n]])
    predTE_dom = Dom_R.predict(M.loc[:, featuresD[:n]])
    calc_perf([TEV_Reg, TEA_Reg, TED_Reg], 'reg', [predTE_val, predTE_aro, predTE_dom])

    print(round(min(predTE_val), 2), round(max(predTE_val), 2), round(min(predTE_aro), 2), round(max(predTE_aro), 2),
          round(min(predTE_dom), 2), round(max(predTE_dom), 2))

    # Classif + Regression (Work in Progress)
    '''
    VL_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_lowV.loc[:, features[:n]], Y_lowV[:, 1])
    VM_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_medV.loc[:, features[:n]], Y_medV[:, 1])
    VH_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_higV.loc[:, features[:n]], Y_higV[:, 1])

    low_pred = VL_R.predict(X_lowV.loc[:, features[:n]])
    med_pred = VM_R.predict(X_medV.loc[:, features[:n]])
    hig_pred = VH_R.predict(X_higV.loc[:, features[:n]])

    print(round(min(low_pred), 2), round(min(med_pred), 2), round(min(hig_pred), 2),
          round(max(low_pred), 2), round(max(med_pred), 2), round(max(hig_pred), 2))

    AL_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_lowA.loc[:, features[:n]], Y_lowA[:, 0])
    AM_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_medA.loc[:, features[:n]], Y_medA[:, 0])
    AH_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_higA.loc[:, features[:n]], Y_higA[:, 0])

    DL_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_lowD.loc[:, features[:n]], Y_lowD[:, 2])
    DM_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_medD.loc[:, features[:n]], Y_medD[:, 2])
    DH_R = RandomForestRegressor(n_estimators=5, random_state=1).fit(X_higD.loc[:, features[:n]], Y_higD[:, 2])

    try:

        df = X.loc[:, features[:n]].copy()
        df['V'] = predTA_val
        df_low, df_med, df_hig = df[df['V'] == 'Low'], df[df['V'] == 'Med'], df[df['V'] == 'Hig']
        s_low = pd.Series(VL_R.predict(df_low.drop(['V'], axis=1)), index=df_low.index)
        s_med = pd.Series(VM_R.predict(df_med.drop(['V'], axis=1)), index=df_med.index)
        s_hig = pd.Series(VH_R.predict(df_hig.drop(['V'], axis=1)), index=df_hig.index)
        pred_val = pd.concat([s_hig, s_med, s_low]).sort_index()
        # df['Val_pred'] = pred_val
        # print(df)
        # exit()
        df = df.drop(['V'], axis=1)

        df['A'] = predTA_aro
        df_hig, df_med, df_low = df[df['A'] == 'Hig'], df[df['A'] == 'Med'], df[df['A'] == 'Low']
        s_low = pd.Series(AL_R.predict(df_low.drop(['A'], axis=1)), index=df_low.index)
        s_med = pd.Series(AM_R.predict(df_med.drop(['A'], axis=1)), index=df_med.index)
        s_hig = pd.Series(AH_R.predict(df_hig.drop(['A'], axis=1)), index=df_hig.index)
        pred_aro = pd.concat([s_hig, s_med, s_low]).sort_index()
        df = df.drop(['A'], axis=1)

        df['D'] = predTA_dom
        df_hig, df_med, df_low = df[df['D'] == 'Hig'], df[df['D'] == 'Med'], df[df['D'] == 'Low']
        s_low = pd.Series(DL_R.predict(df_low.drop(['D'], axis=1)), index=df_low.index)
        s_med = pd.Series(DM_R.predict(df_med.drop(['D'], axis=1)), index=df_med.index)
        s_hig = pd.Series(DH_R.predict(df_hig.drop(['D'], axis=1)), index=df_hig.index)
        pred_dom = pd.concat([s_hig, s_med, s_low]).sort_index()
        df = df.drop(['D'], axis=1)

        print('Training ', end='')
        calc_perf([TRV_Reg, TRA_Reg, TRD_Reg], 'reg', [pred_val, pred_aro, pred_dom])

        df = M.loc[:, features[:n]].copy()
        df['V'] = predTE_val
        df_low, df_med, df_hig = df[df['V'] == 'Low'], df[df['V'] == 'Med'], df[df['V'] == 'Hig']
        s_low = pd.Series(VL_R.predict(df_low.drop(['V'], axis=1)), index=df_low.index)
        s_med = pd.Series(VM_R.predict(df_med.drop(['V'], axis=1)), index=df_med.index)
        s_hig = pd.Series(VH_R.predict(df_hig.drop(['V'], axis=1)), index=df_hig.index)
        pred_val = pd.concat([s_hig, s_med, s_low]).sort_index()
        df = df.drop(['V'], axis=1)

        df['A'] = predTE_aro
        df_hig, df_med, df_low = df[df['A'] == 'Hig'], df[df['A'] == 'Med'], df[df['A'] == 'Low']
        s_low = pd.Series(AL_R.predict(df_low.drop(['A'], axis=1)), index=df_low.index)
        s_med = pd.Series(AM_R.predict(df_med.drop(['A'], axis=1)), index=df_med.index)
        s_hig = pd.Series(AH_R.predict(df_hig.drop(['A'], axis=1)), index=df_hig.index)
        pred_aro = pd.concat([s_hig, s_med, s_low]).sort_index()
        df = df.drop(['A'], axis=1)

        df['D'] = predTE_dom
        df_hig, df_med, df_low = df[df['D'] == 'Hig'], df[df['D'] == 'Med'], df[df['D'] == 'Low']
        s_low = pd.Series(DL_R.predict(df_low.drop(['D'], axis=1)), index=df_low.index)
        s_med = pd.Series(DM_R.predict(df_med.drop(['D'], axis=1)), index=df_med.index)
        s_hig = pd.Series(DH_R.predict(df_hig.drop(['D'], axis=1)), index=df_hig.index)
        pred_dom = pd.concat([s_hig, s_med, s_low]).sort_index()
        df = df.drop(['D'], axis=1)

        print('Testing ', end='')
        calc_perf([TEV_Reg, TEA_Reg, TED_Reg], 'reg', [pred_val, pred_aro, pred_dom])
    except ValueError:
        pass
    '''

    print()

    # if n == 5:
    #     with open('models/reg_val_model2_10s.pkl', 'wb') as file:
    #         dump(Val_R, file)
    #     with open('models/reg_aro_model2_10s.pkl', 'wb') as file:
    #         dump(Aro_R, file)
    #     with open('models/reg_dom_model2_10s.pkl', 'wb') as file:
    #         dump(Dom_R, file)

    if n in [3,4]: # [3, 4]:
        print(n)
        print(featuresV, featuresA, featuresD)
        with open(rf'OnlineProcessing/Models/reg_val_model2_10s_{n}f.pkl', 'wb') as file:
            dump(Val_R, file)
        with open(rf'OnlineProcessing/Models/reg_aro_model2_10s_{n}f.pkl', 'wb') as file:
            dump(Aro_R, file)
        with open(rf'OnlineProcessing/Models/reg_dom_model2_10s_{n}f.pkl', 'wb') as file:
            dump(Dom_R, file)

for n_feat in range(1, max_feat):
    train_models(n_feat)

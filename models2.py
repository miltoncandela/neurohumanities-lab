import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from pickle import dump
from scipy.stats import pearsonr

chan = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
schan = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Eng', 'Fat', 'Exc', 'Rel']
spec_chan = ['{}_{}'.format(c, s) for c in chan for s in bands]
spec_schan = ['{}_{}'.format(c, s) for c in schan for s in bands]
path_pross = 'data/pross/10s_norm/'
max_feat = 20

def conv_class(x):
    a = np.empty(x.shape, dtype='<U4')
    a[x <= 3] = 'Low'
    a[(x > 3) & (x <= 6)] = 'Med'
    a[x > 6] = 'Hig'
    return a


with open(path_pross + 'data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain, allow_pickle=True)

with open(path_pross + 'label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL, allow_pickle=True)

def pross_X(x):
    # Select the 8 OpenBCI channels
    x = pd.DataFrame(x, columns=spec_chan)[spec_schan]
    cols = x.columns
    for i, col in enumerate(cols):
        if col != cols[-1]:
            x[col+'_D_'+cols[i+1]] = x[col]/x[cols[i+1]]
        x[col+'_I'] = 1/x[col]
        x[col + '_L'] = np.log(x[col] + 1)

    return x


# 12 PSD/video * 40 videos/subject * 27 subjects
# X: (27, 480) # 27 subjects, 480 PSD/subject
# X: (12960, 160) # 12960 PSDs from all subjects, 288: 32 channels * (5 frequency bands + 4 indices)
# X = np.stack([np.stack(row) for row in X]).reshape(-1, 288)
X = np.vstack(X.reshape(-1))
print(X.shape)
X = pross_X(X)
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))

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

Arousal_Train = np.ravel(Y[:, 0])
Valence_Train = np.ravel(Y[:, 1])
Domain_Train = np.ravel(Y[:, 2])

Aimp = RandomForestRegressor(random_state=100).fit(X, Arousal_Train).feature_importances_
Vimp = RandomForestRegressor(random_state=100).fit(X, Valence_Train).feature_importances_
Dimp = RandomForestRegressor(random_state=100).fit(X, Domain_Train).feature_importances_

# Aimp = np.abs([pearsonr(X[col], Arousal_Train)[0] for col in X.columns])
# Vimp = np.abs([pearsonr(X[col], Valence_Train)[0] for col in X.columns])
# Dimp = np.abs([pearsonr(X[col], Domain_Train)[0] for col in X.columns])
print(Aimp)

tindices = np.argsort((Aimp + Vimp + Dimp)/3)[::-1]
print(tindices)
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
L = np.ravel(N[:, 1])

# mask = np.all(N >= 6, axis=1)
# M = M[mask]
# N = N[mask]
# print(M.shape)

Arousal_Test = np.ravel(N[:, 0])
Valence_Test = np.ravel(N[:, 1])
Domain_Test = np.ravel(N[:, 2])

def train_models(n):

    print(n)
    # tindices = np.argsort((Aimp + Vimp + Dimp) / 3)[::-1][:max_feat]
    # print(tindices[:n])
    # exit()
    # model = LinearRegression()
    model = RandomForestRegressor(random_state=1)
    Val_R = model.fit(X.iloc[:, tindices[:n]], Valence_Train)
    Aro_R = model.fit(X.iloc[:, tindices[:n]], Arousal_Train)
    Dom_R = model.fit(X.iloc[:, tindices[:n]], Domain_Train)

    val_pred = Val_R.predict(M.iloc[:, tindices[:n]])
    aro_pred = Aro_R.predict(M.iloc[:, tindices[:n]])
    dom_pred = Dom_R.predict(M.iloc[:, tindices[:n]])
    print(round(min(val_pred), 2), round(min(aro_pred), 2), round(min(dom_pred), 2),
          round(max(val_pred), 2), round(max(aro_pred), 2), round(max(dom_pred), 2))

    r2_val = 1 - np.sum((Valence_Test - val_pred) ** 2)/np.sum((Valence_Test - np.mean(Valence_Test))**2)
    r2_aro = 1 - np.sum((Arousal_Test - aro_pred) ** 2)/np.sum((Arousal_Test - np.mean(Valence_Test))**2)
    r2_dom = 1 - np.sum((Domain_Test - dom_pred) ** 2)/np.sum((Domain_Test - np.mean(Valence_Test))**2)
    # r2_val = r2_score(Valence_Test, val_pred)
    # r2_aro = r2_score(Arousal_Test, aro_pred)
    # r2_dom = r2_score(Domain_Test, dom_pred)
    print("R-squared:", round(r2_val, 3), round(r2_aro, 3), round(r2_dom, 3))

    mae_val = mean_absolute_error(Valence_Test, val_pred)
    mae_aro = mean_absolute_error(Arousal_Test, aro_pred)
    mae_dom = mean_absolute_error(Domain_Test, dom_pred)
    print("MAE:", round(mae_val, 3), round(mae_aro, 3), round(mae_dom, 3))

    val_pred = Val_R.predict(X.iloc[:, tindices[:n]])
    aro_pred = Aro_R.predict(X.iloc[:, tindices[:n]])
    dom_pred = Dom_R.predict(X.iloc[:, tindices[:n]])

    r2_val = 1 - np.sum((Valence_Train - val_pred) ** 2) / np.sum((Valence_Train - np.mean(Valence_Train)) ** 2)
    r2_aro = 1 - np.sum((Arousal_Train - aro_pred) ** 2) / np.sum((Arousal_Train - np.mean(Arousal_Train)) ** 2)
    r2_dom = 1 - np.sum((Domain_Train - dom_pred) ** 2) / np.sum((Domain_Train - np.mean(Domain_Train)) ** 2)
    print("R-squared:", round(r2_val, 3), round(r2_aro, 3), round(r2_dom, 3))

    mae_val = mean_absolute_error(Valence_Train, val_pred)
    mae_aro = mean_absolute_error(Arousal_Train, aro_pred)
    mae_dom = mean_absolute_error(Domain_Train, dom_pred)
    print("MAE:", round(mae_val, 3), round(mae_aro, 3), round(mae_dom, 3))
    print()



    if n > 20: # == 2:
        with open('models/reg_val_model2_10s.pkl', 'wb') as file:
            dump(Val_R, file)
        with open('models/reg_aro_model2_10s.pkl', 'wb') as file:
            dump(Aro_R, file)
        with open('models/reg_dom_model2_10s.pkl', 'wb') as file:
            dump(Dom_R, file)
        exit()


for n_feat in range(2, max_feat, 2):
    train_models(n_feat)

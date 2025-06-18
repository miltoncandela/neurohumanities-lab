import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report

chan = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
schan = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Eng', 'Fat', 'Exc', 'Rel']
spec_chan = ['{}_{}'.format(c, s) for c in chan for s in bands]
spec_schan = ['{}_{}'.format(c, s) for c in schan for s in bands]
path_pross = 'data/pross/3s_norm/'

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
    return pd.DataFrame(x, columns=spec_chan)[spec_schan]


# 12 PSD/video * 40 videos/subject * 27 subjects
# X: (27, 480) # 27 subjects, 480 PSD/subject
# X: (12960, 160) # 12960 PSDs from all subjects, 288: 32 channels * (5 frequency bands + 4 indices)
# X = np.stack([np.stack(row) for row in X]).reshape(-1, 288)
X = np.vstack(X.reshape(-1))
print(X.shape)
X = pross_X(X)

# Y = np.stack([np.stack(row) for row in Y]).reshape(-1, 4)
Y = np.vstack(Y.reshape(-1))
print(Y.shape)
# Z = np.ravel(Y[:, 1])
Z = conv_class(Y)

print(X)
print(Z)

# model = RandomForestClassifier(random_state=1, n_jobs=1).fit(X, Z[:, 0])
# model.fit(X, Z[:, 0])
# print(classification_report(model.predict(X), Z[:, 0]))

Arousal_Train = np.ravel(Y[:, 0])
Valence_Train = np.ravel(Y[:, 1])
Domain_Train = np.ravel(Y[:, 2])

# Aimp = RandomForestRegressor(random_state=1, n_jobs=1).fit(X, Arousal_Train).feature_importances_
# Vimp = RandomForestRegressor(random_state=1, n_jobs=1).fit(X, Valence_Train).feature_importances_
# Dimp = RandomForestRegressor(random_state=1, n_jobs=1).fit(X, Domain_Train).feature_importances_

# tindices = np.argsort((Aimp + Vimp + Dimp)/3)[::-1][:20]
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
# M = M.iloc[:, tindices]
print(M.shape)

# N = np.stack([np.stack(row) for row in N]).reshape(-1, 4)
N = np.vstack(N.reshape(-1))
L = np.ravel(N[:, 1])

Arousal_Test = np.ravel(N[:, 0])
Valence_Test = np.ravel(N[:, 1])
Domain_Test = np.ravel(N[:, 2])

from pickle import dump

# Entrenamiento de Regresores
print('Entrenando modelo Valence...')
Val_R = RandomForestRegressor(random_state=1)
#Val_R = LinearRegression()
# print(X[0:468480:16].shape)
# Val_R.fit(X[0:468480:16], Valence_Train[0:468480:16])
Val_R.fit(X, Valence_Train)
with open('reg_val_model2.pkl', 'wb') as file:
    dump(Val_R, file)
print('Modelo Valence entrenado!')

print('Entrenando modelo Arousal...')
Aro_R = RandomForestRegressor(random_state=1)
#Aro_R = LinearRegression()
# Aro_R.fit(X[0:468480:16], Arousal_Train[0:468480:16])
Aro_R.fit(X, Arousal_Train)
with open('reg_aro_model2.pkl', 'wb') as file:
    dump(Aro_R, file)
print('Modelo Arousal entrenado!')

print('Entrenando modelo Dominance...')
# Dom_R = RandomForestRegressor(n_estimators=512, n_jobs=6)
Dom_R = RandomForestRegressor(random_state=1)
#Dom_R = LinearRegression()
# Dom_R.fit(X[0:468480:16], Domain_Train[0:468480:16])
Dom_R.fit(X, Domain_Train)
with open('reg_dom_model2.pkl', 'wb') as file:
    dump(Dom_R, file)
print('Modelo Dominance entrenado!')

val_pred = Val_R.predict(M)
aro_pred = Aro_R.predict(M)
dom_pred = Dom_R.predict(M)

# r2_score, mean_absolute_error, mean_squared_error
# r2_val = r2_score(Valence_Test[0:74240:512], val_pred)
r2_val = r2_score(Valence_Test, val_pred)
print("R-squared Score Valence:", r2_val)
r2_aro = r2_score(Arousal_Test, aro_pred)
print("R-squared Score Arousal:", r2_aro)
r2_dom = r2_score(Domain_Test, dom_pred)
print("R-squared Score Dominance:", r2_dom)

mae_val = mean_absolute_error(Valence_Test, val_pred)
print("Mean Absolute Error Valence:", mae_val)
mae_aro = mean_absolute_error(Arousal_Test, aro_pred)
print("Mean Absolute Error Arousal:", mae_aro)
mae_dom = mean_absolute_error(Domain_Test, dom_pred)
print("Mean Absolute Error Dominance:", mae_dom)

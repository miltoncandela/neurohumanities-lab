import numpy as np
import pandas as pd
import pyeeg as pe
import pickle as pickle
from warnings import filterwarnings

filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Configuración para la función np.load para cargar archivos.
# Los valores por defecto establecidos incluyen el uso de codificación ASCII.
np.load.__defaults__=(None, True, True, 'ASCII')

# Definición de los canales EEG que se desean utilizar (8 canales de openbci).
# channel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
channels_bands = ['{}_{}'.format(c, b) for c in channels for b in bands]

# Frecuencias de las bandas para calcular el espectro de potencia (PSD) de la señal EEG.
band = [.5, 4, 8, 12, 30, 45]  # Definición de 5 bandas.

# Tamaño de ventana para promediar la potencia en bandas durante 5 segundos (downsampleado a 128 Hz).
window_size = 128 * 5  # Ventana de 5 segundos a una frecuencia de muestreo de 128 Hz.

# Paso para actualizar el tiempo cada ciertos pasos.
step_size = 16  # Actualizar cada 0.125 segundos.

# Frecuencia de muestreo de la señal EEG.
sample_rate = 128  # Frecuencia de muestreo de 128 Hz.

# Lista de sujetos para cargar los datos.
# subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
subjectList = ['0' + str(x) if x < 10 else str(x) for x in range(1, 33)]


# Definición de la función para procesar datos EEG
# def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
# Lista para almacenar metadatos
meta = []
target_feat = ['Arousal', 'Valence', 'Domain', 'Liking']
df_tot = pd.DataFrame(columns=channels_bands + target_feat + ['Subject', 'Video'])

for subject in subjectList:
    print(subject)
    meta = []

    # Abrir los datos usando pickle para descomprimir
    with open('datos\s' + subject + '.dat', 'rb') as file:
        subject_p = pickle.load(file, encoding='latin1')  # Resuelve el problema de datos de Python 2 utilizando la codificación latin1

        for i in range(0, 40):
            # Para las 40 pruebas realizadas
            data = subject_p["data"][i]
            labels = subject_p["labels"][i]
            start = 0

            print(data.shape[1])
            exit()

            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []  # Vector de metadatos para el análisis
                for j in range(len(channels)):
                    X = data[j][start: start + window_size]  # Dividir los datos crudos en ventanas de 2 segundos, con un intervalo de 0.125 segundos
                    Y = pe.bin_power(X, band, sample_rate)  # FFT en 5 segundos del canal j, en secuencia de theta, alpha, beta baja, beta alta, gamma
                    meta_data = meta_data + list(Y[0])

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start = start + step_size

            # print(len(meta))
            # print(meta)
            # exit()
        meta = np.array(meta)
        n_col = len(channels_bands) + len(target_feat)
        a = np.zeros(shape=(meta.shape[0], n_col))

        for i in range(meta.shape[0]):
            a[i] = np.concatenate([np.array(meta[i][0]), np.array(meta[i][1])]).reshape(1, n_col)

        df = pd.DataFrame(a, columns=channels_bands + target_feat)
        print(df.shape[0])
        exit()
        df['Subject'] = subject

        # meta.shape = (18560, 2) = x, y
        # df.shape = (18560, 164)
        # df.shape[0] / 40 = 464, cada 464 valores cambia de video

        n_subjects = len(subjectList)
        n_videos = 40
        length_videos = int(df.shape[0] / n_videos)
        l_vid = [[i]*length_videos for i in range(1, n_videos + 1)]
        l_vid = [x for y in l_vid for x in y]
        df['Video'] = l_vid

    df_tot = pd.concat([df_tot, df], axis=0)

print(df_tot)
df_tot.to_csv('datos/df_raw.csv', index=False)

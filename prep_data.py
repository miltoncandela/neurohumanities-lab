import numpy as np
import pickle as pickle
import os
from brainflow.data_filter import DataFilter, WindowOperations
import pyeeg as pe

# Configuración para la función np.load para cargar archivos.
# Los valores por defecto establecidos incluyen el uso de codificación ASCII.
np.load.__defaults__=(None, True, True, 'ASCII')

folder = 'data/pross/10s_norm'
# Definición de los canales EEG que se desean utilizar (8 canales de openbci).
channel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

# Frecuencias de las bandas para calcular el espectro de potencia (PSD) de la señal EEG.
band = [.5,4,8,12,30,45]  # Definición de 5 bandas.

# Tamaño de ventana para promediar la potencia en bandas durante 5 segundos (downsampleado a 128 Hz).
window_size = 128 * 10  # Ventana de 5 segundos a una frecuencia de muestreo de 128 Hz.

# Frecuencia de muestreo de la señal EEG.
fs = 128  # Frecuencia de muestreo de 128 Hz.

# Lista de sujetos para cargar los datos.
subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']


def brainflow_bandpowers(signal, fs, bands):
    # Compute PSD using Welch
    psd = DataFilter.get_psd_welch(signal.astype(np.float64), nfft=fs, overlap=fs//2,
                                   sampling_rate=fs, window=WindowOperations.BLACKMAN_HARRIS)
    # print(psd)
    # Compute band power for each adjacent pair in bands
    band_powers = []
    for i in range(len(bands)-1):
        low, high = bands[i], bands[i+1]
        power = DataFilter.get_band_power(psd, low, high)
        band_powers.append(power)
    return band_powers


def add_index(psd):
    #          0      1      2     3      4
    # PSD = [delta, theta, alpha, beta, gamma]
    psd.append(psd[3]/(psd[2]+psd[1]))  # Engagement (beta/(alpha+theta))
    psd.append(psd[2]/psd[1])           # Fatigue (alpha/theta)
    psd.append(psd[3]/psd[2])           # Excitement (beta/alpha)
    psd.append(psd[1]/psd[0])           # Relaxation (theta/delta)
    return psd


# Definición de la función para procesar datos EEG
def FFT_Processing(sub, channel, band, window_size, fs):
    # Lista para almacenar metadatos
    meta = []

    # Abrir los datos usando pickle para descomprimir
    with open('data/raw/s' + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')  # Resuelve el problema de datos de Python 2 utilizando la codificación latin1

        for i in range(0, 40):
            # Para las 40 pruebas realizadas
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = fs*3-1

            # Baseline-normalized band power
            # cal_psd_min, cal_psd_max = np.zeros((len(channel), len(band)-1 + 4)), np.zeros((len(channel), len(band)-1 + 4))
            # cal_psd_mea, cal_psd_std = np.zeros((len(channel), len(band) - 1 + 4)), np.zeros((len(channel), len(band) - 1 + 4))
            # for j in range(len(channel)):
            #     a = np.zeros((3, len(band)-1 + 4))
            #     for w in range(3):
            #         a[w] = add_index(psd=brainflow_bandpowers(data[j][(fs*w):(fs*(w+1))], fs, band))
            #     print(np.mean(a, axis=0))
            #     print(add_index(psd=brainflow_bandpowers(data[j][:(fs*3)], fs, band)))
            #     cal_psd_min[j], cal_psd_max[j] = np.min(a, axis=0), np.max(a, axis=0)
            #     cal_psd_mea[j], cal_psd_std[j] = np.mean(a, axis=0), np.std(a, axis=0)

            while start + window_size < data.shape[1]:
                meta_array, meta_data = [], []
                for j in range(len(channel)):
                    X = data[j][start:(start+window_size)]  # Dividir los datos crudos en ventanas de 2 segundos, con un intervalo de 0.125 segundos
                    # Y = pe.bin_power(X, band, fs)  # FFT en 5 segundos del canal j, en secuencia de theta, alpha, beta baja, beta alta, gamma
                    # meta_data = meta_data + list(Y[0])
                    # PyEEG: 524.5068,  3251.6598,  3527.7513, 10472.2545, 2828.3445
                    # Brainflow: 0.1634, 1.1050, 1.3249, 2.0719, 0.1612
                    Y = add_index(psd=brainflow_bandpowers(X, fs, band))

                    # Normalization according to: (x - mean(cal))/mean(cal)
                    # for z in range(len(band)-1 + 4):
                    #     Y[z] = (Y[z] - cal_psd[j, z])/cal_psd[j, z]
                    #     Y[z] = (Y[z] - cal_psd_min[j, z]) / (cal_psd_max[j, z] - cal_psd_min[j, z])
                    #     Y[z] = (Y[z] - cal_psd_mea[j, z]) / cal_psd_std[j, z]

                    meta_data += Y

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start += window_size
        meta = np.array(meta)
        np.save(folder + '/s' + sub, meta, allow_pickle=True, fix_imports=True)

for subjects in subjectList:
    FFT_Processing (subjects, channel, band, window_size, fs)

# Listas para almacenar datos y etiquetas de los diferentes conjuntos
data_training, label_training = [], []
data_testing, label_testing = [], []
data_validation, label_validation = [], []

# Iteramos sobre la lista de sujetos
for subject in subjectList:
    # Cargamos los datos procesados para el sujeto actual
    with open(os.path.join(folder, f's{subject}.npy'), 'rb') as file:
        sub = np.load(file)
        # Iteramos sobre los datos procesados del sujeto
        if int(subject) <= 25:
            data_training.append(sub[:, 0])
            label_training.append(sub[:, 1])
        elif int(subject) > 25:
            data_testing.append(sub[:, 0])
            label_testing.append(sub[:, 1])

        '''
        for i in range(sub.shape[0]):
            if i % 8 == 0:
                data_testing.append(sub[i][0])
                label_testing.append(sub[i][1])
            elif i % 8 == 1:
                data_validation.append(sub[i][0])
                label_validation.append(sub[i][1])
            else:
                data_training.append(sub[i][0])
                label_training.append(sub[i][1])
        '''

# Guardamos los conjuntos de datos y etiquetas en archivos
np.save(os.path.join(folder, 'data_training.npy'), np.array(data_training), allow_pickle=True, fix_imports=True)
np.save(os.path.join(folder, 'label_training.npy'), np.array(label_training), allow_pickle=True, fix_imports=True)
np.save(os.path.join(folder, 'data_testing.npy'), np.array(data_testing), allow_pickle=True, fix_imports=True)
np.save(os.path.join(folder, 'label_testing.npy'), np.array(label_testing), allow_pickle=True, fix_imports=True)
# np.save(os.path.join('other/pross/5s_norm', 'data_validation.npy'), np.array(data_validation), allow_pickle=True, fix_imports=True)
# np.save(os.path.join('other/pross/5s_norm', 'label_validation.npy'), np.array(label_validation), allow_pickle=True, fix_imports=True)

# Imprimimos las dimensiones de los conjuntos de datos y etiquetas para cada conjunto
print("Training dataset:", np.array(data_training).shape, np.array(label_training).shape)
print("Testing dataset:", np.array(data_testing).shape, np.array(label_testing).shape)
print("Validation dataset:", np.array(data_validation).shape, np.array(label_validation).shape)

# Restauramos los valores por defecto para la función np.load
np.load.__defaults__=(None, False, True, 'ASCII')
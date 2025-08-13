# Imports for P300
import multiprocessing
import time
from multiprocessing import Process, Value
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
import scipy.stats as stats

# Imports for OpenBCI
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations # WindowFunctions,
import csv

# Imports from Empatica
import socket
import time
import pylsl
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from time import sleep
spectral_signals = {'Alpha': [8, 12], 'Beta': [12, 30],
                    'Gamma': [30, 50], 'Theta': [4, 8], 'Delta': [1, 4]}

# The following object will save parameters to connect with the EEG.
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()

# MAC Adress is the only required parameters for ENOPHONEs
params.mac_address = '00:55:da:bb:0e:72'
# params.serial_port = 'COM12'
# params.serial_number = '5007-BPLE-1E5D'

# devices = BoardShim.get_

# Relevant board IDs available:
# board_id = BoardIds.ENOPHONE_BOARD.value  # (37)
# board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
# board_id = BoardIds.CYTON_BOARD.value # (0)
board_id = BoardIds.MUSE_S_BOARD  # (37)

# Relevant variables are obtained from the current EEG.
eeg_channels = BoardShim.get_eeg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
eta = 4  # Order number of filter
ft = 0  # Filter type (0: Butter, 1: Chev, 2: Bessel)
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

board = BoardShim(board_id, params)

############# Session is then initialized #######################
board.prepare_session()
board.start_stream () # use this for default options
# board.start_stream(45000, "testOpenBCI.csv")
BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Cyton ---')

sleep(5)
data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.
print(data)
eeg_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

############## Data collection #################

# A list of combinations between channels and spectral signals is created.
columns_signals = [s + '_' + 'C' + str(c) for c in range(1, len(eeg_channels) + 1) for s in
                   spectral_signals.keys()]
l_signals = []

# Empty DataFrames are created for raw and PSD data.
df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
df_psd = pd.DataFrame(columns=['PSD' + str(channel) for channel in range(1, len(eeg_channels) + 1)])

# The total number of EEG channels is looped to obtain MV and PSD for each channel, and
# thus saved it on the corresponding columns of the respective DataFrame.
reference = np.sum(np.vstack((data[0], data[1])), axis=0) / 2

for eeg_channel in eeg_channels:
    data_channel = data[eeg_channel] - reference

    df_crudas['MV' + str(eeg_channel)] = data_channel

    # 60 Hz Notch filter
    # DataFilter.perform_bandstop(data=data_channel, sampling_rate=sampling_rate, center_freq=60, band_width=2, order=eta, filter_type=ft, ripple=0)
    DataFilter.perform_bandstop(data=data_channel, sampling_rate=sampling_rate, start_freq=58, stop_freq=62, order=eta,
                                filter_type=ft, ripple=0)

    # 0.1-100 Hz 4th order Butterworth bandpass filter
    DataFilter.perform_lowpass(data=data_channel, sampling_rate=sampling_rate, cutoff=50,
                               order=eta, filter_type=ft, ripple=0)
    DataFilter.perform_highpass(data=data_channel, sampling_rate=sampling_rate, cutoff=1,
                                order=eta, filter_type=ft, ripple=0)

    # Linear de-trend
    DataFilter.detrend(data_channel, DetrendOperations.LINEAR.value)

    psd_data = DataFilter.get_psd_welch(data_channel, nfft, nfft // 2, sampling_rate, 3)
    df_psd['PSD' + str(eeg_channel)] = psd_data[0]

    # Afterwards, a for loop goes over all the spectral signals and saves the unique
    # values into a list, posible because only 1 value is generated per second.
    for spectral_signal in spectral_signals.keys():
        l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                   spectral_signals[spectral_signal][1]))

# The DataFrame that involves the spectral signals is created, using the previously created
# list of combinations and our appended values when looping over each channel and signal.
data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[eeg_time],
                            columns=columns_signals)

print(data_signals)

exit()

# # CODE FOR EEG # #
def EEG(second, folder):
    # PSD 128 rows
    # Band power 1 row each

    ############### Alpha/Beta plot created ###############
    # Create the grid plot considering 4 channels.
    fig3 = plt.figure(3, constrained_layout=True, figsize=(10, 6))
    gs3 = fig3.add_gridspec(2, 2)  # (Rows, Columns)
    ax17_20 = gs3.subplots(sharex=True, sharey=False)

    # The ranges of spectral signals are declared, though can be modified.
    spectral_signals = {'Alpha': [8, 12], 'Beta': [12, 30],
                        'Gamma': [30, 50], 'Theta': [4, 8], 'Delta': [1, 4]}

    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    # params.mac_address = 'f4:0e:11:75:75:ce'
    params.serial_port = 'COM12'

    # Relevant board IDs available:
    # board_id = BoardIds.ENOPHONE_BOARD.value  # (37)
    # board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)
    board_id = BoardIds.MUSE_S_BLED_BOARD  # (37)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eta = 4  # Order number of filter
    ft = 0  # Filter type (0: Butter, 1: Chev, 2: Bessel)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    board = BoardShim(board_id, params)

    # An empty dataframe is created to save Alpha/Beta values to plot in real time.
    alpha_beta_data = pd.DataFrame(columns=['Alpha_C' + str(c) for c in range(1, len(eeg_channels) + 1)])
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Cyton ---')

    try:
        while (True):
            time.sleep(2)
            with second.get_lock():
                # When the seconds reach 332, we exit the functions.
                if (second.value == tiempo_prueba):
                    plt.close()
                    return
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.
            eeg_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

            ############## Data collection #################

            # A list of combinations between channels and spectral signals is created.
            columns_signals = [s + '_' + 'C' + str(c) for c in range(1, len(eeg_channels) + 1) for s in
                               spectral_signals.keys()]
            l_signals = []

            # Empty DataFrames are created for raw and PSD data.
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            df_psd = pd.DataFrame(columns=['PSD' + str(channel) for channel in range(1, len(eeg_channels) + 1)])

            # The total number of EEG channels is looped to obtain MV and PSD for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            reference = np.sum(np.vstack((data[0], data[1])), axis=0) / 2

            for eeg_channel in eeg_channels:
                data_channel = data[eeg_channel] - reference

                df_crudas['MV' + str(eeg_channel)] = data_channel

                # 60 Hz Notch filter
                DataFilter.perform_bandstop(data=data_channel, sampling_rate=sampling_rate, center_freq=60,
                                            band_width=2, order=eta, filter_type=ft, ripple=0)

                # 0.1-100 Hz 4th order Butterworth bandpass filter
                DataFilter.perform_lowpass(data=data_channel, sampling_rate=sampling_rate, cutoff=50,
                                           order=eta, filter_type=ft, ripple=0)
                DataFilter.perform_highpass(data=data_channel, sampling_rate=sampling_rate, cutoff=1,
                                            order=eta, filter_type=ft, ripple=0)

                # Linear de-trend
                DataFilter.detrend(data_channel, DetrendOperations.LINEAR.value)

                psd_data = DataFilter.get_psd_welch(data_channel, nfft, nfft // 2, sampling_rate, 3)
                df_psd['PSD' + str(eeg_channel)] = psd_data[0]

                # Afterwards, a for loop goes over all the spectral signals and saves the unique
                # values into a list, posible because only 1 value is generated per second.
                for spectral_signal in spectral_signals.keys():
                    l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                               spectral_signals[spectral_signal][1]))

            # The DataFrame that involves the spectral signals is created, using the previously created
            # list of combinations and our appended values when looping over each channel and signal.
            data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[eeg_time],
                                        columns=columns_signals)

            def get_columns_signal(spect_signal):
                """
                This function returns a list of combinations between channels and a given spectral signal.

                :param string spect_signal: Name of the spectral signal (Alpha, Beta, ...)
                :return list: List of combinations between channels and the spectral signal.
                """
                return [spect_signal + '_' + 'C' + str(c) for c in range(1, len(eeg_channels) + 1)]

            # Combined signals (Alpha / Beta) are created using all alphas and all betas from all given EEG channels.
            for eeg_channel in eeg_channels:
                data_signals['Alpha_Beta_C' + str(eeg_channel)] = data_signals['Alpha_C' + str(eeg_channel)] / \
                                                                  data_signals['Beta_C' + str(eeg_channel)]

            # For each spectral signal, a CSV is saved according to its name, looping over the DataFrame created.
            for spectral_signal in list(spectral_signals.keys()) + ['Alpha_Beta']:
                data_signals.loc[:, get_columns_signal(spectral_signal)].to_csv(
                    '{}/Raw/{}.csv'.format(folder, spectral_signal), mode='a')

            # Both the raw and PSD DataFrame is exported as a CSV.
            df_crudas.to_csv('{}/Raw/Crudas.csv'.format(folder), mode='a')
            df_psd.to_csv('{}/Raw/PSD.csv'.format(folder), mode='a')
            df_psd.to_json('{}/Raw/PSD.json'.format(folder))

            ###########################################################################################################

            # # # # Alpha / Beta Plot usage # # # 

            # The empty Alpha/Beta DataFrame is appended with the current value, this would be
            # the only value that could be saved for each iteration, all other are overwritten.

            alpha_beta_data = alpha_beta_data.append(data_signals.loc[:, get_columns_signal('Alpha')],
                                                     ignore_index=True)

            def plot_signal(ax, canal, n):
                """
                This function takes an axes object and plots the Alpha/Beta array according to the channel n.
                :param matplotlib.axes._subplots.AxesSubplot ax: Axes where the band power will be plotted. 
                :param pd.Series canal: The current column that is being plotted.
                :param integer n: Number of the current channel.
                """

                # The previous plot is cleared, new data is added and the title is re-established.
                ax.clear()
                ax.plot(canal)
                ax.set_title('Alpha / Beta Channel ' + str(n))
                plt.pause(0.001)

            # The following nested for loops go over the matrix of plots, and uses the previously declared
            # function to plot all the Alpha/Beta data for each channel on their respective plot.

            for i in range(ax17_20.shape[0]):
                for j in range(ax17_20.shape[1]):
                    if (i + j) != 4:
                        plot_signal(ax17_20[i][j], alpha_beta_data.iloc[:, i * 2 + j], i * 2 + j + 1)

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Cyton ---')

    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/


def MachineLearning(second, folder):
    # Pipeline for normalization:
    # 1. Remove outliers based on quantile method (1 < second < 91)
    # 2. Combine features
    # 3. Normalize using minmax scaler (61 < second < 91)

    # Se crea un diagrama donde se registra el nivel de fatiga mental en tiempo real
    fatigue_fig = plt.figure(4, constrained_layout=True)
    ax_fatigue = fatigue_fig.add_subplot(111)
    ax_fatigue.plot([-1], [-1])
    ax_fatigue.set_ylim([0, 100])
    ax_fatigue.set_xlim([0, 1])
    ax_fatigue.set_xlabel('Samples')
    ax_fatigue.set_ylabel('Engagement score')
    plt.pause(1)

    # Variables que modifican el comportamiento del script, así como inicializadores para su correcto funcionamiento
    scaler = None
    free_flag = True
    a_prediction = False
    current_second = 0
    l_fatigue = []

    # Se extraen las features y se dividen entre features que requieran de la EEG
    # En la lista "eeg_features" se almacenan las features de eeg "Delta_C4" por ejemplo
    eeg_features = ['Theta', 'Alpha', 'Beta']

    def get_df(seconds):
        """
        Esta función obtiene la df de acuerdo a las features que se requieran en "all_features", para entonces hacer
        un subset en sus filas de acuerdo a lista "seconds"

        :param list seconds: Segundo inicial y final para el que se hará el subset del df [t_ini, t_fin]
        :return pd.DataFrame: df con subset de acuerdo a la lista de segundos
        """
        df_calibration = pd.DataFrame()

        # channel_to_position = {'A2': ['Right Cushion'], 'A1': ['Left Cushion'], 'C4': ['Top Right'], 'C3': ['Top Left']}
        channel_to_true = {'C1': 'A2', 'C2': 'A1', 'C3': 'C4', 'C4': 'C3'}

        # El siguiente ciclo for itera por cada DataFrame que tiene que leer, de acuerdo a las features que se necesitan
        # para que el modelo de ML prediga. Lo que hace es básicamente reducir la característica "Alpha_C4" a "Alpha",
        # para así obtener el .csv de "Alpha". Aunado a esto, se hace un list(set()) para elimninar los valores
        # duplicados que podrían ocurrir en caso de que la lista tenga "Alpha_C4" y "Alpha_C3".
        # Una manera de diferenciar entre características de empática y EEG, es que las de empática tienen mayúsculas,
        # por lo que, con el método .isupper() podemos saber si se está procesando una característica eeg o emp
        for df_name in eeg_features:
            # EEG Features
            # Se lee la característica de EEG de acuerdo con el folder y la banda actual
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)

            # Además que se procesa al momento de pasar los datos a numérico y eliminar los valores NA, esto se
            # hace de esta forma debido a la estructura del archivo, a la cual no se le quitaron los headers, y
            # al momento de todo pasarlo a numérico, los strings de headers se hacen errores NA y son eliminados
            # con el dropna para las rows con el axis=0
            df_processed_temp = df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0)
            # .reset_index(drop=True)

            # Esta columna de datos procesada "Alpha" se agrega a una DataFrame vacía llamada "df_calibration",
            # se agrega una columna debido a que se específico axis=1
            df_calibration = pd.concat([df_calibration, df_processed_temp], axis=1)

        # Al index de la df se le llama segundo y posteriormente este es asignado como columna
        eeg_idx = df_calibration.index.rename('Second')
        df_calibration.index = eeg_idx
        df_calibration = df_calibration.reset_index()

        columns = df_calibration.columns
        df_calibration.columns = ['Second'] + [
            columns[j].split('_')[0] + '_' + channel_to_true[columns[j].split('_')[1]] for j in
            range(1, df_calibration.shape[1])]
        for feature in eeg_features:
            df_calibration[feature] = (df_calibration[feature + '_' + 'C3'] + df_calibration[feature + '_' + 'C4']) / 2
        df_calibration = df_calibration[eeg_features + ['Second']]

        # Ahora para esta función, se hará un subset de la df con base en la lista de segundos, esto es bastante
        # importante debido a que se utilizará esta misma función para obtener la df, pero la lista de segundos
        # es clave para obtener datos de calibración, escalado o brutos ya para medición
        df_calibration['Second'] = pd.to_datetime(df_calibration['Second'])
        df_calibration['Second'] = (df_calibration['Second'] - df_calibration['Second'].iloc[
            0]).dt.total_seconds().astype(int) + 2
        df_calibration = df_calibration[(df_calibration.Second > seconds[0]) & (df_calibration.Second < seconds[1])]
        return df_calibration

    while True:
        with second.get_lock():
            # Condicional para terminar la función, de acuerdo al tiempo de duración deseado (332 s en este caso)
            if (second.value == tiempo_prueba):
                return

            # En caso de que el tiempo de calibración haya pasado (90 s) y no se tenga un escalador, además que la
            # función esté libre (free_flag), entonces se procede a asignar los valores al escalador
            if (second.value > 31) and (second.value < 35) and (scaler is None) and (free_flag):
                # Lockea la función para que no se puedan ejecutar otras acciones
                free_flag = False

                # Para el tiempo de calibración, se determinó del segundo 30 al 90, entonces se obtiene la df de acuerdo
                # a ese tiempo, para entonces obtener los segundos en forma de lista y eliminarlo de la df, esto se
                # realiza debido a que se proceden a hacer funciones matemáticas, y se necesita que únicamente
                # se tengan valores numericos relevantes de las características, en contraste con el índice de s
                # df_calibration = get_df([31, 91])
                df_calibration = get_df([1, 31])
                print(df_calibration)
                print('***************************************************')
                seconds_col = df_calibration.Second
                df_calibration = df_calibration.drop('Second', axis=1)
                df_calibration = df_calibration.loc[:, ]

                # La columna de segundo regresa para hacer un subset ahora con los segundos de la fase de calibración
                # ojos abiertos 1-11 s
                df_calibration['Second'] = seconds_col
                df_calibration = df_calibration[(df_calibration.Second > 1) & (df_calibration.Second < 31)].drop(
                    'Second', axis=1)

                # Finalmente, se asignan los valores en "scaler" que en este caso se encuentra como función de
                # MinMaxScaler() para aplicar .transform() en una df nueva y que se transformen los valores.
                # De la misma manera, también se pudieron obtener los valores deseados de forma manual a través de
                # operaciones matemáticas a la df para obtener valor máximo, media, desviación estándar y hacer
                # el escalado posteriormente con lso valores en ese df
                scaler = StandardScaler().fit(df_calibration)
                free_flag = True

            # Cuando el segundo sea mayor que 90 (mediciones en bruto y en la prueba), además que se tenga un escalador
            # y el segundo sea un múltiplo de 10 (predicciones cada 10 segundos), además que el segundo actual no es el
            # mismo que el segundo global del multiprocesamiento (esto para evitar que se ejecuten muchas acciones
            # en la misma función sin que se haya terminado la anterior); entonces se podrá hacer una predicción
            if (second.value > 31) and (second.value % 5 == 0) and (scaler is not None) and (
                    current_second != second.value):

                # El valor del segundo en multiprocesamiento se asigna al segundo actual para evitar predicciones
                # dobles en el mismo segundo
                current_second = second.value

                print(current_second - 10, current_second)

                # Se hace un subset de la df con base en los 10 segundos anteriores y el valor del segundo actual
                df_current = get_df([current_second - 10, current_second]).drop('Second', axis=1)
                print('Anterior')
                print(df_current)

                # Se remueven outliers con base en un método estadístico basado en quantiles y "calib_Q"
                # df_current = df_current[~((df_current < (calib_Q[0] - 4.5 * calib_Q[2])) | (df_current > (calib_Q[1] + 4.5 * calib_Q[2]))).any(axis=1)]
                df_current = pd.DataFrame(scaler.transform(df_current), columns=eeg_features)
                df_current = df_current[np.abs(df_current) < 3].dropna(axis=0, how='any')

                print('Filtrada')
                print(df_current)

                # Si existen muestras válidas luego de aplicar el método que remueve outliers, entonces se utiliza
                if df_current.shape[0] > 0:
                    sigma = 80
                    df_current['Pred'] = sigma * (-3 * df_current.Theta + df_current.Alpha - df_current.Beta)
                    # df_current['Pred'] = [x for x in df_current]

                    # Posteriormente, se toma la mediana del valor predicho para graficarlo en caso de que el promedio
                    # del FAS sea mayor a 50 (valor máximo), en caso de que no, entonces se hace un promedio
                    prediction = np.median(np.round(df_current.Pred)) if abs(np.mean(df_current.Pred)) > 100 else round(
                        np.mean(df_current.Pred))

                    # Ya que se tiene una predicción, entonces se enciende la bandera "a_prediction", para indicar
                    # que se pudo obtener una predicción y debe de ser graficada
                    a_prediction = True

                    print(second.value, df_current.Pred, prediction)

                    # Por cada elemento en una predicción cruda, se procede a mapear dicho en valores posibles, es
                    # decir, valores que hacen sentido para la encuesta FAS, cuyo dominio se encuentra: 0 < x < 50.
                    # Adicionalmente, estos valores de predicciones procesadas se guardarán en una lista "l_fatigue"
                    # para poder graficarlos posteriormente
                    # for element in raw_prediction:
                    for element in [prediction]:
                        if element < 0:
                            element = 0
                        elif element > 100:
                            element = 100
                        l_fatigue.append(element)

                    from statistics import mean
                    analog = -15 * mean(prediction) + 1024
                    print(analog)
                    asyncio.run(send_data(analog))
                else:
                    # En caso de que no se tengan observaciones válidas después de remover outliers, la predicción
                    # se le asignará un valor de NAN, indicando que no hay un valor numérico
                    prediction = np.nan

                # Sin importar el valor de predicción FAS que se tuvo (o si quiera si hubo uno), este valor es guardado
                # en "df_prediction" y después exportado a un CSV "predictions.csv", aunque para el caso de tener NAN,
                # se asignará un valor vacio al index con base en el segundo relativo
                df_predictions = pd.Series([prediction], index=[second.value], name='FAS').head(1)
                df_predictions.to_csv('{}/predictions.csv'.format(folder), mode='a')

            # Ahora bien, si se tiene una predicción, el segundo es mayor que 100 (posible debido a que se empiezan
            # a tener predicciones en s > 90 cada 10 s, entonces el primero debería de ser después de 100, además
            # el segundo esté en un múltiplo de 5 o de 10 (tasa de refresco para el cual se asignarán nuevos valores
            # al plot, este funciona mejor con múltiplo de 5 debido a que no se empalma el procesamiento con el
            # statement de predicción, y no detiene el multiprocesamiento de todas las otras funciones, por lo que se
            # realiza la operación de manera distribuida en segundo 5 para graficar y segundo 10 para predecir
            # Adicionalmente, se asegura que el segundo actual es diferente a segundo de multiprocesamiento y que
            # haya un valor en "scaler"
            if (a_prediction) and (second.value > 100) and (second.value % 5 == 0) and (second.value % 10 != 0) and (
                    current_second != second.value) and (scaler is not None):
                current_second = second.value

                # Se crea una serie de predicciones procesadas y se le asigna un índice de acuerdo a su longitud
                s_predictions = pd.Series(data=l_fatigue)
                s_predictions.index = [x + 1 for x in list(s_predictions.index)]

                print(s_predictions)

                # Con base en el plot creado al inicio de la función, se asignan los valores a dicho plot a través de
                # su eje "ax_fatigue", es importante no crear un plot nuevo debido a que no funciona para tiempo real;
                # la manera más cómoda es al eliminar los valores del plot con .clear(), seguido de asignar los valores
                # usando .plot(), así como el nombre de sus ejes y límites para cada uno. A pesar de que únicamente
                # se grafique "s_predictions", esta serie tiene los valores de toda la prueba, puesto que se está
                # utilizando "l_fatigue" como lista global donde se hacen append a todos los valores predichos, crudos
                # pero con el procesamiento de restricción de dominio para el valor FAS
                ax_fatigue.clear()
                ax_fatigue.plot(s_predictions)
                ax_fatigue.set_ylim([0, 100])
                ax_fatigue.set_xlim([0, s_predictions.shape[0] + 1])
                ax_fatigue.set_xlabel('Samples')
                ax_fatigue.set_ylabel('Engagement score')
                plt.pause(1)

                # Finalmente, se apaga la bandera "a_prediction", determinando entonces que se requiere una predicción
                # con valores válidos después de remover outliers para entonces plotear dichos en tiempo real
                a_prediction = False


# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
if __name__ == '__main__':

    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder = 'S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)

    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()

    # # Start processes # #
    process2 = Process(target=timer, args=[seconds, counts, timestamps])
    q = Process(target=EEG, args=[seconds, folder])
    m = Process(target=MachineLearning, args=[seconds, folder])
    process2.start()
    q.start()
    m.start()
    process2.join()
    q.join()
    m.join()

    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    print(Fore.RED + 'Test finished sucessfully, storing data now...' + Style.RESET_ALL)
    # # Save beeps' timestamps in a .csv file # #
    # We must first convert the multiprocess.Manger.List to a normal list
    timestamps_final = list(timestamps)

    # Now we convert each of the UNIX-type timestamps to normal timestam (year-month-day hour-minute-second-ms)
    for i in range(len(timestamps_final)):
        timestamp = datetime.fromtimestamp(timestamps_final[i])
        timestamps_final[i] = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Store data in a .csv
    timestamps_final = pd.Series(timestamps_final)
    df = pd.DataFrame(timestamps_final, columns=['Beep_Timestamp'])
    df.to_csv('{}/Timestamps.csv'.format(folder), index=False)
    print(Fore.GREEN + 'Data stored sucessfully' + Style.RESET_ALL)

    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)


    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the dataset x, and filters the valid rows back to y.

        :param pd.DataFrame df: with non-normalized, source variables.
        :param string method: type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df.quantile(q=.25)
            q3 = df.quantile(q=.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]

        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')
        # print('{} rows removed {}%'.format(diff, round(diff / n_pre * 100, 2)))
        # print(str(diff) + 'rows removed' + str(round(diff / n_pre * 100, 2)) + '%')
        return df


    # The following for loop iterates over all features, and removes outliers depending on the statistical method used.
    # It reads the files saved in the "Raw" folder, and only reads .CSV files, to outputt a .CSV file in "Processed" folder.
    for df_name in os.listdir('{}/Raw/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(
                df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')

            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)

####### Sources ########
# https://www.empatica.com/connect/session_view.php?id=1699109

# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
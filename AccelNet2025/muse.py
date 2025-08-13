import os
import time
import numpy as np
import pandas as pd
from statistics import mean
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from pythonosc import udp_client
from colorama import Fore, Style
from multiprocessing import Process, Value, Manager
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations # WindowFunctions,

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel
# processes
seconds = Value("i", 0)
counts = Value("i", 0)
tiempo_prueba = 5000  # segundos

ip = '192.168.0.108'
client = udp_client.SimpleUDPClient(ip, 7000) # Engagement
# client2 = udp_client.SimpleUDPClient(ip, 9000) # Excitement

# This function is the countup timer, the count is set to 0 before the script
# waits for a second, otherwise the beep will sound several times before the second 
# changes.
def timer(second, count, timestamps):
    # First we initialize a variable that will contain the moment the timer began and 
    # we store this in the timestamps list that will be stored in a CSV.    
    time_start = time.time()
    timestamps.append(time_start)
    while True:
        # The .get_lock() function is necessary since it ensures they are 
        # sincronized between both functions, since they both access to the same 
        # variables
        with second.get_lock(), count.get_lock():
            # We now calculate the time elappsed between start and now. 
            # (should be approx. 1 second)
            second.value = int(time.time() - time_start)
            count.value = 0
            if (second.value == tiempo_prueba):
                return
            print(second.value, end="\r")
        # Once we stored all the info and make the calculations, we sleep the script for
        # one second. This is the magic of the script, it executes every  ~1 second.
        time.sleep(1)  # 0.996


# # CODE FOR EEG # #
def EEG(second, folder):
    # PSD 128 rows
    # Band power 1 row each

    # The ranges of spectral signals are declared, though can be modified.
    spectral_signals = {'Alpha': [8, 12], 'Beta': [12, 30],
                        'Gamma': [30, 50], 'Theta': [4, 8], 'Delta': [1, 4]}

    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    # params.mac_address = 'f4:0e:11:75:75:ce'
    params.mac_address = '00:55:da:bb:0e:72'
    # params.serial_port = 'COM12'

    # Relevant board IDs available:
    # board_id = BoardIds.ENOPHONE_BOARD.value  # (37)
    # board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)
    board_id = BoardIds.MUSE_S_BOARD # (37)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eta = 4  # Order number of filter
    ft = 0  # Filter type (0: Butter, 1: Chev, 2: Bessel)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    board = BoardShim(board_id, params)
    
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Cyton ---')
    # time.sleep(5)

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
            print(second.value, eeg_time)

            ############## Data collection #################

            # A list of combinations between channels and spectral signals is created.
            columns_signals = [s + '_' + c for c in channels for s in spectral_signals.keys()]
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
                DataFilter.perform_bandstop(data=data_channel, sampling_rate=sampling_rate, start_freq=58,
                                            stop_freq=62, order=eta, filter_type=ft, ripple=0)

                # 0.1-100 Hz 4th order Butterworth bandpass filter
                DataFilter.perform_lowpass(data=data_channel, sampling_rate=sampling_rate, cutoff=50,
                                           order=eta, filter_type=ft, ripple=0)
                DataFilter.perform_highpass(data=data_channel, sampling_rate=sampling_rate, cutoff=1,
                                            order=eta, filter_type=ft, ripple=0)

                # Linear de-trend
                # DataFilter.detrend(data_channel, DetrendOperations.LINEAR.value)

                psd_data = DataFilter.get_psd_welch(data_channel, nfft, nfft // 2, sampling_rate, 3)
                df_psd['PSD' + str(eeg_channel)] = psd_data[0]

                # Afterwards, a for loop goes over all the spectral signals and saves the unique
                # values into a list, posible because only 1 value is generated per second.
                for spectral_signal in spectral_signals.keys():
                    l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                               spectral_signals[spectral_signal][1]))

            # The DataFrame that involves the spectral signals is created, using the previously created
            # list of combinations and our appended values when looping over each channel and signal.
            data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[second.value], # index=[eeg_time],
                                        columns=columns_signals)
            print(data_signals)

            def get_columns_signal(spect_signal):
                """
                This function returns a list of combinations between channels and a given spectral signal.

                :param string spect_signal: Name of the spectral signal (Alpha, Beta, ...)
                :return list: List of combinations between channels and the spectral signal.
                """
                return [spect_signal + '_' + c for c in channels]

            # Combined signals (Alpha / Beta) are created using all alphas and all betas from all given EEG channels.
            for eeg_channel in channels:
                data_signals['Alpha_Beta_' + str(eeg_channel)] = data_signals['Alpha_' + str(eeg_channel)] / \
                                                                  data_signals['Beta_' + str(eeg_channel)]

            # For each spectral signal, a CSV is saved according to its name, looping over the DataFrame created.
            for spectral_signal in list(spectral_signals.keys()) + ['Alpha_Beta']:
                data_signals.loc[:, get_columns_signal(spectral_signal)].to_csv(
                    '{}/Raw/{}.csv'.format(folder, spectral_signal), mode='a')

            # Both the raw and PSD DataFrame is exported as a CSV.
            df_crudas.to_csv('{}/Raw/Crudas.csv'.format(folder), mode='a')
            df_psd.to_csv('{}/Raw/PSD.csv'.format(folder), mode='a')
            df_psd.to_json('{}/Raw/PSD.json'.format(folder))

            ###########################################################################################################

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
        # channel_to_true = {'C1': 'A2', 'C2': 'A1', 'C3': 'C4', 'C4': 'C3'}
        channels = ['TP9', 'AF7', 'AF8', 'TP10']

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
            columns[j].split('_')[0] + '_' + columns[j].split('_')[1] for j in range(1, df_calibration.shape[1])]
        for feature in eeg_features:
            df_calibration[feature] = (df_calibration[feature + '_' + channels[0]] + df_calibration[feature + '_' + channels[1]] + df_calibration[feature + '_' + channels[2]] + df_calibration[feature + '_' + channels[3]]) / 4
        df_calibration = df_calibration[eeg_features + ['Second']]
        df_calibration['Engagement'] = df_calibration['Beta'] / (df_calibration['Theta'] + df_calibration['Alpha'])
        df_calibration['Excitement'] = df_calibration['Beta'] / df_calibration['Theta']

        # Ahora para esta función, se hará un subset de la df con base en la lista de segundos, esto es bastante
        # importante debido a que se utilizará esta misma función para obtener la df, pero la lista de segundos
        # es clave para obtener datos de calibración, escalado o brutos ya para medición
        # df_calibration['Second'] = pd.to_datetime(df_calibration['Second'])
        # df_calibration['Second'] = (df_calibration['Second'] - df_calibration['Second'].iloc[
        #     0]).dt.total_seconds().astype(int) + 2
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
                scaler = MinMaxScaler().fit(df_calibration)
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
                df_current = get_df([current_second - 12, current_second - 2]).drop('Second', axis=1)
                print('Anterior')
                print(df_current)

                # Se remueven outliers con base en un método estadístico basado en quantiles y "calib_Q"
                # df_current = df_current[~((df_current < (calib_Q[0] - 4.5 * calib_Q[2])) | (df_current > (calib_Q[1] + 4.5 * calib_Q[2]))).any(axis=1)]
                df_current = pd.DataFrame(scaler.transform(df_current), columns=eeg_features + ['Engagement', 'Excitement'])
                # df_current = df_current[np.abs(df_current) < 3].dropna(axis=0, how='any')

                print('Filtrada')
                print(df_current)

                def pross_index(df, curr_index):
                    # Posteriormente, se toma la mediana del valor predicho para graficarlo en caso de que el promedio
                    # del FAS sea mayor a 50 (valor máximo), en caso de que no, entonces se hace un promedio
                    # prediction = np.median(np.round(df[curr_index])) if abs(np.mean(df[curr_index])) > 1 else round(
                    #     np.mean(df[curr_index]))

                    # Ya que se tiene una predicción, entonces se enciende la bandera "a_prediction", para indicar
                    # que se pudo obtener una predicción y debe de ser graficada
                    # a_prediction = True

                    # print(second.value, df_current.Engagement, prediction)

                    # Por cada elemento en una predicción cruda, se procede a mapear dicho en valores posibles, es
                    # decir, valores que hacen sentido para la encuesta FAS, cuyo dominio se encuentra: 0 < x < 50.
                    # Adicionalmente, estos valores de predicciones procesadas se guardarán en una lista "l_fatigue"
                    # para poder graficarlos posteriormente
                    # for element in raw_prediction:
                    # for element in [prediction]:
                    #     if element < 0:
                    #         element = 0
                    #     elif element > 1:
                    #         element = 1
                    #     l_fatigue.append(element)
                    
                    return round(mean(df[curr_index]), 2)


                # Si existen muestras válidas luego de aplicar el método que remueve outliers, entonces se utiliza
                if df_current.shape[0] > 0:
                    print('****************')
                    eng = pross_index(df_current, 'Engagement')
                    if eng < 0:
                        eng = 0
                    elif eng > 1:
                        eng = 1
                    print('ENGAGEMENT:', eng)
                    client.send_message('/aud_eng', eng)

                    exc = pross_index(df_current, 'Excitement')
                    if exc < 0:
                        exc = 0
                    elif exc > 1:
                        exc = 1
                    print('Excitement:', exc)
                    client.send_message('/aud_exc', exc)
                    print('****************')

                    # analog = -15 * mean(prediction) + 1024
                    # print(analog)
                    # asyncio.run(send_data(analog))
                else:
                    # En caso de que no se tengan observaciones válidas después de remover outliers, la predicción
                    # se le asignará un valor de NAN, indicando que no hay un valor numérico
                    prediction = np.nan

                # Sin importar el valor de predicción FAS que se tuvo (o si quiera si hubo uno), este valor es guardado
                # en "df_prediction" y después exportado a un CSV "predictions.csv", aunque para el caso de tener NAN,
                # se asignará un valor vacio al index con base en el segundo relativo
                # df_predictions = pd.Series([prediction], index=[second.value], name='FAS').head(1)
                # df_predictions.to_csv('{}/predictions.csv'.format(folder), mode='a')


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
    timestamps = Manager().list()

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
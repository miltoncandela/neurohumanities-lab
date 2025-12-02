%% Start EEGLAB
addpath('C:\Users\Milton\Documents\eeglab2024.0'); eeglab;

%% Read raw eeg.csv

dataTable = readtable('data/csv_raw/S001R000_Complete.csv');
data = table2array(dataTable(:,3:end-3)); % drop Var1, Timestamp, acc_xyz
data = data(~any(isnan(data),2), :); % remove NaNs at the end of the file

fs = 250;
EEG = pop_importdata('dataformat','array','data','data','srate',fs); 
EEG.nbchan = 8;

EEG = pop_eegfiltnew(EEG, 'locutoff', 1);       % 0.1 Hz high pass
EEG = pop_cleanline(EEG, 'bandwidth', 2, 'chanlist', [1:EEG.nbchan], ...
                    'computepower', 2, 'linefreqs', [60 120]);  % 60 Hz noise filter
EEG = clean_asr(EEG, 20);                       % ASR
EEG = pop_reref(EEG, []);                       % avg ref
    
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);  % ICA   

%%
load('chanlocs64.mat'); EEG.chanlocs = chanlocs;% channel locations

%%


%%

EEG = pop_cleanline(EEG, 'bandwidth', 2, 'chanlist', [1:EEG.nbchan], ...
                    'computepower', 2, 'linefreqs', [60 120]);  % 60 Hz noise filter


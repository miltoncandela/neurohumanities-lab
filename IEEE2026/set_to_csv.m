
%%%% Práctica 2 %%%%
%clearvars; close all; clc

% List all files in the folder
folder = 'data/set';
files = dir(folder);

% Iterate through the files and read their content
for i = 3:numel(files)
    fileName = files(i).name;
    disp(fileName)

    x = pop_loadset([folder, '/', fileName]).data;
    x = x';
    
    writematrix(x, ['data/csv/', fileName(1:end-4), '.csv'])
end

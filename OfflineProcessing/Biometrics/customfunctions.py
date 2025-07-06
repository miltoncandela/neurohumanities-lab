# This script was written by José Emiliano Calderón Gurubel
import numpy as np

# Time window function
def timewindow(data = np.array([]),fs = 1,winsz = 1,ovlap = 0):
    # Parameters and data
    wn = winsz * fs # Samples per window
    on = int(np.floor(ovlap * wn)) # Samples in overlap
    dl = len(data) # Length of data
    bN = int((dl - wn)/(wn - on) + 1) # Buffer number

    # Buffering
    # This loop goes through rows of the output matrix and starting index of data, first range inside zip establishes rows, the second one 
    # states the indexes, going from value index 0 to data length minus window size plus one, with step of window size minus overlap
    outdata = np.zeros([bN,wn]) # Output data
    for row, idx in zip(range(0,bN),range(0,(dl-wn) + 1,wn-on)):  
        outdata[row,:] = data[idx:idx + wn]
    return outdata

# Time window function with zero padding
def timewindowpadded(data=np.array([]), fs=1, winsz=1, ovlap=0):
    # Parameters and data
    wn = winsz * fs  # Samples per window
    on = int(np.floor(ovlap * wn))  # Samples in overlap
    dl = len(data)  # Length of data
    bN = int(np.ceil((dl - wn) / (wn - on) + 1))  # Buffer number with ceil to ensure no data is lost
    
    # Padding calculation
    total_len = bN * (wn - on) + on  # Total length considering the overlap
    padding_len = total_len - dl  # Number of zeros to pad
    if padding_len > 0:
        data = np.pad(data, (0, padding_len), 'constant')  # Zero padding at the end
    
    # Buffering
    outdata = np.zeros([bN, wn])  # Output data
    for row, idx in zip(range(0, bN), range(0, len(data) - wn + 1, wn - on)):
        outdata[row, :] = data[idx:idx + wn]
    
    return outdata

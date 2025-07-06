import customfunctions as ctf
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
import heartpy as hp

## This code was written by José Emiliano Calderón Gurubel. https://github.com/JoseEmilianoCG ##

### BVP preprocessing functions ###
# Based on methodology from:
# Sadhukhan, D., Pal, S., & Mitra, M. (2018). PPG Noise Reduction based on Adaptive Frequency Suppression using Discrete Fourier Transform
# for Portable Home Monitoring Applications. 2018 15th IEEE India Council International Conference (INDICON), 1-6.
# https://doi.org/10.1109/INDICON45594.2018.8987004


def get_low_cut(segment, fs):
    # Low-cut frequency is established from heartrate in the first signal segment.
    # Signal segment is smoothed, as to obtain a dynamic threshold for peak detection using heartpy toolkit
    bvp_smooth = uniform_filter1d(segment, size=int(0.75 * fs), mode="nearest")

    # Heartpy toolkit is used for peak detection, outliers are removed by applying clean_rr_intervals method.
    wd = {}
    wd = hp.peakdetection.detect_peaks(segment, bvp_smooth, ma_perc=20, sample_rate=fs)
    wd = hp.analysis.calc_rr(wd["peaklist"], sample_rate=fs, working_data=wd)
    wd = hp.peakdetection.check_peaks(
        wd["RR_list"], wd["peaklist"], segment[wd["peaklist"]], working_data=wd
    )
    wd = hp.analysis.clean_rr_intervals(working_data=wd, method="quotient-filter")

    # Heartrate is calculated
    ibimean = np.mean(wd["RR_list_cor"])
    hr = 60000 / ibimean

    # Low-cut frequency is calculated
    low_cut = (hr / 60) - 0.2

    return low_cut


def get_high_cut(segment, fs, low_cut):
    # High-cut frequency is located by finding peak frequency in spectrum of the signal segment. High-cut frequency is then selected by finding the
    # last frequency with a magnitude higher than 15% of the peak frequency.

    t = np.arange(0, len(segment)) * (1 / fs)
    fft_result = np.fft.fft(segment)
    freq = np.fft.fftfreq(t.shape[-1], d=1 / fs)

    positive_freqs = freq[freq >= 0]
    positive_fft = np.abs(fft_result[freq >= 0])

    peak_indices, _ = signal.find_peaks(positive_fft)
    peak_indices = peak_indices[positive_freqs[peak_indices] >= low_cut]
    max_peak_index = peak_indices[np.argmax(positive_fft[peak_indices])]

    mask = positive_fft >= positive_fft[max_peak_index] * 0.15

    high_cut = positive_freqs[mask][-1]

    return high_cut


def partial_dft(signal_segment, fs, low_cut, high_cut):
    # DFT is applied to the signal segment (using fft function from numpy), valid frequencies are isolated in a new array, where all out-of-bound
    # frequencies are replaced by 0 values.
    N = len(signal_segment)
    freqs = np.fft.fftfreq(N, d=1 / fs)
    fft_full = np.fft.fft(signal_segment)
    fft_partial = np.zeros_like(fft_full, dtype=complex)

    valid_pos = np.where((freqs >= low_cut) & (freqs <= high_cut))[0]
    fft_partial[valid_pos] = fft_full[valid_pos]

    # Symmetry is peserved for real IDFT
    # If the number of samples is even...
    if N % 2 == 0:
        # The Nyquist frequency (N//2) is excluded, since it is its own conjugate
        valid_neg = -valid_pos[valid_pos != N // 2]
    else:
        # If odd, all positive frequencies have a corresponding negative frequency
        valid_neg = -valid_pos

    # Copy the complex conjugates of the positive frequencies into their negative counterparts
    # (excluding index 0, which is its own conjugate)
    fft_partial[valid_neg] = np.conj(fft_full[valid_pos[valid_pos != 0]])

    return fft_partial


def reconstruct_signal(fft_segment):
    # Inverse dft is applied to the partial DFT
    return np.fft.ifft(fft_segment).real


def denoise_bvp_segments(segmentedsig, fs, low_cut, high_cut, ovlap=0.02):
    # Segmented signal is denoised after calculation of low and high-cut
    num_segments, segment_len = segmentedsig.shape
    overlap_len = int(segment_len * ovlap)
    reconstructed = []
    # Partial DFT is calculated from each segment, denoised segment is reconstructed by IDFT and segments are concatenated
    for i in range(num_segments):
        segment = segmentedsig[i]
        fft_partial = partial_dft(
            signal_segment=segment, fs=fs, low_cut=low_cut, high_cut=high_cut
        )
        recon = reconstruct_signal(fft_partial)

        if i == 0:
            reconstructed.extend(recon)
        else:
            reconstructed.extend(
                recon[overlap_len :]
            )  # Overlapping sections are discarded

    return np.array(reconstructed)


def preprocess_bvp(sig, fs, winsz=10, ovlap=0.02):
    """
    sig = Array containing the whole BVP signal
    fs = Sampling frequency of BVP signal
    winsz = Window size for signal segmentation (in seconds), default is 10 seconds
    ovlap = Overlap (0-1 scale), default is 0.02 (2%)
    """
    # Signal is segmented. (Sadhukhan et al., 2018) suggests 10 second segments (as to achieve 0.1 Hz frequency resolution) with 2% overlap, this
    # to avoid Gibbs phenomenon by concatenating signal segments and discarding overlapping sections later on in the preprocessing pipeline.
    segmentedsig = ctf.timewindowpadded(data=sig, fs=fs, winsz=winsz, ovlap=ovlap)

    # First signal segment is lowpass filtered, as to reduce high frequency noise for a more precise peak detection
    sos_bvp = signal.butter(2, 8, "lowpass", fs=fs, analog=False, output="sos")
    filteredsegment = signal.sosfilt(sos_bvp, segmentedsig[0])

    # Low-cut and high-cut frequencies are obtained from the first (filtered) signal segment
    low_cut = get_low_cut(segment=filteredsegment, fs=fs)
    high_cut = get_high_cut(segment=filteredsegment, fs=fs, low_cut=low_cut)

    # Signal is denoised by DFT reconstruction, considering only valid frequencies (inbetween low and high-cut)
    reconstructed_signal = denoise_bvp_segments(
        segmentedsig=segmentedsig,
        fs=fs,
        low_cut=low_cut,
        high_cut=high_cut,
        ovlap=ovlap,
    )

    return reconstructed_signal


### EDA preprocessing functions ###
# Based on methodology from:
# Gautam, A., Simoes-Capela, N., Schiavone, G., Acharyya, A., de Raedt, W., & Van Hoof, C. (2018). A Data Driven Empirical Iterative Algorithm for
# GSR Signal Pre-Processing. 2018 26th European Signal Processing Conference (EUSIPCO), 1162-1166. https://doi.org/10.23919/EUSIPCO.2018.8553191


def getmidpoint(x_min, y_min, x_max, y_max):
    # Midpoints between each pair of (x_min, y_min) and (x_max, y_max) are computed
    x_m = (x_max + x_min) / 2
    y_m = (y_max + y_min) / 2
    return x_m, y_m


def perform_interpolation(sig, t):
    # Signal is inverted to allow detection of local minima as peaks
    inv_sig = -1 * sig

    # Local maxima and minima are detected using peak detection
    max_ind = signal.find_peaks(sig)[0]
    min_ind = signal.find_peaks(inv_sig)[0]

    # Number of usable peak pairs is determined
    N = min(len(max_ind), len(min_ind))
    if N < 2:
        # Interpolation is skipped if fewer than two peak pairs are found
        return None

    # Peak indices are trimmed to match in number
    max_ind = max_ind[:N]
    min_ind = min_ind[:N]

    # Time and amplitude values at peak positions are extracted
    x_min = t[min_ind]
    y_min = sig[min_ind]
    x_max = t[max_ind]
    y_max = sig[max_ind]

    # Midpoints between corresponding minima and maxima are computed
    x_m, y_m = getmidpoint(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    # Cubic spline interpolation is performed using midpoint values
    interp_func = CubicSpline(x_m, y_m)
    return interp_func(t)


def preprocess_eda(sig, fs, iter_num=7, verbose=False):
    """
    sig = Array containing the raw EDA signal
    fs = Sampling frequency of the EDA signal
    iter_num = Number of interpolation cycles to perform (default is 7)
    verbose = Boolean input, enable or disable verbose

    """
    # Datatype is ensured
    sig = np.array(sig)

    t = np.arange(0, len(sig)) * (1 / fs)

    # Copy of the signal is created to start algorithm
    x_t = sig.copy()
    # Signal is iteratively smoothed using interpolation between peak midpoints, 7 iterations are suggested in (Gautam et al., 2018)
    for i in range(1, iter_num + 1):
        interpolated = perform_interpolation(x_t, t)
        if interpolated is None:
            if verbose:
                # Iteration is stopped if insufficient peaks are found
                print(f"Edaprepro: Stopped at iteration {i} due to insufficient peaks.") 
            break
        x_t = interpolated

    # Final x_t component is substracted from the original signal to obtain the residual
    r_t = sig - x_t
    return r_t, i

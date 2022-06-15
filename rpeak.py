import numpy as np
import scipy.signal
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
import sys


def detect_beats(
        ecg,  # The raw ECG scignal
        rate,  # Sampling rate in HZ
        # Window size in seconds to use for
        ransac_window_size=5.0,
        # Low frequency of the band pass filter
        lowfreq=5.0,
        # High frequency of the band pass filter
        highfreq=15.0,
):
    ransac_window_size = int(ransac_window_size * rate)

    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    # TODO: Could use an actual bandpass filter
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

    # Square (=signal power) of the first difference of the signal
    decg = np.diff(ecg_band)
    decg_power = decg ** 2

    # Robust threshold and normalizator estimation
    thresholds = []
    max_powers = []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))

    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0

    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 2

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings


if __name__ == '__main__':
    '''
    ecg = np.loadtxt('ecg_sample.csv')
    '''
    ecg = np.loadtxt('ecg_sample.csv')
    # 여기에서 이제 모든 데이터들이 대입되야함.
    # for 문을 이용해 df = pd.read_csv('./mitdataset/num.csv') 받아와야한다.
    df = pd.read_csv('./mitdataset/101.csv')
    # print(df[df.columns[1]])
    # print(df["'MLII'"])

    ecg = df["'MLII'"].to_numpy()
    print("here")
    print(ecg)
    peaks = detect_beats(ecg, 128)
    # print(len(peaks)) : 1900개
    # print(peaks) : 여기까지가 Q-peak 값 (index) 검출

    print(peaks)

    b = 0
    for i in range(1, len(peaks)):
        interval = peaks[i] - peaks[i - 1]
        # print(interval)
        b = b + interval

    avg_peak_interval = (b / len(peaks) - 1) / 2
    print(avg_peak_interval)  # interval의 가로값
    avg_peak_hight = max(df["'MLII'"]) + 200  # interval의 높이 + 200
    print(avg_peak_hight)
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    print(y)

    plt.plot(x, y)
    plt.scatter(x[peaks], y[peaks], color='red')
    plt.show()

    # plt.savefig('data.png') # 데이터 이미지로 저장하기
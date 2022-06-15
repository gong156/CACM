'''
# ========================================================================================
# D-13일, 평일 2~3 시간, 주말 8시간 / 10월 4일 (25시간)
# ========================================================================================
# 논문 작성 (12일)
# ========================================================================================
'''
# ============================================================================================
# Version 1 - with R peak Algo
# Version 2 - with label
# N: Normal Heartbeat, V: Premature Ventricular Contraction (부정맥)
# Naive QRS_PVC Interval을 통해 시계열 데이터 이미지화 및 저장
# CNN based Arrhythmia Classification Model (미완)
# Evaluation Accuracy (미완)
# ============================================================================================
import os
import cv2
import numpy as np
import scipy.signal
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from evaluation import recall, precision, f1score
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================================================
# Version 1 - with R peak Algo -
# ============================================================================================

def detect_rpeaks(
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

def generate_plot_error(ecg, peaks, sequence):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    times = int(len(ecg) / sequence)
    for i in range(times):
        # print(peaks[(peaks < (i + 1) * sequence) & (peaks > i * sequence)]) # 에러 검출용 좌표 프린트
        plt.plot(x[i * sequence:(i + 1) * sequence], y[i * sequence:(i + 1) * sequence])
        plt.scatter(x[peaks][(peaks < (i + 1) * sequence) & (peaks > i * sequence)],
                    y[peaks][(peaks < (i + 1) * sequence) & (peaks > i * sequence)], color='red')
        #plt.show()

def delete_rpeak_errors(peaks, patient):
    error_array = 0
    if patient == 101:
        error_array = np.array(
            [11320, 11391, 11579, 13829, 14734, 40889, 41112, 41157, 41430, 47754, 48685, 48924, 49059, 49193,
             49257, 49475, 49556, 49616, 49810, 49973, 50361, 50743, 63095, 63407, 114713, 115163, 115354, 115475,
             116903, 116979, 117178, 129814, 309502, 314086, 335018, 335332])
    if patient == 106:
        error_array = np.array(
            [29483, 31580, 45547, 49198, 88885, 161241, 161595,214965, 261569,263235, 282433, 291240,307382,
             307536,322502,327133,328244, 329396, 349253, 378450, 379656,382258,388102,390982,392454,395917,
             514210,519519,520153,537348,558175,618763,626229,629679,631585,645391])
    index_error = []
    for i in range(len(peaks)):
        for error in error_array:
            if peaks[i] == error:
                index_error.append(i)
    peaks = np.delete(peaks, index_error, None)
    peaks = peaks[1:]
    return peaks

def generate_image_frame(ecg, peaks, patient):
    interval_arr = []
    for i in range(1, len(peaks)):
        interval = peaks[i] - peaks[i - 1]
        interval_arr.append(interval)
    avg_peak_hight = ecg.max()  # interval의 높이
    avg_peak_width = int(interval.mean())  # interval의 가로
    if patient == 106:
        avg_peak_width = round(avg_peak_width,-2)
        return avg_peak_width, avg_peak_hight
    else:
        return avg_peak_width, avg_peak_hight

def make_ECG_image_per_peak(ecg, peaks, avg_peak_width, patient):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    i = 1
    for peak in peaks:
        plt.plot(x[int(peak - avg_peak_width / 2):int(peak + avg_peak_width / 2)],
                 y[int(peak - avg_peak_width / 2):int(peak + avg_peak_width / 2)])
        plt.axis('off')
        # plt.show()
        file_name = 'mitdataset/' + patient + '_heartbeat_images/' + str(patient) + '_' + str(i) + 'beat.png'
        plt.savefig(file_name)
        plt.cla()
        print(str(patient) + '_' + str(i) + 'beat.png')
        i = i + 1

def make_PVC_image_per_peak(ecg,V_peaks, avg_peak_interval):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    i = 1
    for peak in V_peaks:
        plt.plot(x[int(peak - avg_peak_interval / 2):int(peak + avg_peak_interval / 2)],
                 y[int(peak - avg_peak_interval / 2):int(peak + avg_peak_interval / 2)])
        plt.axis('off')
        plt.show()
        #file_name = 'mitdataset/106_heartbeat_images/' + str(patient) + '_' + str(i) + 'beat.png'
        #plt.savefig(file_name)
        #plt.cla()
        #print(str(patient) + '_' + str(i) + 'beat.png')
        print(peak)
        i = i + 1

# ============================================================================================
# Version 2 - with label -
# ============================================================================================
def load_annotations(anno_path):
    annotation_df = pd.read_csv(anno_path)
    anno = pd.concat(
        [annotation_df[annotation_df.columns[2]].to_frame(), annotation_df[annotation_df.columns[3]].to_frame()],
        axis=1)
    for i in range(len(anno)):
        if anno['Type'][i] == '+' or anno['Type'][i] == '~':
            anno = anno.drop(index=i)
    anno = anno.reset_index(drop=True)
    return anno

def Naive_QRS_PVC_Interval_Setting(anno):
    pvc_interval_array = []
    for i in range(1, len(anno)):
        if anno['Type'][i - 1] == 'N' and anno['Type'][i] == 'V':
            pvc_interval_array.append(anno[anno.columns[0]][i] - anno[anno.columns[0]][i - 1])
        elif anno['Type'][i - 1] == 'V' and anno['Type'][i] == 'V':
            pvc_interval_array.append(anno[anno.columns[0]][i] - anno[anno.columns[0]][i - 1])
    pvc_interval = pd.DataFrame(pvc_interval_array)
    avg_pvc_interval = round(pvc_interval.mean())
    avg_pvc_interval = int(avg_pvc_interval[0])
    return avg_pvc_interval

def spectrogram(data, nperseg, log_spectrogram=True): # nperseg = n per segment
    # Spectrograms can be used as a way of visualizing the change of a nonstationary signal’s frequency content over time.
    # 스펙트로그램은 시간에 따른 비정상 신호의 주파수 내용 변화를 시각화하는 방법으로 사용할 수 있다.
    fs = nperseg #
    # print(1,0,int(data.shape[0]/nperseg))
    # print('frequency: ',len(np.fft.rfftfreq(nperseg, 1/fs))) # for deleting white noise
    frequency, time, Sxx = scipy.signal.spectrogram(data, fs=fs, nperseg=nperseg)
    Sxx = np.transpose(Sxx,[0,2,1])
    if log_spectrogram:
        Sxx = abs(Sxx) # Make sure, all values are positive before taking log
        mask = Sxx > 0 # We dont want to take the log of zero
        Sxx[mask] = np.log(Sxx[mask])
    return frequency, time, Sxx

def Make_ECG_Images(data_path, patient, ecg, avg_pvc_interval, data_type):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    plt.figure(figsize=(1, 1),dpi=128)
    for i in range(len(anno)):
        peak = anno[anno.columns[0]][i]
        type = anno[anno.columns[1]][i]
        signal_x = x[int(peak - avg_pvc_interval / 2):int(peak + avg_pvc_interval / 2)]
        signal_y = y[int(peak - avg_pvc_interval / 2):int(peak + avg_pvc_interval / 2)]
        if data_type == 'Simple':
            plt.plot(signal_x, signal_y)
            plt.axis('off')
            file_name = data_path + str(patient) + '_' + str(i + 1) + 'beat_' + type + '.png'
            plt.savefig(file_name)
            plt.clf()
            print(str(patient) + '_' + str(i) + 'beat_' + type + '.png')
            # plt.show()
        if data_type == 'Spectrogram':
            file_name = data_path + str(patient) + '_' + str(i + 1) + 'beat_' + type + '.png'
            data = np.expand_dims(signal_y, axis=0)
            Sx = spectrogram(data, 10, log_spectrogram=False)[2]
            Sx_log = spectrogram(data, 10, log_spectrogram=True)[2]
            f, t, _ = spectrogram(data, 10, log_spectrogram=False)
            # 스펙트럼 이미지 resizing
            # Plot the spectrograms as images
            img = Sx[0]
            # img = Sx_log[0]
            plt.imshow(np.transpose(img), aspect='auto', cmap='jet')
            plt.axis('off')
            plt.savefig(file_name)
            plt.clf()
            print(str(patient) + '_' + str(i) + 'beat_' + type + '.png')

def Concat_ECGImages(data_path, data_type):
    i = 0
    normal_img = 0
    count_normal = 0
    arrhythmia_img = 0
    count_arrhythmia = 0
    file_list = os.listdir(data_path)
    for file_name in file_list:
        print('Preprocessing for', file_name)
        img = cv2.imread(data_path + file_name)
        if data_type == 'Simple':
            img_black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_black = img_black.reshape(1, img_black.shape[0], img_black.shape[1], 1)
        if data_type == 'Spectrogram':
            img_black = img.reshape(1, img.shape[0], img.shape[1], 3)
        if file_name.split('_')[2].split('.')[0] == 'N':
            if count_normal == 0:
                normal_img = img_black
            else:
                normal_img = np.concatenate((normal_img, img_black), axis=0)
            count_normal = count_normal + 1

        if file_name.split('_')[2].split('.')[0] == 'V':
            if count_arrhythmia == 0:
                arrhythmia_img = img_black
            else:
                arrhythmia_img = np.concatenate((arrhythmia_img, img_black), axis=0)
            count_arrhythmia = count_arrhythmia + 1
        i = i + 1
    return normal_img, arrhythmia_img

def Slide_Images_for_5_Fold(normal_img, arrhythmia_img, rate):
    norm_length_20p = int(normal_img.shape[0] * round(1-rate,1))
    arrh_length_20p = int(arrhythmia_img.shape[0] * round(1-rate,1))

    # 1th
    norm_1th = normal_img[:norm_length_20p, :, :, :]
    arrh_1th = arrhythmia_img[:arrh_length_20p, :, :, :]

    # 2th
    norm_2th = normal_img[norm_length_20p:int(2 * norm_length_20p), :, :, :]
    arrh_2th = arrhythmia_img[arrh_length_20p:int(2 * arrh_length_20p), :, :, :]

    # 3th
    norm_3th = normal_img[int(2 * norm_length_20p):int(3 * norm_length_20p), :, :, :]
    arrh_3th = arrhythmia_img[int(2 * arrh_length_20p):int(3 * arrh_length_20p), :, :, :]

    # 4th
    norm_4th = normal_img[int(3 * norm_length_20p):int(4 * norm_length_20p), :, :, :]
    arrh_4th = arrhythmia_img[int(3 * arrh_length_20p):int(4 * arrh_length_20p), :, :, :]

    # 5th
    norm_5th = normal_img[int(4 * norm_length_20p):, :, :, :]
    arrh_5th = arrhythmia_img[int(4 * arrh_length_20p):, :, :, :]

    array_norm = [norm_1th, norm_2th, norm_3th, norm_4th, norm_5th]
    array_arrh = [arrh_1th, arrh_2th, arrh_3th, arrh_4th, arrh_5th]

    return array_norm, array_arrh

def Divide_Train_and_Test(array):
    test_array = []
    train_array = []
    for i in range(len(array)):  # 5번
        nums = [num for num in range(5)]
        test = array[nums[i]]
        #print('test',nums[i])
        #print('test',array[nums[i]].shape)
        nums.pop(i)
        #print('train',nums)
        train = 0
        j = 0
        for num in nums:  # 4번
            #print('train',array[num].shape)
            if j == 0:
                train = array[num]
            else:
                train = np.concatenate((train, array[num]), axis=0)
            j = j + 1
        test_array.append(test)
        train_array.append(train)
    return test_array, train_array

def Concat_Norm_and_Arrh_with_Save(_type, norm_array, arrh_array, dir_path):
    train_arry = []
    test_arry = []
    i = 1
    for norm, arrh in zip(norm_array, arrh_array):
        train = np.concatenate((norm, arrh), axis=0)
        label_norm = np.array([0 for i in range(len(norm))])
        label_arrh = np.array([1 for i in range(len(arrh))])
        test = np.concatenate((label_norm, label_arrh))
        np.save(dir_path + str(patient) + '_X_' + _type + '_' + str(i) + '_fold', train)
        np.save(dir_path + str(patient) + '_Y_' + _type + '_' + str(i) + '_fold', test)
        train_arry.append(train)
        test_arry.append(test)
        i = i + 1
    return train_arry, test_arry
# ============================================================================================

if __name__ == '__main__':
    patient = 106
    learning_rate = 0.8
    # data_type = 'Simple'
    data_type = 'Spectrogram'
    data_path = 'mitdataset/' + str(patient) + '_heartbeat_images/' + data_type + 'Images/'
    input_images_path = 'mitdataset/' + str(patient) + '_input_images.npy'
    label_path = 'mitdataset/' + str(patient) + '_label.npy'
    pre_dir_path = 'mitdataset/preprocessing_'+str(patient)+'/'+ data_type + '/'
    X_train_array = []
    Y_train_array = []
    X_test_array = []
    Y_test_array = []
    if len(os.listdir(pre_dir_path)) != 0 and len(os.listdir(data_path)) != 0:
        print('the input and output file are existed')
        for i in range(1,6):
            _type = 'train'
            X_train_array.append(np.load(pre_dir_path + str(patient) + '_X_' + _type + '_' + str(i) + '_fold.npy'))
            Y_train_array.append(np.load(pre_dir_path + str(patient) + '_Y_' + _type + '_' + str(i) + '_fold.npy'))
            _type = 'test'
            X_test_array.append(np.load(pre_dir_path + str(patient) + '_X_' + _type + '_' + str(i) + '_fold.npy'))
            Y_test_array.append(np.load(pre_dir_path + str(patient) + '_Y_' + _type + '_' + str(i) + '_fold.npy'))
    else:
        ecg_df = pd.read_csv('mitdataset/' + str(patient) + '.csv')  # 106번 환자 데이터 로드
        ecg = ecg_df["'MLII'"].to_numpy()  # peak의 총 개수: 1900개
        rate = float(128)
        # ========================================================================================
        # Version 2 - with label -
        # ========================================================================================
        # 라벨링 - N: Normal Heartbeat, V: Premature Ventricular Contraction (부정맥)
        anno_path = 'mitdataset/' + str(patient) + 'annotations.csv'
        anno = load_annotations(anno_path)
        # Naive QRS_PVC Interval 설정
        avg_pvc_interval = Naive_QRS_PVC_Interval_Setting(anno)
        # ECG 데이터 이미지화 및 저장
        if len(os.listdir(data_path)) == 0:
            Make_ECG_Images(data_path, patient, ecg, avg_pvc_interval, data_type)
        # ECG 이미지 로드 및 합치는거 (CNN을 위해) - 데이터 셋 구성

        normal_img, arrhythmia_img = Concat_ECGImages(data_path, data_type)
        # 이미지 5-Fold Cross Validation
        array_norm, array_arrh = Slide_Images_for_5_Fold(normal_img, arrhythmia_img, learning_rate)
        norm_test_array, norm_train_array = Divide_Train_and_Test(array_norm)
        #print('test:', norm_test_array[0].shape,',train:',norm_train_array[0].shape)
        arrh_test_array, arrh_train_array = Divide_Train_and_Test(array_arrh)
        #print('test:', arrh_test_array[0].shape,',train:',arrh_train_array[0].shape)
        _type = 'train'
        X_train_array, Y_train_array = Concat_Norm_and_Arrh_with_Save(_type, norm_train_array, arrh_train_array, pre_dir_path)
        _type = 'test'
        X_test_array, Y_test_array = Concat_Norm_and_Arrh_with_Save(_type, norm_test_array, arrh_test_array, pre_dir_path)

    # CNN based Arrhythmia Classification Model 생성: Image, Spectrogram
    i = 0

    for i in range(len(X_train_array)):
        X_train = X_train_array[i]
        Y_train = Y_train_array[i]
        X_test = X_test_array[i]
        Y_test = Y_test_array[i]
        print("==========================================================================================")
        print((i+1),"번째 실험")
        print("==========================================================================================")
        model_path = 'model/' + data_type + '/' + str(patient) + '_model_weights_' + str(i) + '_fold.h5'
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)
        dimension = 0
        if data_type == 'Simple':
            dimension = 1
        if data_type == 'Spectrogram':
            dimension = 3
        model = Sequential()  # 128 x 128
        model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', activation='elu', input_shape=(X_train.shape[1], X_train.shape[2], dimension)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', activation='elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', activation='elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', activation='elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2048, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1score])
        model.summary()
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=20, validation_split=0.1)
        #model.save(model_path, overwrite=True)
        _loss, _acc, _precision, _recall, _f1score = model.evaluate(X_test, Y_test, verbose=2)
        f = open('result/' + data_type + '/' + str(patient) + '_model_weights_' + str(i) + '_fold.txt', 'w')
        data = "loss: " + str(_loss) + "\n" + "acc: " + str(_acc) + "\n" + "recall: " + str(_recall) + "\n" + "flscore: " + str(_f1score) + "\n"
        f.write(data)
        f.close()
        i = i + 1

    # 5-Fold Cross Validation data 관련
    # 모델 생성 저장 및 로드
    # 분류 성능 평가 지표 뽑는법  



    # ========================================================================================
    # Verision 1 - with R peak Algo -
    # ========================================================================================
    '''
    peaks = detect_rpeaks(ecg, 128) # R-peak 값 (index) 검출
    sequence = 1000 # 확인할 그림 가로 길이
    generate_plot_error(ecg, peaks, sequence) # 에러 제거용 Visualization
    # 전처리 단
    peaks = delete_rpeak_errors(peaks, patient)
    avg_peak_width, avg_peak_hight = generate_image_frame(ecg, peaks,  patient)
    make_ECG_image_per_peak(ecg, peaks, avg_peak_width)
    '''
    # ========================================================================================


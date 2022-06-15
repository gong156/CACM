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
# Version 2 - with label -
# ============================================================================================
def setting_path(patient, data_dir_path, data_type):
    data_image_path = data_dir_path + '/' + str(patient) + '_heartbeat_images'
    datatype_image_path = data_image_path + '/' + data_type + 'Images'
    if not os.path.exists(data_image_path):
        os.makedirs(data_image_path, exist_ok=True)
    if not os.path.exists(datatype_image_path):
        os.makedirs(datatype_image_path, exist_ok=True)

    data_folding_path = data_dir_path + '/preprocessing_' + str(patient)
    datatype_folding_path = data_folding_path + '/' + data_type
    if not os.path.exists(data_folding_path):
        os.makedirs(data_folding_path, exist_ok=True)
    if not os.path.exists(datatype_folding_path):
        os.makedirs(datatype_folding_path, exist_ok=True)
    return datatype_image_path, datatype_folding_path


def load_annotations(anno_path):
    annotation_df = pd.read_csv(anno_path)
    anno = pd.concat(
        [annotation_df[annotation_df.columns[2]].to_frame(), annotation_df[annotation_df.columns[3]].to_frame()],
        axis=1)
    for i in range(len(anno)):
        if anno['Type'][i] == 'L' or anno['Type'][i] == 'R' or anno['Type'][i] == 'e' or anno['Type'][i] == 'j':
            anno['Type'][i] = 'N'
        if anno['Type'][i] == 'A' or anno['Type'][i] == 'a' or anno['Type'][i] == 'J' or anno['Type'][i] == 'S' \
            or anno['Type'][i] == 'F' or anno['Type'][i] == '/' or anno['Type'][i] == 'f' or anno['Type'][i] == 'Q' \
            or anno['Type'][i] == '+' or anno['Type'][i] == '|' or anno['Type'][i] == '~' or anno['Type'][i] == '/' \
            or anno['Type'][i] == '!' or anno['Type'][i] == '[' or anno['Type'][i] == ']' or anno['Type'][i] == '"' \
            or anno['Type'][i] == 'x':
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

def Make_ECG_Images(data_image_path, anno, patient, ecg, avg_pvc_interval, data_type):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    plt.figure(figsize=(1, 1),dpi=128)
    for i in range(len(anno)):
        peak = anno[anno.columns[0]][i]
        type = anno[anno.columns[1]][i]
        if peak > avg_pvc_interval:
            signal_x = x[int(peak - avg_pvc_interval / 2):int(peak + avg_pvc_interval / 2)]
            signal_y = y[int(peak - avg_pvc_interval / 2):int(peak + avg_pvc_interval / 2)]
            if data_type == 'Simple':
                plt.plot(signal_x, signal_y)
                plt.axis('off')
                file_name = data_image_path + str(patient) + '_' + str(i + 1) + 'beat_' + type + '.png'
                plt.savefig(file_name)
                plt.clf()
                print(str(patient) + '_' + str(i) + 'beat_' + type + '.png')
                # plt.show()
            if data_type == 'Spectrogram':
                file_name = data_image_path + str(patient) + '_' + str(i + 1) + 'beat_' + type + '.png'
                data = np.expand_dims(signal_y, axis=0)
                Sx = spectrogram(data, 10, log_spectrogram=False)[2]
                f, t, _ = spectrogram(data, 10, log_spectrogram=False)
                # Plot the spectrograms as images
                img = Sx[0]
                # img = Sx_log[0]
                plt.imshow(np.transpose(img), aspect='auto', cmap='jet')
                plt.axis('off')
                plt.savefig(file_name)
                plt.clf()
                print(str(patient) + '_' + str(i) + 'beat_' + type + '.png')

def Concat_ECGImages(data_image_path, data_type):
    i = 0
    normal_img = 0
    count_normal = 0
    arrhythmia_img = 0
    count_arrhythmia = 0
    file_list = os.listdir(data_image_path)
    for file_name in file_list:
        print('Preprocessing for', file_name)
        img = cv2.imread(data_image_path + file_name)
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
    patient = 111
    learning_rate = 0.8
    data_dir_path = 'mitdataset'
    # data_type = 'Simple'
    data_type = 'Spectrogram'
    datatype_image_path, datatype_folding_path = setting_path(patient, data_dir_path, data_type)

    data_image_path = datatype_image_path + '/'
    datatype_folding_path = datatype_folding_path + '/'

    X_train_array = []
    Y_train_array = []
    X_test_array = []
    Y_test_array = []
    if len(os.listdir(datatype_folding_path)) != 0 and len(os.listdir(data_image_path)) != 0:
        print('the input and output file are existed')
        for i in range(1,6):
            _type = 'train'
            X_train_array.append(np.load(datatype_folding_path + str(patient) + '_X_' + _type + '_' + str(i) + '_fold.npy'))
            Y_train_array.append(np.load(datatype_folding_path + str(patient) + '_Y_' + _type + '_' + str(i) + '_fold.npy'))
            _type = 'test'
            X_test_array.append(np.load(datatype_folding_path + str(patient) + '_X_' + _type + '_' + str(i) + '_fold.npy'))
            Y_test_array.append(np.load(datatype_folding_path + str(patient) + '_Y_' + _type + '_' + str(i) + '_fold.npy'))
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
        if len(os.listdir(data_image_path)) == 0:
            Make_ECG_Images(data_image_path, anno, patient, ecg, avg_pvc_interval, data_type)
        # ECG 이미지 로드 및 합치는거 (CNN을 위해) - 데이터 셋 구성

        normal_img, arrhythmia_img = Concat_ECGImages(data_image_path, data_type)
        # 이미지 5-Fold Cross Validation
        array_norm, array_arrh = Slide_Images_for_5_Fold(normal_img, arrhythmia_img, learning_rate)
        norm_test_array, norm_train_array = Divide_Train_and_Test(array_norm)
        #print('test:', norm_test_array[0].shape,',train:',norm_train_array[0].shape)
        arrh_test_array, arrh_train_array = Divide_Train_and_Test(array_arrh)
        #print('test:', arrh_test_array[0].shape,',train:',arrh_train_array[0].shape)
        _type = 'train'
        X_train_array, Y_train_array = Concat_Norm_and_Arrh_with_Save(_type, norm_train_array, arrh_train_array, datatype_folding_path)
        _type = 'test'
        X_test_array, Y_test_array = Concat_Norm_and_Arrh_with_Save(_type, norm_test_array, arrh_test_array, datatype_folding_path)

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
        print('X_train:',X_train.shape)
        print('Y_train:',Y_train.shape)
        print('X_test:',X_test.shape)
        print('Y_test:',Y_test.shape)
        dimension = 0
        if data_type == 'Simple':
            dimension = 1
        if data_type == 'Spectrogram':
            dimension = 3
        model = 0
        if os.path.isfile(model_path):
            models.load_model(model_path)
        else:
            model = Sequential()  # 128 x 128
            model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', activation='elu',
                             input_shape=(X_train.shape[1], X_train.shape[2], dimension)))
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
            model.compile(loss='binary_crossentropy', optimizer='adam',
                          metrics=['accuracy', precision, recall, f1score])
            model.summary()
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=20,
                      validation_split=0.1)
            model.save(model_path, overwrite=True)
        _loss, _acc, _precision, _recall, _f1score = model.evaluate(X_test, Y_test, verbose=2)
        f = open('result/' + data_type + '/' + str(patient) + '_model_weights_' + str(i) + '_fold.txt', 'w')
        data = "loss: " + str(_loss) + "\n" + "acc: " + str(_acc) + "\n"
        f.write(data)
        f.close()
        i = i + 1

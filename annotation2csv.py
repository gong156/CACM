import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_and_make_annotation(dir):
    files = os.listdir(dir)
    for file in files:
        if file.find("annotations.txt") != -1:
            num = file.split("annotations.txt")[0]
            print(num + 'annotations.csv')
            if os.path.isfile('./mitdataset/' + num + 'annotations.csv'):
                continue
            else:
                path = './mitdataset/' + num + 'annotations.txt'
                df_annotations = annotation_txt2csv(path)
                df_annotations.to_csv('./mitdataset/' + num + 'annotations.csv')


def annotation_txt2csv(path):
    f = open(path, 'r')
    lines = f.readlines()
    df_annotations = pd.DataFrame(columns=['Time', 'Sample #', 'Type', 'Sub', 'Chan', 'Num'])
    for line in enumerate(lines):
        if line[0] == 0:
            continue
        else:
            values = []
            for i in line[1].split(" "):
                if i not in values:
                    values.append(i)
            temp = pd.DataFrame({'Time': [values[1]], 'Sample #': [values[2]], 'Type': [values[3]], 'Sub': [values[4]], 'Chan': ['0'], 'Num': ['0']})
            df_annotations = pd.concat([df_annotations, temp], axis=0)
    f.close()
    df_annotations.reset_index(drop=True)
    return df_annotations

 # Verision 2 (D-1)
    # ==================================================================================q
    # Annotation에서 V에 해당하는 시그널만 불러오고
    # PVC 신호 자르기
    # PVC 신호 이미지화 및 저장
    # ==================================================================================

def ecgtype_index():
   df = pd.read_csv('./mitdataset/106annotations.csv')
   ecgtype = df[df.columns[3]]
   np_type = ecgtype.to_numpy()
   print(np_type)

#v_index = np.where(np_type == "'V'")
#print(V_inedx)

   # plt.show()
'''
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

def generate_image_frame(ecg,peaks,patient):
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

def make_ECG_image_per_Vpeak(ecg, peaks, avg_peak_width):
    dt = 1 / 128
    x = np.linspace(0, len(ecg) * dt, len(ecg))
    y = ecg
    i = 1
    for peak in peaks:
        plt.plot(x[int(peak - avg_peak_width / 2):int(peak + avg_peak_width / 2)],
                 y[int(peak - avg_peak_width / 2):int(peak + avg_peak_width / 2)])
        plt.axis('off')
        # plt.show()
        file_name = 'mitdataset/106_heartbeat_images/' + str(patient) + '_' + str(i) + 'beat.png'
        plt.savefig(file_name)
        plt.cla()
        print(str(patient) + '_' + str(i) + 'beat.png')
        i = i + 1
'''
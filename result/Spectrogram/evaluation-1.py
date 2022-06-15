import pandas as pd
import re
import os



patients = [100, 105, 106, 108, 109, 111, 116, 118, 119, 121, 123, 124200, 201, 202, 203, 205,
            207, 208, 209, 210, 213, 214, 215, 217, 219, 221, 223, 228, 230, 231, 233, 234]

patient = 111

if patient in patients:
    print(str(patient)+'번 환자에게서 심실세동이 관찰되었습니다.'+ "\n" + str(patient)+"환자의 5-Fold Cross Validation에 따른 Acc는 다음과 같습니다")
    i=0
    for i in range(5):
        f = open(
            "C:/Users/USER/Desktop/AhyunKim/[2021 추계 한국스마트미디어학회] CNN 기반 ECG 부정맥 심장 박동 분류/CACM/result/Spectrogram/" +
            str(patient) + "_model_weights_" + str(i) + "_fold.txt")
        lines = f.read().splitlines()
        data = lines[1]
        a = data.split()
        b = round(float(a.pop(1)), 4)
        c = 100 * b
        print(c)
    r = open("C:/Users/USER/Desktop/AhyunKim/[2021 추계 한국스마트미디어학회] CNN 기반 ECG 부정맥 심장 박동 분류/CACM/result/Spectrogram/" +
    str(patient) + "_model_acc_average.txt")
    lines1 = r.read()
    print("이 환자에게 검출된 부정맥 비트의 acc 평균은 " + lines1 +" 입니다.")
else:
    print(str(patient)+'번 환자에게서 심실세동이 관찰되지 않았습니다.')


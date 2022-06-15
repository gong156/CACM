import pandas as pd
import re
import os

patient = 106

for i in range(5):
    f = open("C:/Users/USER/Desktop/AhyunKim/[2021 추계 한국스마트미디어학회] CNN 기반 ECG 부정맥 심장 박동 분류/CACM/result/"+str(patient)+"_model_weights_" + str(i) + "_fold.txt")
    lines = f.read().splitlines()
    data = lines[1]
    a = data.split()
    b = round(float(a.pop(1)),4)
    c = 100*b
    print(c)


'''
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
'''
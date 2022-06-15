import os

patient = 114

path = os.getcwd()
path = path.replace('result', 'mitdataset')
heartbeat_path = path + f'\\{patient}_heartbeat_images'
SimpleImages_path = f'{heartbeat_path}\\SimpleImages'
SpectrogramImages_path = f'{heartbeat_path}\\SpectrogramImages'
os.makedirs(SimpleImages_path, exist_ok=True)
os.makedirs(SpectrogramImages_path, exist_ok=True)
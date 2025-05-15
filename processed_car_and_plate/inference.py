from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import natsort

# 학습된 모델 불러오기
model = YOLO('/mnt/hdd_6tb/bill0914/YOLOv11n_carcrop.pt')
print(model.names)  # 클래스 이름 리스트 확인
input_folder = "/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터"



input_folder_list = natsort.natsorted(os.listdir(input_folder)) #5개 [cr06][cr11][cr12][cr13][cr14]
idx=1
total=0
for subfolder in input_folder_list:
     subfolder_path = os.path.join(input_folder,subfolder) #/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터/cr06
     subfolder_list = natsort.natsorted(os.listdir(subfolder_path)) #01번 02번 03번 04번
     for jpgs in subfolder_list:
        jpgs_path = os.path.join(subfolder_path,jpgs)
        for js in natsort.natsorted(os.listdir(jpgs_path)):
            results = model(os.path.join(jpgs_path,js))[0]  
            results.save(filename=f"/mnt/hdd_6tb/bill0914/carandplate/test/result{idx}.png")  # save to disk
            for result in results:
                total+=len(result.boxes)
           

            idx+=1         
print(total) 
           
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import natsort

# 학습된 모델 불러오기
model = YOLO('/mnt/hdd_6tb/bill0914/processed_carcrop/runs/detect/train3-roboflow/weights/best.pt')

# 테스트할 이미지 폴더 경로
save_folder = "/mnt/hdd_6tb/bill0914/processed_original/test"

input_folder = "/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터"
output_dir ="/mnt/hdd_6tb/bill0914/processed_original/test/test_bbox_count"
input_folder_list = natsort.natsorted(os.listdir(input_folder)) #5개 [cr06][cr11][cr12][cr13][cr14]
idx=1
total=0
for subfolder in input_folder_list:
     subfolder_path = os.path.join(input_folder,subfolder) #/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터/cr06
     subfolder_list = natsort.natsorted(os.listdir(subfolder_path)) #01번 02번 03번 04번
     for jpgs in subfolder_list:
        jpgs_path = os.path.join(subfolder_path,jpgs)
        for js in natsort.natsorted(os.listdir(jpgs_path)):
            results = model(os.path.join(jpgs_path,js))  
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs                       
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs        
                result.save(filename=f"/mnt/hdd_6tb/bill0914/processed_carcrop/test/result{idx}.png")  # save to disk
                print(f"bbox의 개수는:{len(boxes)}") 
                total+=len(boxes)
                idx+=1
print(total)  
           
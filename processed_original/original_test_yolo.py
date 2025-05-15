from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import natsort
import cv2
import matplotlib.pyplot as plt
# 학습된 모델 불러오기
model = YOLO('/mnt/hdd_6tb/bill0914/processed_original/runs/detect/train2-원본+DA/weights/best.pt')

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
            im=cv2.imread(os.path.join(jpgs_path,js))
            plt.imshow(im)
            results = model(os.path.join(jpgs_path,js))  
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs                       
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs        
                result.save(filename=f"/mnt/hdd_6tb/bill0914/processed_original/test/result{idx}.png")  # save to disk
                print(f"bbox의 개수는:{len(boxes)}") 
                total+=len(boxes)
              
                txt_filename = f'test_bbox_count{idx}.txt'
                txt_path = os.path.join(output_dir, txt_filename)
                with open(txt_path, 'w') as f:
                    f.write(str(len(boxes)))
            idx+=1         
with open("/mnt/hdd_6tb/bill0914/processed_original/test/test_bbox_count/total_test_bbox_count.txt",'w') as f:
    f.write(str(total))   
           
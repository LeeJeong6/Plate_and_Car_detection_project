'''
모델이 탐지한 bbox와 라벨링 bbox의 정확도를 비교한다.
1.모델 불러오기
2.테스트하기
3.detections.xyxy를 통해 IOU계산하여 정확도가 50%넘으면 같은 bbox라고 인식하여 개수 세기
'''
from ultralytics import YOLO
import os
from natsort import natsorted 
import cv2
import supervision as sv

model =YOLO("/mnt/hdd_6tb/bill0914/mak/runs/detect/train2-원본+4분할+resize+DA1/weights/best.pt")

input_folder = "/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터"
label_folder = "/mnt/hdd_6tb/bill0914/mak/테스트 데이터 bbox좌표"

input_folder_list = natsorted(os.listdir(input_folder))
label_folder_list = natsorted(os.listdir(label_folder))

count=0 #iou 50%이상
seven_count=0 #iou 70%이상
nine_count=0 #iou 90%이상
k=0 #for문 label_folder_list
total_label=0 #전체 라벨 bbox 940개
detect_bbox = 0 #탐지한 bbox 수

def IOU(detect_xyxy,xyxy_list):
    
    box1area = (detect_xyxy[2]-detect_xyxy[0]+1)*(detect_xyxy[3]-detect_xyxy[1]+1)
    box2area = (xyxy_list[2]-xyxy_list[0]+1)*(xyxy_list[3]-xyxy_list[1]+1)
    x1 = max(detect_xyxy[0],xyxy_list[0])
    y1 = max(detect_xyxy[1],xyxy_list[1])
    x2 = min(detect_xyxy[2],xyxy_list[2])
    y2 = min(detect_xyxy[3],xyxy_list[3])
    w = max(0,x2-x1+1)
    h = max(0,y2-y1+1)
    inter = w*h
    iou = inter/(box1area+box2area-inter)
    return iou

for subfolder in input_folder_list: #cr06,cr11,cr12,cr13,cr14
    subfolder_path = os.path.join(input_folder, subfolder)
    subfolder_list = natsorted(os.listdir(subfolder_path)) 
    
    for jpgs in subfolder_list: #01,02,03,04,05
        jpgs_path = os.path.join(subfolder_path, jpgs) #/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터/[cr06]호계사거리/01번
        image_list = natsorted(os.listdir(jpgs_path)) #사진들 리스트

        for i in range(len(image_list)):
            image_path = os.path.join(jpgs_path, image_list[i])
            results = model(image_path)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            with open(os.path.join(label_folder,label_folder_list[k]),'r') as f:
                bbox_label_list = f.readlines()
                total_label += len(bbox_label_list)  
                detect_bbox += len(detections.xyxy) 
            if len(detections.xyxy) !=0:
                for i in range(len(detections.xyxy)): #모델이 탐지한 xyxy
                    for j in range(len(bbox_label_list)): #정답 xyxy
                         
                        xyxy_list = []
                        xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[0]))
                        xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[1]))
                        xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[0])+int(bbox_label_list[j].strip('\n').split(' ')[2]))
                        xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[1])+int(bbox_label_list[j].strip('\n').split(' ')[3]))
                        
                        iou = IOU(detections.xyxy[i],xyxy_list)
                        if iou > 0.5 :
                            print(f"{detections.xyxy[i]} vs {xyxy_list} -> {iou*100}%")
                            count+=1
                        if iou > 0.75 : 
                            seven_count +=1
                        if iou > 0.9 : 
                            nine_count += 1    
            k+=1                    
print(f"라벨 bbox개수 : {total_label}")            
print(f"탐지한 bbox개수 : {detect_bbox}")
print(f"iou-threshold가 50%일때 : {count}")
print(f"iou-threshold가 70%일때 : {seven_count}")
print(f"iou-threshold가 90%일때 : {nine_count}")

                        






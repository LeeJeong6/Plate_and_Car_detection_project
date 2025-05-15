'''
모델이 탐지한 bbox와 라벨링 bbox의 정확도를 비교한다.
1.모델 불러오기
2.4분할 +마진10%로 테스트하기
3.이미지 합친 뒤 NMS함수 만들어서 적용하기
4.IOU함수 만들어서 마진에 겹친 bbox들 쳐내기
5.bbox개수 세기 
'''

import os
from natsort import natsorted
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import time
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # MB 단위로 반환
def calculate_area(bbox):
    """ 주어진 bbox의 면적을 계산 """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def NMS_only_xyxy(bbox_list, iou_threshold=0.2):
    '''
    가장 넓은 bbox를 우선으로 하여 NMS를 적용하는 함수

    Args:
        bbox_list (list) : [[x1,y1,x2,y2],[x3,y3,x4,y4],,,]로 저장된 bbox 리스트
        iou_threshold (float): IoU 임계값 (기본값 0.2)

    Returns:
        result_bbox : NMS를 거쳐 선택된 bbox 리스트
    '''
    if len(bbox_list) == 0:
        return []

    # bbox를 면적 기준으로 내림차순 정렬
    bbox_list = sorted(bbox_list, key=calculate_area, reverse=True)
    
    selected_bboxes = []
    
    while bbox_list:
        # 가장 넓은 bbox를 선택
        base_bbox = bbox_list.pop(0)
        selected_bboxes.append(base_bbox)
        
        # 남은 bbox 중 IoU가 threshold 이상인 bbox 제거
        bbox_list = [bbox for bbox in bbox_list if calculate_iou(base_bbox, bbox) <= iou_threshold]

    return selected_bboxes
def calculate_iou(xyxy1, xyxy2):
    """
    두 bounding box의 IOU 를 계산
    
    Args:
        xyxy1 (list): 첫 번째 bounding box [x1, y1, x2, y2]
        xyxy2 (list): 두 번째 bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IOU ex)0.387
    """
   
    x1_inter = max(xyxy1[0], xyxy2[0])
    y1_inter = max(xyxy1[1], xyxy2[1])
    x2_inter = min(xyxy1[2], xyxy2[2])
    y2_inter = min(xyxy1[3], xyxy2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    area1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    area2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou
def detect(image):
    """ YOLO 모델로 객체 탐지 """
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)    
    return detections.xyxy.tolist(), detections.confidence.tolist()  # 리스트 변환하여 반환
def NMS_xyxy_with_conf(bbox_list, confidence_list, iou_threshold=0.2):
    # 리스트 변환 (튜플이 되지 않도록)
    bbox_list = list(bbox_list)
    confidence_list = list(confidence_list)

    # Confidence 기준으로 내림차순 정렬
    sorted_indices = sorted(range(len(confidence_list)), key=lambda i: confidence_list[i], reverse=True)
    bbox_list = [bbox_list[i] for i in sorted_indices]
    confidence_list = [confidence_list[i] for i in sorted_indices]

    selected_bboxes = []
    selected_confidences = []

    while bbox_list:
        # 현재 confidence가 가장 높은 bbox 선택
        best_bbox = bbox_list.pop(0)
        best_conf = confidence_list.pop(0)
        
        selected_bboxes.append(best_bbox)
        selected_confidences.append(best_conf)

        # Step 2: 현재 선택된 bbox와 IOU 비교하여 threshold 이상이면 제거
        if bbox_list:  # bbox_list가 비어있지 않을 때만 실행
            filtered_bboxes = []
            filtered_confidences = []
            for bbox, conf in zip(bbox_list, confidence_list):
                if calculate_iou(best_bbox, bbox) < iou_threshold:
                    filtered_bboxes.append(bbox)
                    filtered_confidences.append(conf)

            bbox_list = filtered_bboxes
            confidence_list = filtered_confidences

    return selected_bboxes, selected_confidences
def draw(image, bbox_list):
    """ 탐지된 바운딩 박스를 원본 이미지에 그림 """
    image_copy = image.copy()  # 원본 손상 방지
    for box in bbox_list:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    return image_copy


                        
if __name__ == "__main__":
    
    idx=1
    total_bbox=0
    k=0 #for문 label_folder_list
    count=0 #iou 50%이상
    seven_count=0 #iou 70%이상
    nine_count=0 #iou 90%이상
    start_time = time.time()
    model =YOLO("/mnt/hdd_6tb/bill0914/processed_quarter/runs/detect/train2-4분할+resize+DA2/weights/best.pt")

    input_folder = "/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터"
    label_folder = "/mnt/hdd_6tb/bill0914/processed_original_and_quarter/테스트 데이터 bbox좌표"

    input_folder_list = natsorted(os.listdir(input_folder))
    label_folder_list = natsorted(os.listdir(label_folder))
    for subfolder in input_folder_list: #cr06,cr11,cr12,cr13,cr14
        subfolder_path = os.path.join(input_folder, subfolder)
        subfolder_list = natsorted(os.listdir(subfolder_path)) 
    
        for jpgs in subfolder_list: #01,02,03,04,05
            jpgs_path = os.path.join(subfolder_path, jpgs) #/mnt/hdd_6tb/seungeun/HuNature/Test/01.원천데이터/[cr06]호계사거리/01번
            image_list = natsorted(os.listdir(jpgs_path)) #사진들 리스트

            for i in range(len(image_list)): 
                image_path = os.path.join(jpgs_path, image_list[i])
                test_image = cv2.imread(image_path)
                H, W = test_image.shape[:2]
                xyxy_list = []
                conf_list = []
                
                # 마진 계산
                h_margin = H // 10
                w_margin = W // 10
                offsets = [(0, 0), (W//2 - w_margin, 0), (0, H//2 - h_margin), (W//2 - w_margin, H//2 - h_margin)]
                # 4개로 이미지 분할
                image1 = test_image[0:H//2 + h_margin, 0:W//2 + w_margin]  # 좌상단
                image2 = test_image[0:H//2 + h_margin, W//2 - w_margin:W]  # 우상단
                image3 = test_image[H//2 - h_margin:H, 0:W//2 + w_margin]  # 좌하단
                image4 = test_image[H//2 - h_margin:H, W//2 - w_margin:W]  # 우하단
                for image, (w_offset, h_offset) in zip([image1, image2, image3, image4], offsets):
                    xyxy, conf = detect(image)  # 탐지 수행
                    adjusted_xyxy = [[x1 + w_offset, y1 + h_offset, x2 + w_offset, y2 + h_offset] for x1, y1, x2, y2 in xyxy]
    
                    xyxy_list.extend(adjusted_xyxy)  # 변환된 bbox 추가
                    conf_list.extend(conf)  # confidence 추가
                ori_xyxy,ori_conf = detect(test_image)
                xyxy_list.extend(ori_xyxy)
                conf_list.extend(ori_conf)                
                final_xyxy = NMS_only_xyxy(xyxy_list)
                total_bbox+=len(final_xyxy)
                # 최종 바운딩 박스 그리기
                draw_image = draw(test_image, final_xyxy)
                path = os.path.join("/mnt/hdd_6tb/bill0914/processed_quarter/ori+quar",f"result_{idx}.png")
                cv2.imwrite(path,draw_image)
                idx+=1
            '''         

                with open("/mnt/hdd_6tb/bill0914/processed_quarter/test/total_bbox.txt",'w') as f:            
                    f.write(str(total_bbox))
                with open(os.path.join(label_folder,label_folder_list[k]),'r') as f:
                    k+=1
                    bbox_label_list = f.readlines()
                if len(final_xyxy)!=0:
                    for i in range(len(final_xyxy)): #모델이 탐지한 xyxy
                        for j in range(len(bbox_label_list)): #정답 xyxy
                            xyxy_list = []
                            xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[0]))
                            xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[1]))
                            xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[0])+int(bbox_label_list[j].strip('\n').split(' ')[2]))
                            xyxy_list.append(int(bbox_label_list[j].strip('\n').split(' ')[1])+int(bbox_label_list[j].strip('\n').split(' ')[3]))
                            iou = calculate_iou(final_xyxy[i],xyxy_list)
                            if iou > 0.5 :
                            # print(f"{result[i]} vs {xyxy_list} -> {iou*100}%")
                                count+=1
                            if iou > 0.75 : 
                                seven_count +=1
                            if iou > 0.9 : 
                                nine_count += 1 
                                    
    print(f"------------------------4분할+MARGIN10% inference결과--------------------")
    print(f"모델이 탐지한 bounding box수 : {total_bbox}")
    print(f"iou-threshold가 50%일때 : {count}")
    print(f"iou-threshold가 70%일때 : {seven_count}")
    print(f"iou-threshold가 90%일때 : {nine_count}")
    print(f"라벨 외에 새롭게 탐지한 boudingbox 수 :{total_bbox - count}")
    print(f"이 결과는 processed_quarter/test/에 저장됩니다.")
    '''
    end_time = time.time()
    execution_time = end_time - start_time

    # 메모리 사용량 측정
    memory_usage = get_memory_usage()

        # 결과 출력
    print(f"실행 시간: {execution_time:.4f} 초")
    print(f"최대 메모리 사용량: {memory_usage:.2f} MB")
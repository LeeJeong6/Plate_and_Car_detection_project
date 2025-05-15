import cv2
from ultralytics import YOLO
from typing import List, Dict
import IOU

model = YOLO('/mnt/hdd_6tb/bill0914/YOLOv11n_carcrop.pt')
previous_frame_objects = []
object_id_counter = 0

def detect_objects(results, x_offset=0, y_offset=0, scale_x=3, scale_y=6):
    """ Bounding Box 좌표를 조정하고 크기 변경 """
    frame_result = {'xywh': [], 'confidence': []}
    
    for result in results[0].boxes:
        if result.conf[0].item()>0.4:

            xywh = result.xywh[0].tolist()
            conf = result.conf[0].item()
        
            xywh[0] += x_offset
            xywh[1] += y_offset
            xywh[2] *= scale_x
            xywh[3] *= scale_y
        
            frame_result['xywh'].append(xywh)
            frame_result['confidence'].append(conf)
    
    return frame_result

def process_frame(frame):
    H, W = frame.shape[:2]
    results = model(frame)
    
    for result in results[0].boxes:
        x_center, y_center, width, height = result.xywh[0].tolist()
        if (W//2 - width/2 < x_center < W//2 + width/2) or (H//2 - height/2 < y_center < H//2 + height/2):
            return detect_objects(results)
    
    # 4분할 처리
    quarters = [
        (frame[0:H//2, 0:W//2], 0, 0),
        (frame[0:H//2, W//2:W], W//2, 0),
        (frame[H//2:H, 0:W//2], 0, H//2),
        (frame[H//2:H, W//2:W], W//2, H//2)
    ]
    
    combined_results = {'xywh': [], 'confidence': []}
    for img, x_offset, y_offset in quarters:
        quarter_results = model(img)
        detected = detect_objects(quarter_results, x_offset, y_offset)
        combined_results['xywh'].extend(detected['xywh'])
        combined_results['confidence'].extend(detected['confidence'])
    
    return combined_results

def xywh_to_xyxy(xywh):
    x_center, y_center, width, height = xywh
    return [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2]

def tracking(frame_idx, frame_result):
    global previous_frame_objects, object_id_counter
    current_objects = []
    
    for xywh, conf in zip(frame_result['xywh'], frame_result['confidence']):
        xyxy = xywh_to_xyxy(xywh)
        assigned_id = None
        
        if frame_idx > 0:
            for prev_objs in previous_frame_objects:
                for prev_xyxy, prev_id in prev_objs:
                    if IOU.calculate_iou(xyxy, prev_xyxy) > 0.2:
                        assigned_id = prev_id
                        break
                if assigned_id is not None:
                    break
        
        if assigned_id is None:
            assigned_id = object_id_counter
            object_id_counter += 1
        
        current_objects.append({'id': assigned_id, 'xyxy': xyxy, 'conf': conf})
    
    previous_frame_objects.append([(obj['xyxy'], obj['id']) for obj in current_objects])
    if len(previous_frame_objects) > 7:
        previous_frame_objects.pop(0)
    
    return current_objects

def check_nms(frame_result):
    bbox = [xywh_to_xyxy(xywh) for xywh in frame_result['xywh']]
    confidence = frame_result['confidence']
    
    result_bbox, result_conf = IOU.Dict_NMS(bbox, confidence)
    final_boxes = [[(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)] for x1, y1, x2, y2 in result_bbox]
    
    return {'xywh': final_boxes, 'confidence': result_conf}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    tracked_objects = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_result = process_frame(frame)
        frame_result = check_nms(frame_result)
        tracked_objects.append(tracking(frame_idx, frame_result))
    
    cap.release()
    
    return tracked_objects

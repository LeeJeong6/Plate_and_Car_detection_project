import cv2
import os

def draw_tracking_results(frame_idx, tracked_objects, frame_dir, output_dir, resize_factors=(1/3, 1/6)):
    """
    프레임에 ID, xyxy, confidence 정보를 그려서 저장
    초록색 bbox : 크기를 키운 bbox
    빨간색 bbox : 원래 bbox

    Args:
        frame_idx (int): 현재 프레임 인덱스
        tracked_objects (list): 프레임에 대한 정보 (ID, xyxy, confidence)
        frame_dir (str): 원본 프레임 이미지가 저장된 경로
        output_dir (str): 추적된 객체 정보가 추가된 이미지를 저장할 경로
        resize_factors (tuple): 가로와 세로의 크기 비율 

    Returns:
        None
    """
  
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx + 1}.jpg")
    frame = cv2.imread(frame_path)
    for obj in tracked_objects:
        
        x1, y1, x2, y2 = map(int, obj['xyxy'])  

        # 초록색 bbox (원래 크기) 그리기
        color_original = (0, 255, 0)  # 초록색 (원래 크기)
        thickness = 2
        #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color_original, thickness)

        # 초록색 bbox의 중심 좌표 
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 초록색 bbox의 가로/세로 크기
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # 빨간색 bbox 크기 
        red_bbox_width = int(bbox_width * resize_factors[0])
        red_bbox_height = int(bbox_height * resize_factors[1])

        # 빨간색 bbox 좌표 계산
        red_x1 = center_x - red_bbox_width // 2
        red_y1 = center_y - red_bbox_height // 2
        red_x2 = center_x + red_bbox_width // 2
        red_y2 = center_y + red_bbox_height // 2

        # 빨간색 bbox 그리기
        color_resized = (0, 0, 255)  
        frame = cv2.rectangle(frame, (red_x1, red_y1), (red_x2, red_y2), color_resized, thickness)

        # ID와 confidence 텍스트 표시 (빨간색 bbox 위에)
        label = f"ID: {obj['id']} Conf: {obj['conf']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color_text = (0, 0, 255)  # 빨간색 텍스트
        thickness = 2
        
        # 텍스트로 ID와 confidence 추가 (빨간색 bbox 위)
        cv2.putText(frame, label, (red_x1, red_y1 - 10), font, fontScale, color_text, thickness)
    
    
    output_frame_path = os.path.join(output_dir, f"frame_{frame_idx + 1}_tracked.jpg")
    cv2.imwrite(output_frame_path, frame) 

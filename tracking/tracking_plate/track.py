import IOU
def xywh_to_xyxy(xywh:list) -> list :
    '''
    xywh를 xyxy로 변환하는 함수입니다

    Args:
        xywh (list) : xywh로 저장된 리스트입니다
    Returns:
        xyxy (list) : xyxy로 변환된 리스트입니다    
    '''
    x_center, y_center, width, height = xywh
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    xyxy = [x1, y1, x2, y2] 
    return xyxy

def tracking(detection_results) -> list:
   
    '''
    프레임별 xywh, conf를 기반으로 IOU matching.

    Args:
        detection_results (list): 프레임별 xywh, conf 

    Returns:
        object_ids: matching된 ID와 xyxy좌표
    '''
    
    id = 0
    object_ids = []   #모든 프레임에서 탐지된 정보 다 모아둠

    # 5프레임 전까지의 정보 저장소
    previous_frame_objects = []
 
    for frame_idx, frame_data in enumerate(detection_results): #프레임 단위로 처리하기
        current_frame_objects = []
        for (xywh, conf) in zip(frame_data['xywh'], frame_data['confidence']): #프레임 내 객체 단위로 처리하기
            xyxy = xywh_to_xyxy(xywh)
            
            assigned_id = None

            # 이전 프레임의 객체들과 IOU를 비교
            if frame_idx > 0:
                for prev_objs in previous_frame_objects: # 5프레임 동안의 정보가 담긴 리스트
                    for prev_obj in prev_objs: #프레임 단위로 정보가 튜플로 묶여있음 ([12, 23, 43, 51], 0), ([12, 53, 643, 511], 1)
                        prev_xyxy, prev_id = prev_obj
                        iou = IOU.calculate_iou(xyxy, prev_xyxy)
                        if iou > 0.2: #겹치는 게 있다 -> 같은 객체
                            assigned_id = prev_id 
                            break
                    if assigned_id is not None:
                        break

            if assigned_id is None: #새로 등장했다면 새로운 ID 부여
                assigned_id = id
                id += 1
  
            current_frame_objects.append({'id': assigned_id, 'xyxy': xyxy, 'conf': conf})

        #하나의 프레임 단위 끝->previous에다가 튜플로 저장
        previous_frame_objects.append([(obj['xyxy'], obj['id']) for obj in current_frame_objects])
        
        # 7프레임 이상 유지하지 않도록 제한 -> 너무 많으면 의미X,용량 큼
        if len(previous_frame_objects) > 10:
            previous_frame_objects.pop(0) 

        object_ids.append(current_frame_objects) #각 프레임별로 정보 저장

    return object_ids


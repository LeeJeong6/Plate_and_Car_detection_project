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

def Dict_NMS(bbox_list, confidence_list):
    '''
    bbox_list에 대한 confidence를 같이 처리하기 위한 NMS함수입니다

    Args:
        bbox_list (list) : [[x1,y1,x2,y2],[x3,y3,x4,y4],,,]로 저장된 bbox리스트
        confidence_list (list) : [0.3 , 0.5,,,]와 같이 저장된 confidence리스트
    Returns:
        result_bbox : bbox_list끼리 NMS를 통해 최종 선별된 bbox_list
        result_confidence : 선별된 bbox_list의 confidence리스트
    '''
    holy = []
    for i in range(len(bbox_list)):
        for j in range(len(bbox_list)):
            if j < i:
                continue
            elif 1 > calculate_iou(bbox_list[i], bbox_list[j]) > 0.2:
                holy.append(bbox_list[j])

    unique_list = []
    for item in holy:
        if item not in unique_list:
            unique_list.append(item)

    result_bbox = [x for x in bbox_list if x not in unique_list]
    result_confidence = [confidence_list[i] for i in range(len(bbox_list)) if bbox_list[i] not in unique_list]

    return result_bbox, result_confidence

def List_NMS(bbox_list):
    '''
    bbox_list에 대한 NMS를 계산하는 함수입니다

    Args:
        bbox_list (list) : [[x1,y1,x2,y2],[x3,y3,x4,y4],,,]로 저장된 bbox리스트

    Returns:
        result_bbox : bbox_list끼리 NMS를 통해 최종 선별된 bbox_list
    '''
    holy=[]
    for i in range(len(bbox_list)):
        for j in range(len(bbox_list)):
             if j<i:
                continue
             elif 1>calculate_iou(bbox_list[i],bbox_list[j])>0.2:
                 
                holy.append(bbox_list[j])
    unique_list = []
    for item in holy:
        if item not in unique_list:
            unique_list.append(item)
    result = [x for x in bbox_list if x not in unique_list]
                 
    return result
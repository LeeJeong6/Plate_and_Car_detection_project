import vi_quarter
import result_draw
import video_creator
import argparse
video_path = '/mnt/hdd_6tb/seungeun/HuNature/sample_video/input/cctv50mm.mp4'
frame_dir = '/mnt/hdd_6tb/bill0914/tracking/tracking_plate/original_frame'
result_frame_dir = '/mnt/hdd_6tb/bill0914/tracking/tracking_plate/final_result_frame' 


def result_print(object_ids,result_video_path):
    '''
    프레임별로 탐지한 bbox를 동영상으로 만들어 완성합니다

    Args:
        object_ids (list) : 프레임에 대한 정보 (ID, xyxy, confidence) 

    Returns:
        None
    '''
    for frame_idx, frame_objects in enumerate(object_ids):
        print(f"Frame {frame_idx + 1}:")
        for obj in frame_objects:
            print(f"  ID: {obj['id']}, xyxy: {obj['xyxy']}, confidence: {obj['conf']}")
    for frame_idx, frame_objects in enumerate(object_ids):
   
        # result_draw 모듈의 함수를 호출하여 프레임에 ID, xyxy, confidence 그리기
        result_draw.draw_tracking_results(frame_idx, frame_objects, frame_dir, result_frame_dir)
    video_creator.create_video_from_frames(result_frame_dir,result_video_path)   

if __name__ == '__main__':
    '''
    작은 객체를 탐지하기 위한 Trakcing 알고리즘
    가로를 n배 , 세로를 m배 키워서 프레임 간 객체의 IOU를 높게 키우는 원리
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_video_path", type=str,required=True,help = "original_video_path")
    parser.add_argument("--result_video_path", type=str,required=True,help = "result_video_path")
    args = parser.parse_args()
    object_ids = vi_quarter.process_video(args.ori_video_path) 
    
    result_print(object_ids,args.result_video_path)
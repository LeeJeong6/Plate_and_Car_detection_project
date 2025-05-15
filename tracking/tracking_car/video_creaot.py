import cv2
import os
from natsort import natsorted
def create_video_from_frames(frames_dir, output_video_path, fps=30, codec='XVID'):
    """
    지정된 경로의 이미지들을 합쳐서 동영상을 생성

    Args:
        frames_dir (str): 프레임들이 저장된 경로
        output_video_path (str): 생성될 동영상 파일 경로
        fps (int): 동영상의 초당 프레임 수 (기본값: 30) ->300프레임이면 10초짜리 영상이 만들어짐
        codec (str): 비디오 코덱 (기본값: 'XVID')

    Returns:
        None
    """
    
    frame_files =natsorted(os.listdir(frames_dir))
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)  # 비디오 코덱 설정
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 모든 프레임을 동영상에 추가
    for idx,frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if idx%100==0:
            print(f"동영상 저장중입니다.")
        video_writer.write(frame)  # 프레임을 동영상에 추가

    # 동영상 파일 저장
    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    frames_dir = "/mnt/hdd_6tb/bill0914/tracking/tracking_car/result_frame"
    output_video_path = "/mnt/hdd_6tb/bill0914/tracking/tracking_car/output_video.avi"
    create_video_from_frames(frames_dir, output_video_path, fps=30, codec='XVID')
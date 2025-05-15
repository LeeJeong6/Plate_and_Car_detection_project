import ultralytics
from ultralytics import YOLO

model=YOLO('yolo11n.pt')
model.train(data="/mnt/hdd_6tb/bill0914/mak/dataset.yaml", epochs=100,translate=0.3,flipud=0.3,hsv_h = 0.3)


from ultralytics import YOLO

model=YOLO('yolo11n.pt')
model.train(data="/mnt/hdd_6tb/bill0914/carandplate/data/dataset.yaml", epochs=30)

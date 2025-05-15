import torch
from ultralytics import YOLO
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
model=YOLO('yolo11n.pt')
model.train(data="/mnt/hdd_6tb/bill0914/dataset/data.yaml", epochs=50)

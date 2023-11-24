from ultralytics import YOLO
import torch

pathToData = "/Users/bp512/Desktop/junction-hackathon/project_root/good_alloc_junction_project/dataset/data.yaml"

model = YOLO("yolov8n.pt")

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for training.")

model.train(data=pathToData, epochs=1, device=0)
metrics = model.val()

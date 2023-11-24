from ultralytics import YOLO

pathToBest = "/Users/bp512/Desktop/junction-hackathon/project_root/good_alloc_junction_project/runs/detect/train/weights/best.pt"
model = YOLO(pathToBest)

results = model(source=1, show=True, conf=0.1, save=True)

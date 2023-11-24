from ultralytics import YOLO

pathToBest = "RealOrFake.pt"
model = YOLO(pathToBest)

results = model(source=1, show=True, conf=0.1, save=True)

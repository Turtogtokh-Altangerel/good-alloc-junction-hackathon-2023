import os
from ultralytics import YOLO

pathToBest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RealOrFake.pt")
model = YOLO(pathToBest)


results = model(source=0, show=True, conf=0.3, save=True)

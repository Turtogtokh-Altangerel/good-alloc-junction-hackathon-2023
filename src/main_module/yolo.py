from ultralytics import YOLO

pathToData = "/Users/bp512/Desktop/junction-hackathon/project_root/good_alloc_junction_project/dataset/data.yaml"

model = YOLO("yolov8n.pt")
model.train(data=pathToData, epochs=1)
metrics = model.val()

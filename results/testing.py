from ultralytics import YOLO

model = YOLO("E:\\major\\results\\yolov5\\runs\\detect\\train\\weights\\best.pt")

metrics = model.val(data="E:\major\data\dataset.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6) # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 
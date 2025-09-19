from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='/home/kjonghun0828/mobility training/python/train_crossroad.yaml' , epochs=20)
# model.train(data='/home/kjonghun0828/mobility training/python/train_crossroad.yaml' , epochs=30)
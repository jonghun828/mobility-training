import cv2
from ultralytics import YOLO

model = YOLO('/home/kjonghun0828/mobility training/python/crossroad_best.pt').to('cuda')
video_file_path = "/home/kjonghun0828/mobility training/test_video.mp4"

results = model(video_file_path, save = True)
print(results)



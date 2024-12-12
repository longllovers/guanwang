from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2 

model = YOLO('./ultralytics/yolo11n-pose.pt')
video_path = '/home/ialover/document/ultralytics/video.mp4'
results = model.predict(video_path,save =False,show=True)


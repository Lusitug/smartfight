from ultralytics import YOLO
import cv2
import numpy as np
from exibe_keypoints import detectar_keypoints, pre_process
from keypoints2csv import keypoints2csv
import os 

path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")
model_predict = YOLO(path_yolo)
# model_predict = YOLO('../yolo_pesos/yolo11x-pose.pt')
video_path = r"src\pose\v2.mp4"

cap = cv2.VideoCapture(video_path)

video_keypoints = []

print("Abriu?", cap.isOpened())
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame = pre_process(frame)
    frame, keypoints = detectar_keypoints(frame, model_predict)
    video_keypoints.append(keypoints)
    
    cv2.imshow('Vídeo', frame)

    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

############################################################################

video_keypoints = np.array(video_keypoints, dtype=np.float32)
print("tipo: ", type(video_keypoints))
print("tipo2: ", video_keypoints.dtype)

video_keypoints = np.squeeze(video_keypoints, axis=1)

print("shape: ", video_keypoints.shape)  # (28, 17, 2)
print("Formato: ", video_keypoints) # Formato

keypoints2csv(video_keypoints=video_keypoints)


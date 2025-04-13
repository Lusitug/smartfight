import cv2
from ultralytics import YOLO
from pose.processar_keypoints import detectar_keypoints
from pose.pre_processamento import reajustar_frame

modelo_path = "modelos/yolov8x-pose.pt"
video_path = "dataset/Direto/direto_EyHSY2mj_scale.mp4"  # exemplo
modelo = YOLO(modelo_path)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = reajustar_frame(frame)

    frame, keypoints_lista = detectar_keypoints(frame, modelo)

    if keypoints_lista:
        keypoints = keypoints_lista[0]
        h, w = frame.shape[:2]
        for (x, y) in keypoints:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

    cv2.imshow("Teste detectar_keypoints", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from pose.preprocessamento.pre_processamento import PreProcessamentoVideo
from src.utils.caminhos import Caminhos
import numpy as np

# Instanciando classes
extr = ExtracaoKeypoints(dataset_path="", modelo_yolo_path=Caminhos.path_yolo, saida_csv_path="")
pre = PreProcessamentoVideo()

# Abre o vídeo
cap = cv2.VideoCapture(Caminhos.path_teste1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona frame como no pipeline
    frame = pre.reajustar_frame(frame)

    # Detecta os keypoints com a classe
    frame, keypoints_lista = extr.detectar_keypoints(frame)

    if keypoints_lista:
        h, w = frame.shape[:2]
        keypoints = keypoints_lista[0]  # pegando apenas a pessoa principal

        idx_ref = 11
        # if(len(keypoints) > idx_ref):
        # x_ref, y_ref = keypoints[idx_ref]
        # x_ref, y_ref = int(x_ref * w), int(y_ref * h)

        # dx = w // 2 - x_ref 
        # dy = h // 2 - y_ref
        # centralizados = []

        for (x, y) in keypoints:
            cx, cy = int(x * w ), int(y * h ) 
            # centralizados.append((cx, cy))

            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            
        # cv2.circle(frame , (w//2, h//2), 6, (0, 0, 255), -1)

    cv2.imshow("Visualização Keypoints", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from pose.preprocessamento.pre_processamento import PreProcessamentoVideo
from utils.utilidades import Utilidades

# Instanciando classes
extr = ExtracaoKeypoints(dataset_path="", modelo_yolo_path=Utilidades.path_yolo, saida_csv_path="")
pre = PreProcessamentoVideo()

# Abre o vídeo
cap = cv2.VideoCapture(Utilidades.path_teste1)

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
        for (x, y) in keypoints:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow("Visualização Keypoints", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import cv2
import utilities
import numpy as np
from ultralytics import YOLO
from keypoints2csv import keypoints2csv
# from labels_keypoints import nome_articulacoes
from pre_processamento import pre_process, tipagem_compativel, espremer_estrutura_keypoint

# nome_articulacoes = nome_articulacoes()

def detectar_keypoints(frame, model):
    # height, width, _ = frame.shape
    estimacao = model(frame)

    frame_keypoints = []

    for predict in estimacao:
        if predict.keypoints is not None:
            keypoints = predict.keypoints.xyn.cpu().numpy()
            
            for coordenadas in keypoints:
                lista_keypoints = []

                for i in range(17):
                    x_norm, y_norm = coordenadas[i][0], coordenadas[i][1]
                    #util para visualizar                    # x_coord = int(x_norm * width)  # y_coord = int(y_norm * height)
                    lista_keypoints.append((x_norm, y_norm))
                    #util para visualizar                    # cv2.circle(frame, (x_coord, y_coord), radius=3, color=(0, 0, 255), thickness=-1)  # cv2.putText(frame, nome_articulacoes[i], (x_coord + 5, y_coord - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                frame_keypoints.append(lista_keypoints)
    
    return frame, frame_keypoints


def extrair_keypoints_dataset():
    # path's dos respectivos diretorios/modelo
    dataset = utilities.path_videos2estimate
    saida = utilities.path_keypoints2csv
    modelo_estimacao = YOLO(utilities.path_yolo)

    # classe golpe = nome da classe
    for classe_golpe in os.listdir(dataset): 
        path_pasta_golpe = os.path.join(dataset, classe_golpe) # pasta da classe/golpe
        if not os.path.isdir(path_pasta_golpe):
            continue

        print(f"📁 Classe: {classe_golpe}")
        # pasta de saida dos keypoints2csv
        path_pasta_saida = os.path.join(saida, classe_golpe)
        os.makedirs(path_pasta_saida, exist_ok=True)

        for nome_arquivo in os.listdir(path_pasta_golpe):
            if not nome_arquivo.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            path_videos = os.path.join(path_pasta_golpe, nome_arquivo)
            capt = cv2.VideoCapture(path_videos)

            video_keypoints = []

            print(f"🎥 Processando: {path_videos}")

            print("Abriu?", capt.isOpened())
            while capt.isOpened():
                _, frame = capt.read()

                if not _:
                    break
                
                frame = pre_process(frame)
                _, keypoints = detectar_keypoints(frame, modelo_estimacao)
                
                if keypoints:
                    video_keypoints.append(keypoints[0]) # somente em videos com 1 pessoa

            capt.release()

            if not video_keypoints:
                print(f"⚠️ Sem keypoints: {nome_arquivo}")

            video_keypoints = tipagem_compativel(video_keypoints)
            video_keypoints = espremer_estrutura_keypoint(video_keypoints)
            print("shape: ", video_keypoints.shape)  # (28, 17, 2) ? !

            nome_csv = os.path.splitext(nome_arquivo)[0] + ".csv"
            caminho_csv = os.path.join(path_pasta_saida, nome_csv)

            keypoints2csv(video_keypoints=video_keypoints, path_saida=caminho_csv)


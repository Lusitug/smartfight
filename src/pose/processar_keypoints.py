import os
import cv2
import utils.utilidades as utilidades
import numpy as np
from ultralytics import YOLO
from pose.keypoints2csv import keypoints2csv
# from labels_keypoints import nome_articulacoes
from pose.pre_processamento import pre_process, tipagem_compativel, espremer_estrutura_keypoint

# nome_articulacoes = nome_articulacoes()

def detectar_keypoints(frame, model):
    # height, width, _ = frame.shape
    estimacao = model(frame)

    frame_keypoints = []

    for predict in estimacao:
        if predict.keypoints is not None:
            keypoints = predict.keypoints.xyn.cpu().numpy()
            boxes = predict.boxes.xyxy.cpu().numpy()
            # evitar detecção de sombras
            print("KPS: ", len(keypoints))
    
            if len(keypoints) == 0: 
                return frame, []
    
            if len(keypoints) == 1: 
                coordenadas = keypoints[0]
            else:
                areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
                maior_idx = np.argmax(areas)
                coordenadas = keypoints[maior_idx]
                
            # for coordenadas in keypoints:
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
    dataset = utilidades.path_videos2estimate
    saida = utilidades.path_keypoints2csv
    modelo_estimacao = YOLO(utilidades.path_yolo)

    # classe golpe = nome da classe
    for classe_golpe in os.listdir(dataset): 
        path_pasta_golpe = os.path.join(dataset, classe_golpe) # pasta da classe/golpe
        if not os.path.isdir(path_pasta_golpe):
            continue

        print(f"\n📁 Classe: {classe_golpe}")
        # pasta de saida dos keypoints2csv
        path_pasta_saida = os.path.join(saida, classe_golpe)
        os.makedirs(path_pasta_saida, exist_ok=True)

        for nome_video in os.listdir(path_pasta_golpe):
            if not nome_video.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            path_videos = os.path.join(path_pasta_golpe, nome_video)
            capt = cv2.VideoCapture(path_videos)

            video_keypoints = []

            print(f"\n🎥 Processando: {path_videos}")

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
                print(f"⚠️ Sem keypoints: {nome_video}")

            video_keypoints = tipagem_compativel(video_keypoints)
            video_keypoints = espremer_estrutura_keypoint(video_keypoints)
            print("\nshape: ", video_keypoints.shape)  # (28, 17, 2) ? !

            nome_csv_saida = os.path.splitext(nome_video)[0] + ".csv"
            caminho_csv_saida = os.path.join(path_pasta_saida, nome_csv_saida)

            keypoints2csv(video_keypoints=video_keypoints, path_saida=caminho_csv_saida)


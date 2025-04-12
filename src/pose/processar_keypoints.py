import os
import cv2
import utils.utilidades as utilidades
import numpy as np
from ultralytics import YOLO
from pose.keypoints2csv import converter_keypoints_csv
# from labels_keypoints import nome_articulacoes
from pose.pre_processamento import reajustar_frame, converter_float32, espremer_estrutura_keypoint
from time import time
# nome_articulacoes = nome_articulacoes()

def detectar_keypoints(frame, model):
    # height, width, _ = frame.shape
    estimacao = model(frame)

    frame_keypoints = []

    for predict in estimacao:
        if predict.keypoints is not None:
            keypoints = predict.keypoints.xyn.cpu().numpy()
            print("[PESSOAS DETECTADAS: ", len(keypoints),"]")
            # evitar detecção de sombras
            boxes = predict.boxes.xyxy.cpu().numpy()
    
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
    tempo_inicial = time()
    # path's dos respectivos diretorios/modelo
    dataset = utilidades.path_videos2estimate
    saida = utilidades.path_keypoints2csv
    modelo_estimacao = YOLO(utilidades.path_yolo)

    # classe golpe = nome da classe
    for classe_golpe in os.listdir(dataset): 
        path_pasta_golpe = os.path.join(dataset, classe_golpe) # pasta da classe/golpe
        if not os.path.isdir(path_pasta_golpe):
            continue

        print(f"\n📁 [GOLPE ANALISADO: {classe_golpe}]")
        # pasta de saida dos keypoints2csv
        path_pasta_saida = os.path.join(saida, classe_golpe)
        os.makedirs(path_pasta_saida, exist_ok=True)

        for nome_video in os.listdir(path_pasta_golpe):
            if not nome_video.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            path_videos = os.path.join(path_pasta_golpe, nome_video)
            capt = cv2.VideoCapture(path_videos)

            video_keypoints = []

            print(f"\n🎥 [PROCESSANDO: {path_videos}]")

            print("\nO VIDEO ABRIU? ", capt.isOpened())
            while capt.isOpened():
                _, frame = capt.read()

                if not _:
                    break
                
                frame = reajustar_frame(frame)
                _, keypoints = detectar_keypoints(frame, modelo_estimacao)
                
                if isinstance(keypoints, list) and len(keypoints) > 0:
                    video_keypoints.append(keypoints[0]) # somente em videos com 1 pessoa
                else:
                    print(f"\n⚠️ [NENHUMA PESSOA DETECTADA NO FRAME: {nome_video}]")


            capt.release()

            if not video_keypoints:
                print(f"\n⚠️ [NENHUM KEYPOINT: {nome_video}]")

            video_keypoints = converter_float32(video_keypoints)
            video_keypoints = espremer_estrutura_keypoint(video_keypoints)
            print("\n[SHAPE: ", video_keypoints.shape,"]")  # (28, 17, 2) ? !

            nome_csv_saida = os.path.splitext(nome_video)[0] + ".csv" 
            caminho_csv_saida = os.path.join(path_pasta_saida, nome_csv_saida)

            converter_keypoints_csv(video_keypoints=video_keypoints, path_saida=caminho_csv_saida)
            tempo_final = time()
            print(f"\n✅ [SALVO: {caminho_csv_saida}]")
            print(f"\n\t⏰ [DURAÇÃO DE EXTRAÇÃO DE KEYPOINTS: {tempo_final - tempo_inicial:.2f}]")

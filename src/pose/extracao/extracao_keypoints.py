import os
import cv2
import numpy as np
from time import time
from ultralytics import YOLO
from typing import List, Tuple, Optional
from pose.preprocessamento.pre_processamento import PreProcessamentoVideo
from pose.preprocessamento.transformar_keypoints import TransformarKeypoints
from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV

class ExtracaoKeypoints:
    def __init__(self, modelo_yolo_path: str, dataset_path: str, saida_csv_path: str):
        self.modelo_yolo_path = YOLO(modelo_yolo_path)
        self.dataset_path = dataset_path
        self.saida_csv_path = saida_csv_path
        self.pre_processador = PreProcessamentoVideo()
        self.transformador_dados = TransformarKeypoints()
        self.conversor_keypoint_csv = ConverterKeypointsCSV()

    def detectar_keypoints(self, frame) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        estimacao = self.modelo_yolo_path(frame)
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
                lista_keypoints = [(coordenadas[i][0], coordenadas[i][1]) for i in range(17)]
                frame_keypoints.append(lista_keypoints)
        
        return frame, frame_keypoints

    def processar_video(self, path_videos: str) -> Optional[np.ndarray]:
        video_keypoints = []
       
        print(f"\n🎥 [PROCESSANDO: {path_videos}]")
       
        capt = cv2.VideoCapture(path_videos)
        print("\nO VIDEO ABRIU? ", capt.isOpened())

        while capt.isOpened():
            ret, frame = capt.read()

            if not ret:
                break
                
            frame = self.pre_processador.reajustar_frame(frame)
            _, keypoints = self.detectar_keypoints(frame=frame)
                
            if isinstance(keypoints, list) and len(keypoints) > 0:
                video_keypoints.append(keypoints[0]) # somente em videos com 1 pessoa
            else:
                print(f"\n⚠️ [NENHUMA PESSOA DETECTADA NO FRAME: {path_videos}]")


        capt.release()

        if not video_keypoints:
            print(f"\n⚠️ [NENHUM KEYPOINT: {path_videos}]")

        video_keypoints = self.transformador_dados.converter_float32(video_keypoints)
        video_keypoints = self.transformador_dados.espremer_estrutura_keypoint(video_keypoints)

        print("\n[SHAPE: ", video_keypoints.shape,"]")  # (28, 17, 2) 
      
        return video_keypoints
        
    def extrair_keypoints_dataset(self):
        tempo_inicial = time()
        # nome da classe
        for classe_golpe in os.listdir(self.dataset_path): 
            path_pasta_golpe = os.path.join(self.dataset_path, classe_golpe)
            
            if not os.path.isdir(path_pasta_golpe):
                continue
            # pasta de saida kpt2csv
            print(f"\n📁 [GOLPE ANALISADO: {classe_golpe}]")
            path_pasta_saida = os.path.join(self.saida_csv_path, classe_golpe)
            os.makedirs(path_pasta_saida, exist_ok=True)

            for nome_video in os.listdir(path_pasta_golpe):
                if not nome_video.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue

                path_videos = os.path.join(path_pasta_golpe, nome_video)
                lista_keypoints = []
                lista_keypoints = self.processar_video(path_videos=path_videos)

                if lista_keypoints is None:
                    continue

                nome_csv_saida = os.path.splitext(nome_video)[0] + ".csv" 
                caminho_csv_saida = os.path.join(path_pasta_saida, nome_csv_saida)

                self.conversor_keypoint_csv.keypoints2csv(
                    lista_keypoints_video=lista_keypoints,
                    path_saida=caminho_csv_saida)
                
                print(f"\n✅ [SALVO: {caminho_csv_saida}]")

        tempo_final = time()
        print(f"\n\t⏰ [DURAÇÃO DE EXTRAÇÃO DE KEYPOINTS: {tempo_final - tempo_inicial:.2f}]")

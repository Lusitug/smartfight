import os
import cv2
import numpy as np
from time import time
from utils.utilidades import Utilidades
from utils.augmentations_itens import AugmentationItens

class DataAugumentation:
    def __init__(self, dataset_path: str = Utilidades.path_videos2estimate):
        self.dataset_path = dataset_path

    def aplicar_augumentation_dataset(self):
        print("[DEBUG] path_videos2estimate:", self.dataset_path) 
        tempo_inicial = time()

        for classe_golpe in os.listdir(self.dataset_path):
            path_pasta_golpe = os.path.join(self.dataset_path, classe_golpe)

            if not os.path.isdir(path_pasta_golpe):
                continue

            print(f"\n📁 [GOLPE ANALISADO: {classe_golpe}]")

            for nome_video in os.listdir(path_pasta_golpe):
                if not nome_video.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue

                path_videos = os.path.join(path_pasta_golpe, nome_video)
                nome_base_video = os.path.splitext(nome_video)[0]

                capt = cv2.VideoCapture(path_videos)
                print("\nO VIDEO ABRIU? ", capt.isOpened())

                width_video = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
                heigth_video = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = capt.get(cv2.CAP_PROP_FPS)
                frames_total = int(capt.get(cv2.CAP_PROP_FRAME_COUNT))

                print(f"\n🎬 [PROCESSANDO: {nome_video} ({frames_total} FRAMES])")

                writers = {}
                for nome_tecnica_augm, _ in AugmentationItens.AUGMENTATIONS.items():
                    nome_video_saida = f"{nome_base_video}_{nome_tecnica_augm}.mp4"
                    path_saida_videos = os.path.join(path_pasta_golpe, nome_video_saida)

                    writers[nome_tecnica_augm] = cv2.VideoWriter(
                        path_saida_videos,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (width_video, heigth_video)
                    )

                while True:
                    _, frame = capt.read()
                    if not _:
                        break

                    for nome_tecnica_augm, funcao_augm in AugmentationItens.AUGMENTATIONS.items():    
                        frame_augm = funcao_augm(frame.copy())
                        writers[nome_tecnica_augm].write(frame_augm)

                capt.release()
                for nome_tecnica_augm, writer in writers.items():
                    writer.release()
                    print(f"\n✅ [SALVO: {os.path.join(path_pasta_golpe, f'{nome_base_video}_{nome_tecnica_augm}.mp4')}]")
        
        tempo_final = time()
        print(f"\n\t⏰ [DURAÇÃO DE PRE PROCESSAMENTO: {tempo_final - tempo_inicial:.2f}]")

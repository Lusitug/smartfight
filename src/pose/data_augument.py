from utils.utilidades import path_videos2estimate
import os
from utils.augmentations_itens import AUGMENTATIONS
import cv2
from time import time

def aplicar_data_augmentation_dataset(dataset_saida=path_videos2estimate):
    print("[DEBUG] path_videos2estimate:", dataset_saida) 
    tempo_inicial = time()
    for classe_golpe in os.listdir(dataset_saida):
        path_pasta_golpe = os.path.join(dataset_saida, classe_golpe)

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

            print(f"\n🎬 [PROCESSANDO: {nome_video} ({int(capt.get(cv2.CAP_PROP_FRAME_COUNT))} FRAMES])")
            
            writers = {}
            for nome_tecnica_augm, _ in AUGMENTATIONS.items():
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
                
                for nome_tecnica_augm, funcao_augm in AUGMENTATIONS.items():    
                    frame_augm = funcao_augm(frame.copy())
                    writers[nome_tecnica_augm].write(frame_augm)
                
            capt.release()
            for nome_tecnica_augm, writer in writers.items():
                writer.release()
                print(f"\n✅ [SALVO: {os.path.join(path_pasta_golpe, f'{nome_base_video}_{nome_tecnica_augm}.mp4')}]")
    
    
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO DE PRE PROCESSAMENTO: {tempo_final - tempo_inicial:.2f}]")
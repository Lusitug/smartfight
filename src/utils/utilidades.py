import os
import pandas as pd

class Utilidades:
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'
    path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

    path_keypoints2csv =  os.path.join("src", "saidas", "keypoints_extraidos")

    path_videos2estimate = os.path.join("src", "dataset")

    path_modelo_treinado = os.path.join("src", "saidas", "modelos_ml")

    path_teste1 = os.path.join("src", "soco", "soco.mp4")
    path_teste2 = os.path.join("src", "soco", "soco.csv")
    path_teste0 = os.path.join("src", "soco")

    @staticmethod
    def gerar_init(caminho_pasta):
        init_path = os.path.join(caminho_pasta, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass

    @staticmethod
    def remover_coluna_frame(df):
        if 'frame' in df.columns:
            df = df.drop(columns=['frame'])
        return df
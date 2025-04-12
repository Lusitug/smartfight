import pandas as pd
from utils.labels_keypoints import nome_articulacoes
from time import time
import os

class ConverterKeypointsCSV:
    def __init__(self):
        self.colunas = nome_articulacoes()
       
    def keypoints2csv(self, lista_keypoints_video, path_saida: str):
        tempo_inicial = time()

        linhas = []

        for frame in lista_keypoints_video:  # (17, 2)
            linha = []
            for keypoint in frame:
                x, y = keypoint
                linha.append(f"({x:.4f},{y:.4f})") # formata como string "(x, y)"
            linhas.append(linha)

        pasta_saida = os.path.dirname(path_saida)
        os.makedirs(pasta_saida, exist_ok=True)
        
        df = pd.DataFrame(linhas, columns=self.colunas)
        
        df.to_csv(path_saida, index_label="frame")

        tempo_final = time()
        print(f"\n✅ [SALVO: {path_saida}]")
        print(f"\n\t⏰ [DURAÇÃO DE CONVERSÃO KPT->CSV: {tempo_final - tempo_inicial:.2f}]")

import os
import pandas as pd
from utils.labels_keypoints import nome_articulacoes
from time import time
def converter_keypoints_csv(video_keypoints, path_saida):
    tempo_inicial = time()
    # recebe o nome das colunas
    colunas = nome_articulacoes()
    # armazenar as linhas do CSV
    linhas = []

    for frame in video_keypoints:  # (17, 2)
        linha = []
        for keypoint in frame:
            x, y = keypoint
            linha.append(f"({x:.4f},{y:.4f})") # formata como string "(x, y)"
        linhas.append(linha)

    df = pd.DataFrame(linhas, columns=colunas)
    
    os.makedirs(os.path.dirname(path_saida), exist_ok=True)

    df.to_csv(path_saida, index_label="frame")

    tempo_final = time()
    print(f"\n✅ [SALVO: {path_saida}]")
    print(f"\n\t⏰ [DURAÇÃO DE CONVERSÃO KPT->CSV: {tempo_final - tempo_inicial:.2f}]")

import os
import pandas as pd
from utils.labels_keypoints import nome_articulacoes

def keypoints2csv(video_keypoints, path_saida):
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

    print(f"\n✅ [SALVO: {path_saida}]")
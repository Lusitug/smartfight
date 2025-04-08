import os
import pandas as pd
from labels_keypoints import nomes_juntas

# converte um unico arquivo # necessario passar caminho de cada arquivo a ser convertido
def keypoints2csv(video_keypoints, path_saida):
    # Gera as colunas do DataFrame
    colunas = nomes_juntas()
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

    print(f"✅ CSV salvo em: {path_saida}")
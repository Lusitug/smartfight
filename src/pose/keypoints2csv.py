import pandas as pd
from labels_keypoints import nomes_juntas
import os



# converte um unico arquivo # necessario passar caminho de cada arquivo a ser convertido
def keypoints2csv(video_keypoints):
    # Gera as colunas do DataFrame
    colunas = nomes_juntas()

    # armazenar as linhas do CSV
    linhas = []

    for frame in video_keypoints:  # (17, 2)
        linha = []
        for keypoint in frame:
            x, y = keypoint
            linha.append(f"({x:.4f},{y:.4f})")  # formata como string "(x, y)"
        linhas.append(linha)

    # df
    df = pd.DataFrame(linhas, columns=colunas)
    
    path_csv =  os.path.join("src", "keypoints_extraidos", "video_keypoints_teste.csv")   

    # 2csv
    df.to_csv(path_csv, index_label="frame")

    print("✅ CSV salvo como 'video_keypoints_teste.csv'")





# converte multiplos arquivo # necessario passar caminho da pasta a ser convertida
def multi_keypoints2csv():
    pass
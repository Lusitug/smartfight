import os
import ast  # transformar string em tupla
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.pose.utilities import path_keypoints2csv

path_csv =  os.path.join(path_keypoints2csv, "Direto", "direto_T9xF3T7V.csv")   

df = pd.read_csv(path_csv)
df.columns = df.columns.str.strip() # remove caracteres indesejados

# nomes dos keypoints exceto (0,0) = frame
labels_keypoints = [col for col in df.columns if col != "frame"]
print(labels_keypoints)

# strings "(x, y)" para tuplas float (x, y)
for col in labels_keypoints:
    df[col] = df[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else (0.0, 0.0))

# transforma odf em uma lista de dicionários por frame
dados_frames = []
for _, linha in df.iterrows():
    frame = []
    for keypoint in labels_keypoints:
        x, y = linha[keypoint]
        frame.append((x, y))
    dados_frames.append(frame)

# animaçao
fig, ax = plt.subplots(figsize=(6, 8))
scat = ax.scatter([], [], c='blue')
textos = []

def inicializar():
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # inverte o eixo Y
    ax.set_title("Golpe executado")
    ax.grid(True)
    return scat,

def atualizar(frame):
    global textos

    # remove textos anteriores
    for txt in textos:
        txt.remove()
    textos = []

    # atualiza pontos
    xs = [pt[0] for pt in frame]
    ys = [pt[1] for pt in frame]
    scat.set_offsets(list(zip(xs, ys)))

    # add laels
    for i, (x, y) in enumerate(frame):
        textos.append(ax.text(x + 0.01, y, labels_keypoints[i], fontsize=8))

    return scat, *textos

ani = FuncAnimation(fig, atualizar, frames=dados_frames, init_func=inicializar,
                    interval=300, blit=True, repeat=True)

plt.show()

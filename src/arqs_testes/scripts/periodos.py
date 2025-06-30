import numpy as np
import pandas as pd
from utils.caminhos import Caminhos
import ast
import matplotlib.pyplot as plt

# GRAFICOS - KEYPOINTS NO PLANOI CARTESIANO

df = pd.read_csv(Caminhos.teste_periodiciodade10)

"""
"""
# s4
articulacoes = ["l_h","l_k", "l_a"]  # punho direito e esquerdo
n_points = 50
indices = np.linspace(0, len(df['frame']) - 1, n_points, dtype=int)
frames = df['frame'].iloc[indices]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for articulacao in articulacoes:
    coordenadas = df[articulacao].apply(ast.literal_eval)
    x = coordenadas.apply(lambda p: p[0])
    x_sub = x.iloc[indices]
    plt.plot(frames, x_sub, label=f'{articulacao} - X')
plt.xlabel("Frame")
plt.ylabel("Coord. X")
plt.title("Movimento Lateral (X)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
for articulacao in articulacoes:
    coordenadas = df[articulacao].apply(ast.literal_eval)
    y = coordenadas.apply(lambda p: p[1])
    y_sub = y.iloc[indices]
    plt.plot(frames, y_sub, label=f'{articulacao} - Y')
plt.xlabel("Frame")
plt.ylabel("Coord. Y")
plt.title("Movimento Vertical (Y)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




"""
# s3 - DE MAIOR INTERESSE
# Definir articulação de interesse (ex: punho esquerdo)
articulacao = "l_a"

# Extrair coordenadas
coordenadas = df[articulacao].apply(ast.literal_eval)
x_coords = coordenadas.apply(lambda p: p[0])
y_coords = coordenadas.apply(lambda p: p[1])

# Amostragem (para não plotar todos os frames, só 50)
n_points = 50
indices = np.linspace(0, len(df['frame']) - 1, n_points, dtype=int)
frames = df['frame'].iloc[indices]
x_sub = x_coords.iloc[indices]
y_sub = y_coords.iloc[indices]

# Plotar gráficos
plt.figure(figsize=(15, 4))

# Subplot 1: Movimento Lateral
plt.subplot(1, 3, 1)
plt.plot(frames, x_sub, color='blue', label='Coord. X')
plt.xlabel("Frame")
plt.ylabel("Coord. X")
plt.title("Movimento Lateral")
plt.grid(True)
plt.legend()

# Subplot 2: Movimento Vertical
plt.subplot(1, 3, 2)
plt.plot(frames, y_sub, color='green', label='Coord. Y')
plt.xlabel("Frame")
plt.ylabel("Coord. Y")
plt.title("Movimento Vertical")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
"""


"""
 # s0 - DE MAIOR INTERESSE
articulacao = "r_w"

coordenadas = df[articulacao].apply(ast.literal_eval)

x = coordenadas.apply(lambda p: p[0])
y = coordenadas.apply(lambda p: p[1])

n_points = 50
indices = np.linspace(0, len(df['frame']) - 1, n_points, dtype=int)
frames = df['frame'].iloc[indices]
x_sub = x.iloc[indices]
y_sub = y.iloc[indices]

plt.figure(figsize=(10, 5))
plt.plot(frames, y_sub, label=f'{articulacao} - Y (altura)', color='blue')
plt.plot(frames, x_sub, label=f'{articulacao} - X (largura)', color='green', linestyle='--')
plt.xlabel("Frame")
plt.ylabel("Coordenada Normalizada")
plt.title(f"Variação da articulação '{articulacao}' ao longo do tempo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
















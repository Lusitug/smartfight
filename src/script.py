from re import X
from utils.globais import Globais
from utils.caminhos import Caminhos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from dtw import dtw, dtwPlot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dtaidistance import dtw as dtw2
import os
from typing import List
from analise.visualizacao_movimento import VisualizarMovimentoArticulacao
from analise.analisar_periodos import AnalisarCiclosDataset
from analise.dtw_analise import AnalisarSequenciasDTW
# df = pd.read_csv(Caminhos.teste_periodiciodade7)
# articulacao_l_w = np.array([ast_func(point) for point in df["l_w"].values])
# # df = converter_array32(df) # soco do dataset
# articulacao_l_w_1d = np.linalg.norm(articulacao_l_w, axis=1) if articulacao_l_w.ndim > 1 else articulacao_l_w

# articulacao = np.array([Globais.ast_func(point) for point in df["l_w"].values])
# articulacao_1d = np.linalg.norm(articulacao, axis=1) if articulacao.ndim > 1 else articulacao

# visualização
df = pd.read_csv(Caminhos.teste_periodiciodade7)
# visu = VisualizarMovimentoArticulacao(golpe_csv=df)
# visu.plotar_movimento01(["l_s","l_el","l_w"])

# fft
# df = Globais.converter_array32(df)
# analise = AnalisarCiclosDataset(golpe_csv=df)
# feq =  analise.verificar_periodo(idx_ponto=9)
# # obtendo frequencia
# print(f" {feq['period_frames']:.2f} " )

# dtw
df2 = pd.read_csv(Caminhos.teste_periodiciodade15)
analise2 = AnalisarSequenciasDTW(series1=df, series2=df2, articulacao="l_w")
distance_plot = analise2.calcular_distancia_dtw_lib(tipo_distancia="squared_euclidean_distance")
print(distance_plot.distance)
print(distance_plot.normalizedDistance)
final = analise2.plotar_dtw_lib(alinhamento=distance_plot)
# print(final)
distance_plot2, similaridad, paths = analise2.calcular_distancia_dtaidistance_lib()
print(distance_plot2)
print(similaridad)
analise2.plotar_dtaidistance_lib(melhor_caminho=paths)


# dtw-knn (testar outros ml)





# print(df)










"""  



def ast_func(points):
    return ast.literal_eval(points)

def converter_frame_vetor(frame: pd.Series) -> np.ndarray:
    vetor = []
    for coluna in frame.index:
        if coluna == "frame":
            continue
        valor = frame[coluna]

        if isinstance(valor, str):
            try:
                ponto = ast.literal_eval(valor)
                if isinstance(ponto, (list, tuple)) and len(ponto) == 2:
                    vetor.extend(ponto)
                else:
                    # Caso seja uma lista malformada
                    vetor.extend([0.0, 0.0])
            except Exception:
                vetor.extend([0.0, 0.0])
        else:
            vetor.extend([0.0, 0.0])

    return np.array(vetor, dtype=np.float32)

def converter_array32(df: pd.DataFrame) -> np.ndarray:
    vetores = df.apply(converter_frame_vetor, axis=1).values.tolist()
    return np.array(vetores, dtype=np.float32)


# distance metric between diferent signals
# dtw1 - video longo x video curto
# dtw2 - video curto x video curto

# df = pd.read_csv(Caminhos.teste_periodiciodade10)
df = pd.read_csv(Caminhos.teste_periodiciodade7)
articulação_l_w = np.array([ast_func(point) for point in df["l_w"].values])
# df = converter_array32(df) # soco do dataset

df2 = pd.read_csv(Caminhos.teste_periodiciodade12)
articulação_r_w = np.array([ast_func(point) for point in df2["l_w"].values])
# df2 = converter_array32(df2) # soco curto

articulação_l_w_1d = np.linalg.norm(articulação_l_w, axis=1) if articulação_l_w.ndim > 1 else articulação_l_w
articulação_r_w_1d = np.linalg.norm(articulação_r_w, axis=1) if articulação_r_w.ndim > 1 else articulação_r_w

def squared_euclidean_distance(x, y):  ## melhor resultado, menor distancia
    return np.sum((x - y) ** 2)

def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_distance(x, y): # nao funcionou
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def canberra_distance(x, y):
    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-10))


# Compute DTW
alignment_plot  = dtw(articulação_l_w_1d, articulação_r_w_1d, 
                keep_internals=True,  # Keep matrices for visualization
                distance_only=False,
                dist_method=squared_euclidean_distance)

# Access results
print("Distance: dtw lib: ", alignment_plot.distance)
# Distance: 287.29683062447003
print("Normalized distance: dtw lib: ", alignment_plot.normalizedDistance)
# Normalized distance: 0.15462692713911197 / ~15.5% de diferença

# show 3
# 1. mostra a o grafico das articulações de interesse em analise
fig, ax1 = plt.subplots(figsize=(16, 10)) 

ax1.plot(articulação_l_w_1d, label='video dataset', color='blue')
ax1.plot(articulação_r_w_1d, label='video analise', linestyle='--', color='black')

ax1.set_title('Original Time Series')
ax1.legend()
plt.xticks(np.arange(0, len(articulação_l_w_1d), 60),  rotation=45, ha="right")
plt.show()


# 2. 4 vnjm/g45-
distance, paths = dtw2.warping_paths(articulação_l_w_1d, articulação_r_w_1d, use_c=False)
best_path = dtw2.best_path(paths)
similarity_score = distance / len(best_path)

print("distance - dtai lib analise: ",distance)
# print(paths)
# print(best_path)
print("similarity_score - dtai lib analise:",similarity_score)

plt.figure(figsize=(12, 8))

# Original Time Series Plot
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.plot(articulação_l_w_1d, label='video dataset', color='green')
ax1.plot(articulação_r_w_1d, label='video analise', linestyle='--',color='black')
ax1.set_title('Original Time Series')
ax1.legend()

# Shortest Path Plot (Cost Matrix with the path)
# In this example, only the path is plotted, not the entire cost matrix.

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.plot(np.array(best_path)[:, 0], np.array(best_path)[:, 1], 'green', marker='o', linestyle='-')
ax2.set_title('Shortest Path (Best Path)')
ax2.set_xlabel('video dataset frames')
ax2.set_ylabel('video analise frames')
ax2.grid(True)

# Point-to-Point Comparison Plot
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3.plot(articulação_l_w_1d, label='video dataset', color='green', marker='o')
ax3.plot(articulação_r_w_1d, label='video analise', color='black', marker='x', linestyle='--')

for a, b in best_path:
    ax3.plot([a, b], [articulação_l_w_1d[a], articulação_r_w_1d[b]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)
ax3.set_title('Point-to-Point Comparison After DTW Alignment')
ax3.legend()

plt.tight_layout()
plt.show()

# Create a DataFrame to display the similarity score and correlation coefficient
results_df = pd.DataFrame({
    'Metric': ['DTW Similarity Score'],
    'Value': [similarity_score]
})

# Add descriptions for the results
results_df['Description'] = [
    "Lower scores indicate greater similarity between the time series."
]

results_df

print(results_df)



"""
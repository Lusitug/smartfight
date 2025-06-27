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
from ml.classificacao_dtw_knn import ClassificadorDTW_KNN


####################################

# df = pd.read_csv(Caminhos.teste_periodiciodade7)
# articulacao_l_w = np.array([ast_func(point) for point in df["l_w"].values])
# # df = converter_array32(df) # soco do dataset
# articulacao_l_w_1d = np.linalg.norm(articulacao_l_w, axis=1) if articulacao_l_w.ndim > 1 else articulacao_l_w

# articulacao = np.array([Globais.ast_func(point) for point in df["l_w"].values])
# articulacao_1d = np.linalg.norm(articulacao, axis=1) if articulacao.ndim > 1 else articulacao

####################################

df = pd.read_csv(Caminhos.teste_periodiciodade_loop2) # direto lk
df2 = pd.read_csv(Caminhos.teste_periodiciodade_loop3)


# visualização # eixo x e eixo y em relação ao tempo/franes

visu = VisualizarMovimentoArticulacao(golpe_csv=df)
visu.plotar_movimento01(["l_a"])

######################################

# # fft
# # df = Globais.converter_array32(df)
analise = AnalisarCiclosDataset(golpe_csv=df)
feq =  analise.verificar_periodo(idx_ponto=9) # obtem valores dos ciclos
print(f" frequencia:  {feq['period_frames']:.2f} " )

analise1 = AnalisarCiclosDataset(golpe_csv=df2)
feq2 =  analise1.verificar_periodo(idx_ponto=9) # obtem valores dos ciclos
print(f" frequencia:  {feq2['period_frames']:.2f} " )
# print(feq)

# ###################################

# comparaão dtw
analise2 = AnalisarSequenciasDTW(series1=df, series2=df2, articulacao="l_a") 

distance_plot = analise2.calcular_distancia_dtw_lib(tipo_distancia="squared_euclidean_distance") # manhattan_distance
print("Distancia (DTW): ", distance_plot["distance"])
print("Distancia Normalizada (DTW): ", distance_plot["normalized_distance"])
analise2.plotar_dtw_lib(alinhamento=distance_plot["alinhamento"])

# distance_plot2, similaridad, paths = analise2.calcular_distancia_dtaidistance_lib()
# print(distance_plot2)
# print(similaridad)
# analise2.plotar_dtaidistance_lib(melhor_caminho=paths)


###

def calcular_similaridade(alinhamento, period_frames1, period_frames2, tolerancia_freq=0.05):
    sim_dtw = 1 - min(alinhamento.normalizedDistance, 1)
    freq1 = 1 / period_frames1
    freq2 = 1 / period_frames2

    freq_match = 1 if abs(freq1 - freq2) <= tolerancia_freq else 0

    score = sim_dtw * 0.8 + freq_match * 0.2

    print(f"Similaridade DTW normalizada: {sim_dtw:.3f}")
    print(f"Frequência Vídeo 1: {freq1:.3f}, Vídeo 2: {freq2:.3f}, match: {freq_match}")
    print(f"Score combinado: {score:.3f}")

    return score
def interpretar_similaridade(score):
    if score >= 1.0:
        return "Movimentos idênticos (ou extremamente semelhantes)"
    elif score >= 0.8:
        return "Movimentos muito semelhantes, com variações pequenas"
    elif score >= 0.5:
        return "Similaridade média, com alguns padrões parecidos"
    elif score >= 0.2:
        return "Baixa similaridade"
    else:
        return "Movimentos bem diferentes"

score = calcular_similaridade(distance_plot['alinhamento'], feq['period_frames'], feq2['period_frames'])
descricao = interpretar_similaridade(score)
print(f"Similaridade final: {score:.3f} → {descricao}")
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

df = pd.read_csv(Caminhos.teste_peridiocidade_cloop1) 
df2 = pd.read_csv(Caminhos.teste_peridiocidade_sloop)


# visualização # eixo x e eixo y em relação ao tempo/franes

visu = VisualizarMovimentoArticulacao(golpe_csv=df)
visu.plotar_movimento01(["l_w"])

visu = VisualizarMovimentoArticulacao(golpe_csv=df2)
visu.plotar_movimento01(["l_w"])

######################################

# # fft
# # df = Globais.converter_array32(df)
analise = AnalisarCiclosDataset(golpe_csv=df)
feq =  analise.verificar_periodo2(idx_ponto=9) # obtem valores dos ciclos
print(f" frequencia dom:  {feq['dominant_frequency']} " )
print(f" periodos:  {feq['period_frames']:.2f} " )

analise1 = AnalisarCiclosDataset(golpe_csv=df2)
feq2 =  analise1.verificar_periodo2(idx_ponto=9) # obtem valores dos ciclos
print(f" frequencia dom:  {feq2['dominant_frequency']} " ) # pico de maior amplitude / componente de frequência com maior intensidade (amplitude) no espectro do movimento analisado.
print(f" periodos:  {feq2['period_frames']:.2f} " )
# print(feq)


# ###################################

# comparaão dtw
# analise2 = AnalisarSequenciasDTW(series1=df, series2=df2, articulacao="l_w") 

# distance_plot = analise2.calcular_distancia_dtw_lib(tipo_distancia="squared_euclidean_distance") # manhattan_distance
# print("Distancia (DTW): ", distance_plot["distance"])
# print("Distancia Normalizada (DTW): ", distance_plot["normalized_distance"])
# analise2.plotar_dtw_lib(alinhamento=distance_plot["alinhamento"])

# distance_plot2, similaridad, paths = analise2.calcular_distancia_dtaidistance_lib()
# print(distance_plot2)
# print(similaridad)
# analise2.plotar_dtaidistance_lib(melhor_caminho=paths)


##

# def calcular_similaridade(alinhamento, period_frames1, period_frames2, tolerancia_freq=0.05):
#     sim_dtw = 1 - min(alinhamento.normalizedDistance, 1)
#     freq1 = 1 / period_frames1
#     freq2 = 1 / period_frames2

#     freq_match = 1 if abs(freq1 - freq2) <= tolerancia_freq else 0

#     score = sim_dtw * 0.8 + freq_match * 0.2

#     print(f"Similaridade DTW normalizada: {sim_dtw:.3f}")
#     print(f"Frequência Vídeo 1: {freq1:.3f}, Vídeo 2: {freq2:.3f}, match: {freq_match}")
#     print(f"Score combinado: {score:.3f}")

#     return score

# def interpretar_similaridade(score):
#     if score >= 1.0:
#         return "Movimentos idênticos (ou extremamente semelhantes)"
#     elif score >= 0.8:
#         return "Movimentos muito semelhantes, com variações pequenas"
#     elif score >= 0.5:
#         return "Similaridade média, com alguns padrões parecidos"
#     elif score >= 0.2:
#         return "Baixa similaridade"
#     else:
#         return "Movimentos bem diferentes"

# score = calcular_similaridade(distance_plot['alinhamento'], feq['period_frames'], feq2['period_frames'])
# descricao = interpretar_similaridade(score)
# print(f"Similaridade final: {score:.3f} → {descricao}")

# ######

# erro_m = analise2.calcular_erro_medio_p2p(distance_plot["path"])
# def interpretar_erro_m(score):
#     if score == 0:
#         return "Movimentos praticamente idênticos"
#     elif score < 0.01:
#         return "Diferenças pequenas, ótima precisão"
#     elif score < 0.05:
#         return "Diferenças visíveis, mas toleráveis"
#     else:
#         return "Diferenças significativas"
    
# print(interpretar_erro_m(erro_m))

########################
def media_geral():
    resultados = []
    for articulacao in Globais.nome_articulacoes():
        analisador = AnalisarSequenciasDTW(df, df2, articulacao=articulacao)
        resultado_dtw = analisador.calcular_distancia_dtw_lib("euclidean_distance")
        erro_medio = analisador.calcular_erro_medio_p2p(path_alinhamento=resultado_dtw['path'])

        resultados.append({
            'articulacao': articulacao,
            'dtw_distancia': resultado_dtw['distance'],
            'dtw_normalizada': resultado_dtw['normalized_distance'],
            'erro_medio': erro_medio
        })

    df_resultados = pd.DataFrame(resultados)

    media_geral = {
        'dtw_media': df_resultados['dtw_distancia'].mean(),
        'dtw_normalizada_media': df_resultados['dtw_normalizada'].mean(),
        'erro_medio_global': df_resultados['erro_medio'].mean()
    }
            
    print(df_resultados)
    print("Média Geral:", media_geral)

# media_geral()


# Supondo que você já tenha uma instância do analisador e o resultado do DTW:
analisador = AnalisarSequenciasDTW(df, df2, articulacao="l_w")
resultado_dtw = analisador.calcular_distancia_dtw_lib("squared_euclidean_distance")

# 1. Calcule as distâncias ponto a ponto
distancias_info = analisador.calcular_distancias_p2p(resultado_dtw['path'])

# 2. Print dos valores obtidos
print("Média:", distancias_info["media"])
print("Desvio padrão:", distancias_info["desvio"])
print("Máximo:", distancias_info["max"])
print("Mínimo:", distancias_info["min"])
print("Distâncias:", distancias_info["distancias"])

# 3. Envie as distâncias para o plot
analisador.plotar_distancias_ponto_a_ponto(distancias_info["distancias"], resultado_dtw['path'])



## CALCULAR ERRO MEDIO PONTO A PONTO
## CALCULAR DISTANCIAS PONTO A PONTO
## VERIFICAR GRAFICOS ERRO MEDIO E DISTANCIAS
## VERIFICAR MEDIA geral DOS PONTOS
## ATÉ AGORA: ANALISE EM UMA ARTICULAÇÃO 
## REALIZAR: ANALISE TODOS OS PONTOS
## 



"""  
✅ Como detectar outliers?
📈 Regra prática mais usada:

Se o valor estiver acima da média + 2 desvios padrão, é um outlier.

media = df['erro_medio'].mean()
desvio = df['erro_medio'].std()

limite_superior = media + 2 * desvio
outliers = df[df['erro_medio'] > limite_superior]

✅ Como remover e recalcular a média?

# Remove outliers antes de calcular a média
sem_outliers = df[df['erro_medio'] <= limite_superior]

media_geral_corrigida = {
    'dtw_media': sem_outliers['dtw_distancia'].mean(),
    'dtw_normalizada_media': sem_outliers['dtw_normalizada'].mean(),
    'erro_medio_global': sem_outliers['erro_medio'].mean()
}
"""
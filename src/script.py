import glob
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
feq =  analise.verificar_periodo(idx_ponto=9) # obtem valores dos ciclos
print(f" frequencia dom:  {feq['dominant_frequency']} " )
print(f" periodos:  {feq['period_frames']:.2f} " )

analise1 = AnalisarCiclosDataset(golpe_csv=df2)
feq2 =  analise1.verificar_periodo(idx_ponto=9) # obtem valores dos ciclos
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


# MAIS COMPLETO
def analisar_todas_articulacoes_df(df1: pd.DataFrame, df2: pd.DataFrame, colunas_articulacoes: list):
    resultados = []

    for art in colunas_articulacoes:
        print(f"\n🔍 Analisando: {art}")
        analisador = AnalisarSequenciasDTW(df1, df2, art)
        res_dtw = analisador.calcular_distancia_dtw_lib("squared_euclidean_distance")

        distancias_info = analisador.calcular_distancias_p2p(res_dtw['path'])

        analisador.plotar_dtw_lib(res_dtw['alinhamento'])
        analisador.plotar_distancias_ponto_a_ponto(distancias_info["distancias"], res_dtw['path'])

        resultados.append({
            "articulacao": art,
            "dtw_distance": res_dtw["distance"],
            "normalized": res_dtw["normalized_distance"],
            "erro_medio": distancias_info["media"],
            "desvio": distancias_info["desvio"]
        })

    df_resultados = pd.DataFrame(resultados)
    
    print("\n📊 Tabela de Resultados:")
    print(df_resultados.sort_values("erro_medio", ascending=False).round(4))

    media_geral = {
        "dtw_media": df_resultados["dtw_distance"].mean(),
        "dtw_normalizada_media": df_resultados["normalized"].mean(),
        "erro_medio_global": df_resultados["erro_medio"].mean(),
        "desvio_global": df_resultados["desvio"].mean()
    }

    print("\n📈 Média Geral:")
    for k, v in media_geral.items():
        print(f"{k}: {v:.4f}")

    return df_resultados
analisador = AnalisarSequenciasDTW(df, df2, articulacao="l_w")
resultado_dtw = analisador.calcular_distancia_dtw_lib("squared_euclidean_distance")

distancias_info = analisador.calcular_distancias_p2p(resultado_dtw['path'])

print("Média:", distancias_info["media"])
print("Desvio padrão:", distancias_info["desvio"])
print("Máximo:", distancias_info["max"])
print("Mínimo:", distancias_info["min"])
print("Distâncias:", distancias_info["distancias"])

analisador.plotar_distancias_ponto_a_ponto(distancias_info["distancias"], resultado_dtw['path'])

def carregar_dataset_loop(caminho):
    estrutura = {}
    for caminho_csv in glob(f"{caminho}/*/*.csv"):
        nome_classe = os.path.basename(os.path.dirname(caminho_csv))
        if '-' not in nome_classe:
            continue
        tipo, lado = nome_classe.rsplit('-', 1)
        estrutura.setdefault(tipo, {}).setdefault(lado, []).append(caminho_csv)
    return estrutura

def carregar_dataset_teste(caminho):
    estrutura = {}
    for caminho_csv in glob(f"{caminho}/*/*.csv"):
        tipo = os.path.basename(os.path.dirname(caminho_csv))
        estrutura.setdefault(tipo, []).append(caminho_csv)
    return estrutura

def comparar_loops_testes(dataset_loop, dataset_teste, articulacoes):
    resultados_detalhados = []
    resumo = []

    for tipo_golpe, testes in dataset_teste.items():
        if tipo_golpe not in dataset_loop:
            print(f"Tipo {tipo_golpe} não está no dataset loop, pulando.")
            continue

        for lado in dataset_loop[tipo_golpe].keys():
            modelos = dataset_loop[tipo_golpe][lado]

            for csv_teste in testes:
                df_teste = pd.read_csv(csv_teste)

                for csv_modelo in modelos:
                    df_modelo = pd.read_csv(csv_modelo)

                    df_resultados = analisar_todas_articulacoes_df(df_modelo, df_teste, articulacoes)

                    for idx, row in df_resultados.iterrows():
                        resultados_detalhados.append({
                            "golpe": tipo_golpe,
                            "lado_modelo": lado,
                            "modelo": os.path.basename(csv_modelo),
                            "teste": os.path.basename(csv_teste),
                            "articulacao": row["articulacao"],
                            "dtw_distance": row["dtw_distance"],
                            "normalized": row["normalized"],
                            "erro_medio": row["erro_medio"],
                            "desvio": row["desvio"]
                    })
                    media_global = {
                        "golpe": tipo_golpe,
                        "lado_modelo": lado,
                        "modelo": os.path.basename(csv_modelo),
                        "teste": os.path.basename(csv_teste),
                        "dtw_media": df_resultados["dtw_distance"].mean(),
                        "dtw_normalizada_media": df_resultados["normalized"].mean(),
                        "erro_medio_global": df_resultados["erro_medio"].mean(),
                        "desvio_global": df_resultados["desvio"].mean()
                    }
                    resumo.append(media_global)
    df_resultados_detalhados = pd.DataFrame(resultados_detalhados)
    df_resumo = pd.DataFrame(resumo)

    return df_resultados_detalhados, df_resumo

## CALCULAR ERRO MEDIO PONTO A PONTO
## CALCULAR DISTANCIAS PONTO A PONTO
## VERIFICAR GRAFICOS ERRO MEDIO E DISTANCIAS
## VERIFICAR MEDIA geral DOS PONTOS
## ATÉ AGORA: ANALISE EM UMA ARTICULAÇÃO 
## REALIZAR: ANALISE TODOS OS PONTOS
## 


"""
💡 1. dtw_distance:

Distância absoluta (quanto maior, mais diferente as curvas são). Pode ser afetado por escala ou amplitude dos movimentos.
💡 2. normalized:

Versão normalizada da DTW (divide pelo comprimento total do caminho de alinhamento). Bom para comparar vídeos de durações diferentes.
💡 3. erro_medio:

Erro médio ponto a ponto entre as séries alinhadas. Isso quantifica a diferença média entre os keypoints após alinhamento DTW.
💡 4. desvio:

Mostra a variação dos erros ponto a ponto. Um desvio alto indica movimentos inconsistentes.
"""

"""  
Como detectar outliers?
Regra prática mais usada:

Se o valor estiver acima da média + 2 desvios padrão, é um outlier.

media = df['erro_medio'].mean()
desvio = df['erro_medio'].std()

limite_superior = media + 2 * desvio
outliers = df[df['erro_medio'] > limite_superior]

Como remover e recalcular a média?

# Remove outliers antes de calcular a média
sem_outliers = df[df['erro_medio'] <= limite_superior]

media_geral_corrigida = {
    'dtw_media': sem_outliers['dtw_distancia'].mean(),
    'dtw_normalizada_media': sem_outliers['dtw_normalizada'].mean(),
    'erro_medio_global': sem_outliers['erro_medio'].mean()
}
"""
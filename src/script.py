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

# visualização
# df = pd.read_csv(Caminhos.teste_periodiciodade13) #eixo x e eixo y em relação ao tempo/franes
# visu = VisualizarMovimentoArticulacao(golpe_csv=df)
# visu.plotar_movimento01(["l_k"])

######################################

# # fft
# # df = Globais.converter_array32(df)
# analise = AnalisarCiclosDataset(golpe_csv=df)
# feq =  analise.verificar_periodo(idx_ponto=9) # obtem valores dos ciclos
# # obtendo frequencia
# print(f" frequencia:  {feq['period_frames']:.2f} " )

###################################

# # comparaão dtw
# df2 = pd.read_csv(Caminhos.teste_periodiciodade17)
# analise2 = AnalisarSequenciasDTW(series1=df, series2=df2, articulacao="r_k") 

# distance_plot = analise2.calcular_distancia_dtw_lib(tipo_distancia="squared_euclidean_distance") # manhattan_distance
# print(distance_plot["distance"])
# print(distance_plot["normalized_distance"])
# analise2.plotar_dtw_lib(alinhamento=distance_plot["alinhamento"])

# distance_plot2, similaridad, paths = analise2.calcular_distancia_dtaidistance_lib()
# print(distance_plot2)
# print(similaridad)
# analise2.plotar_dtaidistance_lib(melhor_caminho=paths)

####################################

# classificação dtw-knn (testar outros ml)

dtw_knn_classfy = ClassificadorDTW_KNN()
dtw_knn_classfy.carregar_dataset()
# dtw_knn_classfy.fit()

x_train, x_test, y_train, y_test = dtw_knn_classfy.train_test()
dtw_knn_classfy.fit(x_train, y_train)
# print("x_train", type(x_train), len(x_train))
# print("y_train",  type(y_train), len(y_train))
# print("x_test", type(x_test), len(x_test))
# print("y_test", type(y_test) , len(y_test))

print("[DEBUG] x_train[0] type:", type(x_train[0]), "shape:", getattr(x_train[0], "shape", "sem shape"))
print("[DEBUG] x_test[0] type:", type(x_test[0]), "shape:", getattr(x_test[0], "shape", "sem shape"))

# avaliação

y_pred = dtw_knn_classfy.predict(x_train=x_train, y_train=y_train, x_test=x_test)
print(dtw_knn_classfy.evualuate(x_test=x_test, y_test=y_test,y_pred=y_pred))

# predição

# 2. carregar vídeo externo
golpe = pd.read_csv(Caminhos.teste_periodiciodade20)
golpe_conver = Globais.converter_array32(golpe)

segmentos_teste = Globais.segmentar_com_janela_sliding(
    golpe_conver,
    janela=len(golpe_conver),
    passo=len(golpe_conver)
)

# 3. Predição
if not segmentos_teste:
    print("⚠️ O vídeo externo é muito curto ou a janela está maior que o número de frames.")
else:
    y_pred_externo = dtw_knn_classfy.predict(
        x_train=dtw_knn_classfy.x_train,
        y_train=dtw_knn_classfy.y_train,
        x_test=segmentos_teste
    )
    for i, pred in enumerate(y_pred_externo):
        nome_classe = dtw_knn_classfy.nomes_golpes_classe[pred]
        print(f"Segmento {i+1} predito como ➤  {nome_classe}")
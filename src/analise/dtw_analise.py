import dis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.globais import Globais
from dtw import dtw
from dtaidistance import dtw as dtw2
from utils.distance_methods import DistancesDTW

class AnalisarSequenciasDTW: # reorganizar saidas(returns) dos metodos principais
    def __init__(self, series1: np.ndarray, series2: np.ndarray, articulacao: str):
        self.articulacao = articulacao  # Salva o nome da articulação
        # self.serie1 = self._preparar_articulação(series1, articulacao) 
        self.serie1 = self._preparar_articulação(series1, "l_k") 
        self.serie2 = self._preparar_articulação(series2, articulacao)

        if self.serie1.ndim > 1:
            self.serie1 = np.linalg.norm(self.serie1, axis=1)

        if self.serie2.ndim > 1:
            self.serie2 = np.linalg.norm(self.serie2 , axis=1)
    
    def _preparar_articulação(self, df: pd.DataFrame, articulacao: str) -> np.ndarray:
        points = np.array(
            [Globais.ast_func(point) 
             for point in df[articulacao].values])
        return np.linalg.norm(points, axis=1) if points.ndim > 1 else points
    
    # calcula a distância DTW entre duas séries temporais.
    def calcular_distancia_dtw_lib(self, tipo_distancia: str):
        dict_distancias = {
            "squared_euclidean_distance": DistancesDTW.squared_euclidean_distance,
            "chebyshev_distance": DistancesDTW.chebyshev_distance,
            "manhattan_distance": DistancesDTW.manhattan_distance,
            "euclidean_distance": DistancesDTW.euclidean_distance,
            "cosine_distance": DistancesDTW.cosine_distance,
            "canberra_distance": DistancesDTW.canberra_distance,
        }
        dist_methods = dict_distancias.get(tipo_distancia, DistancesDTW.squared_euclidean_distance)
        #dtw
        alinhamento = dtw(self.serie1, self.serie2,
                         keep_internals=True,
                         distance_only=False,
                         dist_method=dist_methods)
        path = list(zip(alinhamento.index1, alinhamento.index2))

        return {
            'alinhamento': alinhamento,
            'distance': alinhamento.distance,
            'normalized_distance': alinhamento.normalizedDistance,
            'path': path
        }  # alinhamento.distance, alinhamento.normalizedDistance,

    def calcular_distancia_dtaidistance_lib(self):
        distancia, paths = dtw2.warping_paths(self.serie1, self.serie2, use_c=False)
        melhor_caminho = dtw2.best_path(paths)
        valor_similaridade = distancia / len(melhor_caminho)     

        return distancia, valor_similaridade, melhor_caminho
    
    def plotar_dtw_lib(self, alinhamento):
        plt.figure(figsize=(15, 10))
        plt.plot(self.serie1, label=f'Série 1 ({self.articulacao})',  color='green')
        plt.plot(self.serie2, label=f'Série 2 ({self.articulacao})', linestyle='-', color='grey')
        plt.title(f'Séries Temporais Originais - {self.articulacao}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # plt.figure(figsize=(15, 10))
        # path = list(zip(alinhamento.index1, alinhamento.index2))
        # plt.plot(self.serie1, label=f'Série 1 ({self.articulacao})', color='green', marker='o' , linestyle=':')
        # plt.plot(self.serie2, label=f'Série 2 ({self.articulacao})', color='grey', marker='x', linestyle='--')
        # for i, j in path:
        #     plt.plot([i, j], [self.serie1[i], self.serie2[j]], color='white', alpha=0.1)
        # plt.title(f'Comparação Ponto-a-Ponto (lib dtw) - {self.articulacao}')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


    def plotar_dtaidistance_lib(self, melhor_caminho):
        plt.figure(figsize=(15, 10))
        plt.plot(self.serie1, label=f'Série 1 ({self.articulacao})', color='green')
        plt.plot(self.serie2, label=f'Série 2 ({self.articulacao})', linestyle='-', color='grey')
        plt.title(f'Séries Temporais Originais - {self.articulacao}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
        caminho = np.array(melhor_caminho)
        plt.plot(caminho[:, 0], caminho[:, 1], 'green', marker='o', linestyle='-')
        n = len(self.serie1)
        m = len(self.serie2)
        plt.plot([0, n-1], [0, m-1], 'r--', label='Diagonal (referência)')
        plt.title(f'Melhor Caminho- {self.articulacao}')
        plt.xlabel('Índice Série 1')
        plt.ylabel('Índice Série 2')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # plt.figure(figsize=(15, 10))
        # plt.plot(self.serie1, label=f'Série 1 ({self.articulacao})', color='green', marker='o'  , linestyle=':')
        # plt.plot(self.serie2, label=f'Série 2 ({self.articulacao})', color='grey', marker='x', linestyle='--')
        # for i, j in melhor_caminho:
        #     plt.plot([i, j], [self.serie1[i], self.serie2[j]], color='white', alpha=0.1)
        # plt.title(f'Comparação Ponto-a-Ponto (lib dtaidistance) - {self.articulacao}')

        # plt.legend()
        # plt.tight_layout()
        # plt.show()
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
        self.serie1 = self._preparar_articulação(series1, articulacao) 
        # self.serie1 = self._preparar_articulação(series1, "r_w")
         
        self.serie2 = self._preparar_articulação(series2, articulacao)
        # self.serie2 = self._preparar_articulação(series2, "r_w")

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

    def plotar_dtw_lib(self, alinhamento):
        fig, axs = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
        axs[0].plot(self.serie1, color='blue')
        axs[0].set_title(f'Vídeo 1: ({self.articulacao})')
        axs[0].grid(True)
        axs[1].plot(self.serie2, color='black')
        axs[1].set_title(f'Vídeo 2: ({self.articulacao})')
        axs[1].grid(True)

        fig.suptitle(f'Séries Temporais Originais - ARTICULAÇÃO: {self.articulacao}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # espaço para o suptitle
        plt.show()

        plt.figure(figsize=(15, 6))
        path = list(zip(alinhamento.index1, alinhamento.index2))
        plt.plot(self.serie1, label='Vídeo 1', color='blue', marker='o', linestyle=':')
        plt.plot(self.serie2, label='Vídeo 2', color='black', marker='x', linestyle='--')
        for i, j in path:
            plt.plot([i, j], [self.serie1[i], self.serie2[j]], color='grey', alpha=0.4)
        plt.title(f'Ponto-a-Ponto (lib dtw) - ARTICULAÇÃO: {self.articulacao}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 8))
        i_vals, j_vals = zip(*path)
        n = len(self.serie1)
        m = len(self.serie2)
        plt.plot(i_vals, j_vals, color='green')
        plt.plot([0, n-1], [0, m-1], 'r--', label='Diagonal (referência)')
        plt.xlabel("Frames Vídeo 1")
        plt.ylabel("Frames Vídeo 2")
        plt.title(f'MELHOR CAMINHO - ARTICULAÇÃO: ({self.articulacao})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def calcular_erro_medio_p2p(self, path_alinhamento) -> float:
        # path = list(zip(alinhamento.index1, alinhamento.index2))
        erros = []

        for i, j in path_alinhamento:
            if i < len(self.serie1) and j < len(self.serie2):
                v1 = self.serie1[i]
                v2 = self.serie2[j]
                erro = abs(v1 - v2)
                print(erro)
                erro2 = abs(v1 - v1) ** 2
                print(erro2)
                erros.append(erro)
        erro_medio = np.mean(erros)
       
        return erro_medio

    def calcular_distancias_p2p(self, path_alinhamento):
        distancias = [abs(self.serie1[i] - self.serie2[j]) for i,j in path_alinhamento]
        
        return {
            "distancias": distancias,
            "media": np.mean(distancias),
            "desvio": np.std(distancias),
            "max": np.max(distancias),
            "min": np.min(distancias)
        }
    
    def plotar_distancias_ponto_a_ponto(self, distancias, alinhamento):
        x_vals = range(len(distancias))
        plt.figure(figsize=(15, 5))
        plt.plot(x_vals, distancias, label='Erro ponto-a-ponto', color='purple')
        plt.axhline(np.mean(distancias), color='green', linestyle='--', label='Média')
        plt.title(f'Erro ponto-a-ponto (DTW) - ARTICULAÇÃO: {self.articulacao}')
        plt.xlabel("Par Alinhado (i,j)")
        plt.ylabel("Distância |v1 - v2|")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
""" 
    # def calcular_distancia_dtaidistance_lib(self):
    #     distancia, paths = dtw2.warping_paths(self.serie1, self.serie2, use_c=False)
    #     melhor_caminho = dtw2.best_path(paths)
    #     valor_similaridade = distancia / len(melhor_caminho)     

    #     return distancia, valor_similaridade, melhor_caminho


    # def plotar_dtaidistance_lib(self, melhor_caminho):
    #     fig, axs = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
    #     axs[0].plot(self.serie1, color='blue')
    #     axs[0].set_title('Vídeo 1')
    #     axs[0].grid(True)
    #     axs[1].plot(self.serie2, color='black')
    #     axs[1].set_title('Vídeo 2')
    #     axs[1].grid(True)

    #     fig.suptitle(f'Séries Temporais Originais - ARTICULAÇÃO: {self.articulacao}', fontsize=14)
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])  # espaço para o suptitle
    #     plt.show()


    #     plt.figure(figsize=(15, 10))
    #     plt.plot(self.serie1, label='Vídeo 1', color='blue', marker='o'  , linestyle=':')
    #     plt.plot(self.serie2, label='Vídeo 2', color='black', marker='x', linestyle='--')
    #     for i, j in melhor_caminho:
    #         plt.plot([i, j], [self.serie1[i], self.serie2[j]], color='grey', alpha=0.4)
    #     plt.title(f'Ponto-a-Ponto (lib dtaidistance) - ARTICULAÇÃO: {self.articulacao}')

    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()


    #     plt.figure(figsize=(15, 10))
    #     caminho = np.array(melhor_caminho)
    #     plt.plot(caminho[:, 0], caminho[:, 1], 'green')
    #     n = len(self.serie1)
    #     m = len(self.serie2)
    #     plt.plot([0, n-1], [0, m-1], 'r--', label='Diagonal (referência)')
    #     plt.title(f'MELHOR CAMINHO - ARTICULAÇÃO: ({self.articulacao})')
    #     plt.xlabel('Frames Vídeo 1')
    #     plt.ylabel('Frames Vídeo 2')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
"""
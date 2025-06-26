from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.globais import Globais
from utils.caminhos import Caminhos
import ast

class VisualizarMovimentoArticulacao:
    def __init__(self, golpe_csv: pd.DataFrame):
        self.df = golpe_csv
        
    # eixo x e eixo y plotados em relação ao tempo/frames
    def plotar_movimento01(self, articulacoes: List[str], n_points=50):
        if isinstance(articulacoes, str):
            articulacoes = [articulacoes]
            
        indices = np.linspace(0, len(self.df['frame']) - 1, n_points)
        indices = np.round(indices).astype(int)
        
        frames = self.df["frame"].iloc[indices]

        plt.figure(figsize=(18, 10))  # Aumenta largura e altura

    # Plot coordenada X (primeiro gráfico, linha 1)
        plt.subplot(2, 1, 1)
        for articulacao in articulacoes:
            coordinates = self.df[articulacao].apply(ast.literal_eval)
            x = coordinates.apply(lambda p: p[0])
            x_sub = x.iloc[indices]
            plt.plot(frames, x_sub, label=f'{articulacao} - X')
        plt.xlabel("sequência temporal")
        plt.ylabel("Coord. X")
        plt.title("Movimento Lateral (X)")
        plt.grid(True)
        plt.legend()

        # Plot coordenada Y (segundo gráfico, linha 2)
        plt.subplot(2, 1, 2)
        for articulacao in articulacoes:
            coordinates = self.df[articulacao].apply(ast.literal_eval)
            y = coordinates.apply(lambda p: p[1])
            y_sub = y.iloc[indices]
            plt.plot(frames, y_sub, label=f'{articulacao} - Y')
        plt.xlabel("sequência temporal")
        plt.ylabel("Coord. Y")
        plt.title("Movimento Vertical (Y)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # ou outro valor, como 0.7
        plt.show()


        # plotar o movimento no formato de articulação/ções
        def plotar_movimento02(self):
            pass
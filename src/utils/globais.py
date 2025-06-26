import os 
import numpy as np
import pandas as pd
import ast
from typing import List

class Globais:        
    @staticmethod
    def nome_articulacoes():
        
        nome_articulacoes = [
        'nose', 'l_ey', 'r_ey', 'l_ea', 'r_ea', 'l_s', 'r_s', 'l_el', 'r_el',
        'l_w', 'r_w', 'l_h', 'r_h', 'l_k', 'r_k', 'l_a', 'r_a' ]

        return nome_articulacoes
    
    @staticmethod
    def gerar_init(caminho_pasta):
        init_path = os.path.join(caminho_pasta, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass

    @staticmethod
    def remover_coluna_frame(df):
        if 'frame' in df.columns:
            df = df.drop(columns=['frame'])
        return df
    
    @staticmethod
    def ast_func(points):
        return ast.literal_eval(points)

    @staticmethod
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

    @staticmethod
    def converter_array32(df: pd.DataFrame) -> np.ndarray:
        vetores = df.apply(Globais.converter_frame_vetor, axis=1).values.tolist()
        return np.array(vetores, dtype=np.float32)
    

    @staticmethod
    def segmentar_com_janela_sliding(array: np.ndarray, janela=35, passo=35) -> List[np.ndarray]:
        segmentos = []
        for i in range(0, len(array) - janela + 1, passo):
            trecho = array[i:i+janela]
            if trecho.shape[0] == janela:
                segmentos.append(trecho)
        return segmentos

import os 
import numpy as np

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
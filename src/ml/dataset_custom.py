import os
import ast
from re import A
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DatasetPersonalizado(Dataset):
    def __init__(self, dataset_csv_path):
        self.numero_amostras_path = [] # paath completo para cada csv?
        self.rotulos_golpes = [] # indices correspondentes as classes/golpes
        self.nomes_golpes = [] # nomes das classes/golpes

        self.golpes_classe = sorted(os.listdir(dataset_csv_path)) # nome das subpastas que representam o rotulo das classess
        self.golpe_idx = {golpe_nome: idx for idx, golpe_nome in enumerate(self.golpes_classe)}

        for nome_classe in self.golpes_classe:
            classe_path = os.path.join(dataset_csv_path, nome_classe)
            if not os.path.isdir(classe_path):
                continue
            for arquivo in os.listdir(classe_path):
                if arquivo.endswith('.csv'):
                    self.numero_amostras_path.append(os.path.join(classe_path, arquivo))
                    self.rotulos_golpes.append(self.golpe_idx[nome_classe])
                    self.nomes_golpes.append(nome_classe)

    def __len__(self):
        return len(self.numero_amostras_path)
    
    def indice_nome(self, index):
        return self.golpes_classe[index]

    def __getitem__(self, index):
        csv_path = self.numero_amostras_path[index]
        
        df = pd.read_csv(csv_path)

        #remover conluna frame
        if 'frame' in df.columns: 
            df = df.drop(columns=['frame'])

        coordenadas_extraidas = df.applymap(lambda s: ast.literal_eval(s) if isinstance(s, str) else s) # "(x,y)" -> (x,y) float
        
        coordenadas = []

        for linha in coordenadas_extraidas.itertuples(index=False):
            linha_convertida = []
            for ponto in linha:
                if( isinstance(ponto, (tuple, list)) and
                   len(ponto) == 2 and
                   all(isinstance(v, (int, float)) or str(v).replace('.','', 1).isdigit() for v in ponto)
                ):
                    linha_convertida.append((float(ponto[0]), float(ponto[1])))
                else:
                    linha_convertida.append((0.0, 0.0))
            coordenadas.append(linha_convertida)

        coordenadas = np.array(coordenadas, dtype=np.float32)
        # numpy.ndarray # (frames, 17, 2)

        coordenadas = coordenadas.reshape(coordenadas.shape[0], -1)  # reshape (frames, 34)


        x = torch.tensor(coordenadas, dtype=torch.float32)
        y = torch.tensor(self.rotulos_golpes[index], dtype=torch.long)
        

        return x, y
    # 🏷️ 🔁 
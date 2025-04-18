import os
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from ml.preparar_entrada.validacao_vals import Validacao
from utils.utilidades import Utilidades

class DatasetPersonalizado(Dataset):
    def __init__(self, dataset_csv_path): # verbose = true / false - controlar logs 
        self.numero_amostras_path = [] # paath completo para cada csv?
        self.rotulos_golpes = [] # indices correspondentes as classes/golpes
        self.nomes_golpes = [] # nomes das classes/golpes
        self.nome_golpes_classe = sorted([
            pasta.strip() for pasta in os.listdir(dataset_csv_path)
            if os.path.isdir(os.path.join(dataset_csv_path, pasta)) and not pasta.startswith("__")
        ]) # nome das subpastas que representam o rotulo das classess
        self.golpe_idx = {golpe_nome: idx for idx, golpe_nome in enumerate(self.nome_golpes_classe)}
        
        print("📁 [CLASSES DETECTADAS: ", self.nome_golpes_classe,"]")
       
        for nome_classe in self.nome_golpes_classe:
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
        return self.nome_golpes_classe[index]

    def __getitem__(self, index):
        csv_path = self.numero_amostras_path[index]
        
        df = pd.read_csv(csv_path)

        df = Utilidades.remover_coluna_frame(df) 

        coordenadas_extraidas = df.applymap(lambda s: Validacao.eval_valido(s) if isinstance(s, str) else s) # "(x,y)" -> (x,y) float
        
        coordenadas = []

        for linha in coordenadas_extraidas.itertuples(index=False):
            if all(ponto is None or ponto == '' for ponto in linha): # ignorar linhas vazias
                continue

            linha_convertida = []
            for ponto in linha:
                if(isinstance(ponto, (tuple, list)) and
                    len(ponto) == 2 and
                    all(Validacao.valor_valido(v) for v in ponto)
                ):
                    linha_convertida.append((float(ponto[0]), float(ponto[1])))
                else:
                    linha_convertida.append((0.0, 0.0))

            coordenadas.append(linha_convertida)

        coordenadas = np.array(coordenadas, dtype=np.float32)
        # numpy.ndarray # (frames, 17, 2)

        coordenadas = coordenadas.reshape(coordenadas.shape[0], -1)  
        # reshape (frames, 34)  # achata o shape

        x = torch.tensor(coordenadas, dtype=torch.float32)
        y = torch.tensor(self.rotulos_golpes[index], dtype=torch.long)
        
        return x, y

    def collate_pad_batch(self, batch): 
        xs, ys = zip(*batch)
        frames_total_batch = [len(seq) for seq in xs]
        # seleciona o video com maior quantidade de frame dentro do batch
        xs_padd = pad_sequence(xs, batch_first=True, padding_value=0)
        # preenche com 0 / tratar esses 0 preenchidos com padding
        return xs_padd, torch.tensor(ys), frames_total_batch
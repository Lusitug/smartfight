from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report #,  accuracy_score, f1_score, recall_score
import os
from utils.caminhos import Caminhos
from utils.globais import Globais
from utils.distance_methods import DistancesDTW
import pandas as pd
import numpy as np
from typing import List
from dtw import dtw
from dtaidistance import dtw as dtai

class ClassificadorDTW_KNN:
    def __init__(self, k: int = 5): # add cliclos para slidding
        self.dataset = Caminhos.dataset_csv
        self.k = k
        self.nomes_golpes_classe = []
        self.indices_golpes_classe = {}
        self.x = []
        self.y = []

    def carregar_dataset(self):
        self.nomes_golpes_classe = sorted ([
            pasta for pasta in os.listdir(self.dataset)
            if os.path.isdir(os.path.join(self.dataset, pasta)) and not pasta.startswith('__')
            ])
        self.indices_golpes_classe = {name: idx for idx, name in enumerate(self.nomes_golpes_classe)}

        print("Classes detectadas:", self.nomes_golpes_classe)

        for nome_classe in self.nomes_golpes_classe:
            classe_pasta = os.path.join(self.dataset, nome_classe)
            for arquivo in os.listdir(classe_pasta):
                if arquivo.endswith('.csv'):
                    df = pd.read_csv(os.path.join(classe_pasta, arquivo))
                    keypoints = Globais.converter_array32(df)
                   
                    print(f"[INFO] {arquivo} | keypoints.shape = {keypoints.shape}, {keypoints.dtype}")
                    
                    segmentos = self._segmentar_janela_com_sliding(keypoints)
                    
                    for i, segmento in enumerate(segmentos):
                        if segmento.shape[1] != 34:  # ajuste para seu total de features (17 pontos x 2 = 34)
                            print(f"⚠️ Segmento inválido em {arquivo} | segmento {i} shape: {segmento.shape}")

                    self.x.extend(segmentos)
                    self.y.extend([self.indices_golpes_classe[nome_classe]] * len(segmentos)) 

    
    def _segmentar_janela_com_sliding(self, array: np.ndarray, janela: int = 50, passos: int = 50)  -> List[np.ndarray]:
        segmentos = []
        for i in range(0, len(array) - janela +1 , passos):
            segmento = array[i:i+janela]
            if len(segmento) == janela:
                segmentos.append(segmento)
        return segmentos
    
    def train_test(self, dividir_treino_teste=0.1):
        return train_test_split(self.x, self.y, test_size=dividir_treino_teste, stratify=self.y)
    
    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train

    def predict(self, x_train, y_train, x_test):
        y_pred = []

        # print("x_train", type(x_train), len(x_train))        # print("y_train",  type(y_train), len(y_train))       # print("x_test", type(x_test), len(x_test))
        
        for exemplo in x_test:
            # print("x_train", type(exemplo), len(exemplo))
            
            distancias = [dtw(exemplo, exemplo_treino, distance_only=True).distance 
                          for exemplo_treino in x_train]
            
            # print("[DEBUG] distâncias calculadas:", distancias[:5])  # Mostra só as 5 primeiras
            # print("[DEBUG] tipo do primeiro:", type(distancias[0]))

            # distancias = [dtai.distance(exemplo, exemplo_treino,) for exemplo_treino in x_train]
            indices = np.argsort(distancias)[:self.k]
            votos = [y_train[i] for i in indices]
            classes = max(set(votos), key=votos.count)
            y_pred.append(classes)

        return y_pred
    
    def evualuate(self, x_test, y_test, y_pred):
        print("[DEBUG] classes no y_test:", sorted(set(y_test)))
        print("[DEBUG] classes no y_pred:", sorted(set(y_pred)))
        print("[DEBUG] nomes das classes:", self.nomes_golpes_classe)
        
        labels_usadas = sorted(unique_labels(y_test, y_pred))
        return classification_report(
            y_test,
            y_pred,
            labels=labels_usadas,
            target_names=[self.nomes_golpes_classe[i] for i in labels_usadas],
            zero_division=0
        )

        # distancia = [dtw2.distance(amostra, amostra_treino)
        #              for amostra_treino in self.X_train]

        #  distancias = [dtw(amostra, amostra_treino, distance_only=True).distance
        #     #               for amostra_treino in self.X_train]def evualuate(self, x_test, y_test, y_pred):
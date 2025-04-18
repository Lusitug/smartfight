
import os

import pandas as pd
import torch
from ml.lstm.classificador_lstm import ClassificadorLSTM
from ml.preparar_entrada.preparar_entrada_treino import DatasetPersonalizado
from ml.preparar_entrada.validacao_vals import Validacao
from utils.utilidades import path_modelo_treinado, path_keypoints2csv

class InferenciaLSTM:
    def __init__(self,
                modelo_path=os.path.join(
                    path_modelo_treinado, "smartfight_lstm.h5"),
                feature_frame=34, neuronios_ocultos=128,
                bidirecional=False, num_camadas=1           
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # carregar dataset para acessar nomes das classes - com muitos daddos sera uitil?
        self.dataset_temporario = DatasetPersonalizado(dataset_csv_path=path_keypoints2csv)
        self.golpes_classe = self.dataset_temporario.nome_golpes_classe
        # iniciar o modelo
        self.modelo_predicao = ClassificadorLSTM(
            len_featrures_frames=feature_frame,
            len_neuronios_ocultos=neuronios_ocultos,
            len_tipos_golpes=len(self.golpes_classe),
            bidirecional=bidirecional,
            len_camadas=num_camadas
        ).to(device=self.device)
        # carregrr o modelo treinado
        self.modelo_predicao.state_dict(torch.load(modelo_path, map_location=self.device))
        self.modelo_predicao.eval()# avaliar

    def carregar_csv_inferencia(self, path_csv):
        df = pd.read_csv(path_csv)
        
        if 'frame' in df.columns:
            df = df.drop(columns=['frame'])
        print(df.columns)

        keypoints = df.applymap(eval).values
        keypoints = [[(float(x), float(y)) 
                        if isinstance(x, (int, float)) and
                            Validacao.valor_valido(x) and
                            Validacao.valor_valido(y) 
                        else (0.0, 0.0) 
                        for (x, y) in linha] 
                    for linha in keypoints]
        keypoints = torch.tensor(keypoints).view(len(keypoints), -1).unsqueeze(0)
        # camada de lote 1 (1,frames,features)
        return keypoints
    
    def prever(self, path_csv):
        tensor_keypoints = self.carregar_csv_inferencia(path_csv=path_csv).to(self.device)
        print(tensor_keypoints)
        with torch.no_grad():
            # tensor_keypoints = self.carregar_csv_inferencia(path_csv=path_csv).to(self.device)
            real_frame = [tensor_keypoints.shape[1]]
            saida = self.modelo_predicao(tensor_keypoints, real_frame)
            print("saida",saida)
            predicao = torch.argmax(saida, dim=1).item()
            print("predicao", predicao)
            return self.golpes_classe[predicao]
        


        

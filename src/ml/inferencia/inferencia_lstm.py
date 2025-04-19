import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ml.lstm.classificador_lstm import ClassificadorLSTM
from ml.preparar_entrada.preparar_entrada_treino import DatasetPersonalizado
from ml.preparar_entrada.validacao_vals import Validacao
from utils.utilidades import Utilidades

class InferenciaLSTM:
    def __init__(self, # os parametros para a inferencia deve conter q mesma arquitetura do treinamento 
                modelo_path=os.path.join(
                    Utilidades.path_modelo_treinado, "smartfight_lstm.h5"),
                feature_frame=34, neuronios_ocultos=128,
                bidirecional=False, num_camadas=1           
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # carregar dataset para acessar nomes das classes - com muitos daddos sera uitil?
        self.dataset_temporario = DatasetPersonalizado(dataset_csv_path=Utilidades.path_keypoints2csv)
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
        self.modelo_predicao.load_state_dict(torch.load(modelo_path, map_location=self.device))
        # self.modelo_predicao.state_dict(torch.load(modelo_path, map_location=self.device)) #  strict=False
        self.modelo_predicao.eval()# avaliar

    def carregar_csv_inferencia(self, path_csv):
        df = pd.read_csv(path_csv)

        df = Utilidades.remover_coluna_frame(df)

        df = df.applymap(lambda s: Validacao.eval_valido(s) if isinstance(s, str) else s)

        coordenadas = []
        for linha in df.itertuples(index=False):
            linha_convertida = []
            for ponto in linha:
                if isinstance(ponto, (tuple, list)) and len(ponto) == 2 and all(Validacao.valor_valido(v) for v in ponto):
                    linha_convertida.append((float(ponto[0]), float(ponto[1])))
                else:
                    linha_convertida.append((0.0, 0.0))
            coordenadas.append(linha_convertida)

        coordenadas = np.array(coordenadas, dtype=np.float32).reshape(len(coordenadas), -1)
        tensor_keypoints = torch.tensor(coordenadas, dtype=torch.float32).unsqueeze(0)  
        return tensor_keypoints
    
    def prever(self, path_csv):
        with torch.no_grad():
            tensor_keypoints = self.carregar_csv_inferencia(path_csv=path_csv).to(self.device)
            real_frame = [tensor_keypoints.shape[1]] # print(real_frame)
            saida = self.modelo_predicao(tensor_keypoints, real_frame) # print("saida: ",saida)
            probabilidade = F.softmax(saida, dim=1)
            print("probabilidade: ", probabilidade)
            predicao = torch.argmax(probabilidade, dim=1).item()
            print("indice predicao: ", predicao)
            return self.golpes_classe[predicao]
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

    def prever_golpes_em_memoria(self, keypoints, tamanho_janela=35, etapa=35):
        with torch.no_grad():
                tensor_keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
                tensor_keypoints = tensor_keypoints.view(tensor_keypoints.shape[0], tensor_keypoints.shape[1], -1)
                print(tensor_keypoints.shape)
                janelas = self._janelas_deslizantes(tensor_keypoints, tamanho_janela, etapa)

                # predicoes = []
                predicao_conf = []
                for janela in janelas:
                    real_frame = [janela.shape[1]] 
                    print("real_frame: ",real_frame)

                    saida = self.modelo_predicao(janela, real_frame)
                    print("saida: ", saida)

                    probabilidade = F.softmax(saida, dim=1)
                    print("Probabilidade: ", probabilidade)

                    indice_predicao = torch.argmax(probabilidade, dim=1).item()
                    print("Indice Previsao: ", indice_predicao)

                    probabilidade_classe = probabilidade[0, indice_predicao].item()
                    print("Probabilidade por Classe: ", probabilidade_classe)

                    predicao_conf.append({
                        "classe": self.golpes_classe[indice_predicao],
                        "confianca": probabilidade_classe,
                    })
                print(predicao_conf)
                return predicao_conf

    def _janelas_deslizantes(self, keypoints, tamanho_janela=35, etapa=35):
        janelas = []
        print(keypoints.shape)
        num_frames = keypoints.shape[1]
        print("num_frames: ", num_frames)

            # divisao = 4 if num_frames % 4 == 0 else 3
        if num_frames > tamanho_janela:
            divisao = max(3, min(5, num_frames // tamanho_janela))
            tamanho_janela = num_frames // divisao
            etapa = tamanho_janela
            print(f"Tamanho ajustado: {tamanho_janela}, Etapa : {etapa}")

        if num_frames < tamanho_janela:
            etapa = keypoints[:, :num_frames, :]
            janelas.append(keypoints[:, :num_frames, :])
            print("chegou aqui - janela única criada com tamanho: ", num_frames)
        else:
            print("Etapa: ", etapa)
            print("Tamanho da Janela: ", tamanho_janela)
            
            for i in range(0, num_frames - tamanho_janela + 1, etapa):
                janela = keypoints[:, i:i + tamanho_janela, :]
                print("Janela tamanho: ", janela.shape[1])
                janelas.append(janela)

            if (num_frames - tamanho_janela) % etapa != 0:
                ultima_janela = keypoints[:, -(num_frames % tamanho_janela):, :]
                print("Última janela  tamanho: ", ultima_janela.shape[1])
                    # janelas.append(ultima_janela)
                    
                if ultima_janela.shape[1] >= tamanho_janela // 4:
                    janelas.append(ultima_janela)
                    print("Última janela adicionda tamanho: ", ultima_janela.shape[1])
                else:
                    print("Última janela ignorada por ser menor que 1/4 do tamanho da janela.")
        print("janelas: ", len(janelas))
        return janelas

    """  obsoleto 
    def carregar_csv_inferencia(self, path_csv): # obsoleto
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
        """ 
    
    """ obsoleto
    def prever_sequencias_videos(self, path_csv, tamanho_janela=40, etapa=40): #obsoleto
        with torch.no_grad():
            tensor_keypoints = self.carregar_csv_inferencia(path_csv=path_csv).to(self.device)
            janelas = self._janelas_deslizantes(tensor_keypoints, tamanho_janela, etapa)

            # predicoes = []
            predicao_conf = []
            for janela in janelas:
                real_frame = [janela.shape[1]] 
                print("real_frame: ",real_frame)

                saida = self.modelo_predicao(janela, real_frame) # print("saida: ",saida)
                print("probabilidade_classe: ", saida)

                probabilidade = F.softmax(saida, dim=1)
                print("Probabilidade: ", probabilidade)

                indice_predicao = torch.argmax(probabilidade, dim=1).item()
                print("Indice Previsao: ", indice_predicao)

                probabilidade_classe = probabilidade[0, indice_predicao].item()
                print("Probabilidade por Classe: ", probabilidade_classe)

                predicao_conf.append({
                    "classe": self.golpes_classe[indice_predicao],
                    "confianca": probabilidade_classe,
                })

            return predicao_conf
            # predicoes_filtradas = [predicoes[i] for i in range(len(predicoes)) if i == 0 or predicoes[i] != predicoes[i - 1]]
            # print("Predições Filtradas: ", predicoes_filtradas)
            # return predicoes_filtradas
        """
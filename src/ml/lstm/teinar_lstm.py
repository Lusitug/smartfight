import os
import torch
from time import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ml.lstm.classificador_lstm import ClassificadorLSTM
from utils.utilidades import gerar_init, path_keypoints2csv, path_modelo_treinado
from src.ml.preparar_entrada.preparar_entrada_treino import DatasetPersonalizado

class TreinadorLSTM:
    def __init__(self,
                dataset_path=path_keypoints2csv,
                featrures_frame=34, neuronios_ocultos=128,
                num_epocas=10,
                batch_size=4,
                bidirecional=False,
                num_camadas=1,
                lr=0.001,
                salvar_em= os.path.join(path_modelo_treinado,"smartfight_lstm.h5")
                ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conjunto_amostras = DatasetPersonalizado(dataset_csv_path=dataset_path)
        self.loader = DataLoader(self.conjunto_amostras,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=self.conjunto_amostras.collate_pad_batch
                                )
        
        self.modelo_LSTM = ClassificadorLSTM(
            len_featrures_frames=featrures_frame,
            len_neuronios_ocultos=neuronios_ocultos,
            len_tipos_golpes=len(self.conjunto_amostras.nome_golpes_classe),
            bidirecional=bidirecional,
            len_camadas=num_camadas
        ).to(self.device)

        self.criteria = nn.CrossEntropyLoss()
        self.otimizador = optim.Adam(self.modelo_LSTM.parameters(), lr=lr)

        self.num_epocas = num_epocas
        self.salvar_em = salvar_em
        self.perdas_epoca = []

    def treinar_modelo(self, plotar_grafico_perdas=False):
        print("MODELO LSTM: \n\t", self.modelo_LSTM)
        print(f"🏷️  QUANTIDADE DE AMOSTRAS ({len(self.conjunto_amostras)})")
        print(f"📁 {self.conjunto_amostras.nome_golpes_classe} | TAMANHO: {len(self.conjunto_amostras.nome_golpes_classe)}")

        tempo_inicial = time()

        for epoca in range(self.num_epocas):
            self.modelo_LSTM.train()
            perda_total = 0.0

            for frames_keypoints, rotulos_golpes, real_frames_batch in self.loader:
                frames_keypoints, rotulos_golpes = frames_keypoints.to(self.device), rotulos_golpes.to(self.device)

                self.otimizador.zero_grad() # zerar grardientes acumulados
                saida = self.modelo_LSTM(frames_keypoints, real_frames_batch)
                perda = self.criteria(saida, rotulos_golpes)
                perda.backward() # calcular gradientes
                self.otimizador.step() # atualizar pesos com grardss

                perda_total += perda.item()

            perda_media = perda_total / len(self.loader)
            self.perdas_epoca.append(perda_media)
            print(f"🔁  [ÉPOCA: {epoca+1}/{self.num_epocas}] |\n [PERDA MÉDIA: {perda_media:.4f}]")

        if plotar_grafico_perdas:
            self._plotar_grafico_perdas()

        os.makedirs(os.path.dirname(self.salvar_em), exist_ok=True)
        gerar_init(caminho_pasta=path_modelo_treinado)

        torch.save(self.modelo_LSTM.state_dict(), self.salvar_em)
        print(f"\n✅ [SALVO: {self.salvar_em}]")
        
        tempo_final = time()
        print(f"\n⏰ [DURAÇÃO DE TREINAMENTO: {tempo_final - tempo_inicial:.2f}]")

    def _plotar_grafico_perdas(self):
        plt.plot(range(1, self.num_epocas + 1), self.perdas_epoca, marker='o', label='Loss')
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.title("Evolução da Loss durante o treinamento")
        plt.grid(True)
        plt.legend()
        plt.show()
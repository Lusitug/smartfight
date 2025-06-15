from utils.caminhos import Caminhos
from utils.globais import Globais
from ultralytics import YOLO
import os 
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


class DefinirGuarda:
    def __init__(self):
        self.saida_csv_guarda = Caminhos.saida_csv_guarda # pre-definido
        self.saida_plot_guarda = Caminhos.saida_plot_guarda # pre-definido
        self.saida_media_guarda = Caminhos.saida_media_guarda # pre-definido
        self.data_guarda = Caminhos.dataset_guarda # pre-definido

        self.yolo_run = YOLO(Caminhos.path_yolo) # pre-definido
        
        self._gerar_dirs_saidas()

    def _gerar_dirs_saidas(self):
        print("INICIALIZANDO PROCESSO DE DEFINIR MEDIA DA GUARDA")
        pastas = [
            self.saida_csv_guarda,
            self.saida_plot_guarda,
            self.saida_media_guarda,
        ]
        for pasta in pastas:
            os.makedirs(pasta, exist_ok=True)
            Globais.gerar_init(pasta)

        print("[PASTAS GERADAS]")

    def converter_dataset_guarda_em_csv(self):   
        for base_data_guarda, _, imagens in os.walk(self.data_guarda):
            for imagem in imagens:
                self._verificar_imagem_valida(base_data_guarda, imagem)
        print("Ler dataset-guarda -> converter csv\nProcessamento [1] completo.")


    def _verificar_imagem_valida(self, base_data_guarda, arq_imagem):
        if not arq_imagem.lower().endswith(('.png', '.jpg','.jpeg',)):
            return
        
        caminho_imagem = os.path.join(base_data_guarda, arq_imagem)
        nome_base = os.path.splitext(arq_imagem)[0]
        caminho_csv = os.path.join(
            self.saida_csv_guarda, 
            nome_base+"_kps.csv"
        )

        if os.path.exists(caminho_csv):
            print(f"\n⏩ [PULANDO: {nome_base}_kps.csv - CSV já existe]")
            return
        
        self._yolo_run(
            caminho_img=caminho_imagem,
            caminho_csv=caminho_csv,
            nome_base=nome_base )
    
    def _yolo_run(self, caminho_img, nome_base, caminho_csv):
        img = Image.open(caminho_img)
        resultados = self.yolo_run(img)

        self._salvar_imagem_plotada(resultados, nome_base)
        self._salvar_keypoints_em_csv(resultados, caminho_csv)

    def _salvar_imagem_plotada(self, resultados, nome_base):
        caminho_saida = os.path.join(self.saida_plot_guarda, nome_base + "_pred.jpg")
        
        resultados[0].save(filename=caminho_saida)
    
    def _salvar_keypoints_em_csv(self, resultados, caminho_csv):
        for resultado in resultados:
            if resultado.keypoints is not None:
                with open(caminho_csv, 'w') as f:
                    for keypoint in resultado.keypoints.xyn.cpu().numpy():
                        f.write(','.join(map(str, keypoint.flatten())) + '\n')

    def _carregar_csv(self, path):
        return np.loadtxt(path, delimiter=',')

    def criar_modelo_media_guarda(self):
        guardas_csv = glob(os.path.join(self.saida_csv_guarda, '*.csv'))

        if not guardas_csv:
            print(f"Nenhum arquivo CSV encontrado em {self.saida_csv_guarda}")
            return
        
        lista_keypoints = []
        for guarda in guardas_csv:
            keypoints = self._carregar_csv(guarda)
            lista_keypoints.append(keypoints)

        keypoints_empilhados = np.stack(lista_keypoints)
        modelo_guarda_media = np.mean(keypoints_empilhados, axis=0)
        saida_modelo_guarda_media = os.path.join(self.saida_media_guarda, 'guarda_media.npy')
        np.save(saida_modelo_guarda_media, modelo_guarda_media)

        print(f"Modelo de guarda salvo em: {self.saida_media_guarda}")
        print(f"modelo de guarda: {modelo_guarda_media}")
        print(f"Shape do modelo de guarda: {modelo_guarda_media.shape}")

        print("Ler dataset-guarda convertido em csv -> calcular media e criar modelo\nProcessamento [2] completo.")

    def plotar_modelo(self, modo='importantes'):
        """
        Plota o modelo de guarda.
        modo: 'importantes' para mostrar só as articulações importantes,
            'todos' para mostrar todos os pontos.
        """
        caminho_modelo = os.path.join(self.saida_media_guarda, 'guarda_media.npy')

        if not os.path.exists(caminho_modelo):
            print(f"❌ Modelo não encontrado em {caminho_modelo}")
            return
        
        modelo_guarda = np.load(caminho_modelo)

        if modo == 'importantes':
            # Articulações importantes (índices 5 a 12)
            articulacoes_importantes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            
            def extrair_articulacoes_importantes(vetor_kps):
                pontos = []
                for idx in articulacoes_importantes:
                    x = vetor_kps[2 * idx]
                    y = vetor_kps[2 * idx + 1]
                    pontos.extend([x, y])
                return np.array(pontos)

            modelo_importante = extrair_articulacoes_importantes(modelo_guarda)
            x_imp = modelo_importante[0::2]
            y_imp = modelo_importante[1::2]
            print(modelo_importante)

            nomes_articulacoes_importantes = [
                'Ombro Esq', 'Ombro Dir',
                'Cotovelo Esq', 'Cotovelo Dir',
                'Pulso Esq', 'Pulso Dir',
                'Quadril Esq', 'Quadril Dir',
                'Joelho Esq', 'Joelho Dir',
                'Tornozelo Esq', 'Tornozelo Dir'
            ]

            conexoes = [
                (0, 1), (0, 2), (2, 4),
                (1, 3), (3, 5),
                (0, 6), (1, 7), (6, 7),
                (6, 8) , (8, 10),
                (7, 9) , (9, 11)
            ]

            plt.figure(figsize=(4, 6))
            plt.scatter(x_imp, y_imp, c='blue', s=70)

            for i, nome in enumerate(nomes_articulacoes_importantes):
                plt.text(x_imp[i], y_imp[i], nome, fontsize=9, ha='right', va='bottom')

            for idx1, idx2 in conexoes:
                plt.plot([x_imp[idx1], x_imp[idx2]], [y_imp[idx1], y_imp[idx2]], 'k-', linewidth=2)

            plt.title("Modelo de Guarda - Articulações Importantes (Média)")
        else:
            # Exibe todos os pontos
            n_pontos = len(modelo_guarda) // 2
            x_all = modelo_guarda[0::2]
            y_all = modelo_guarda[1::2]
            plt.figure(figsize=(4, 6))
            plt.scatter(x_all, y_all, c='red', s=50)
            for i in range(n_pontos):
                plt.text(x_all[i], y_all[i], f"P{i}", fontsize=8, ha='right', va='bottom')
            plt.title("Modelo de Guarda - Todos os Pontos (Média)")

        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Abrir modelo e vizualizar informações de interesse\nProcessamento [3] completo.")
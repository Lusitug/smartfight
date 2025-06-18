# import os
# import uvicorn
from time import time
# from fastapi import FastAPI
from utils.caminhos import Caminhos
# from ml.lstm.teinar_lstm import TreinadorLSTM
# from api.inferencia_api import app as inferencia_app
# from ml.inferencia.inferencia_lstm import InferenciaLSTM
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
# from analise.media_guarda import DefinirGuarda
from pose.preprocessamento.data_augumentation import DataAugumentation
# from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# app = FastAPI()

# app.mount("/", inferencia_app)
 # uvicorn main:app --reload

if __name__ == "__main__":
    tempo_inicial = time()
    
    ## teste extracao keypoints

    # guarda = DefinirGuarda()

    # guarda.converter_dataset_guarda_em_csv()
    # guarda.criar_modelo_media_guarda()
    # guarda.plotar_modelo()

    
    # extrator = ExtracaoKeypoints( dataset_path= Caminhos.path_videos2estimate,
    #                              modelo_yolo_path=Caminhos.path_yolo,
    #                              saida_csv_path=Caminhos.path_keypoints2csv)
    
    # extrator.extrair_keypoints_dataset()



    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    # treinador = TreinadorLSTM(neuronios_ocultos=128,
    #                         num_epocas=70, 
    #                         batch_size=3,
    #                         bidirecional=True,
    #                         num_camadas=3)
    
    # treinador.treinar_modelo(plotar_grafico_perdas=True)


    ## teste inferencia


    # extrair_kps = ExtracaoKeypoints(modelo_yolo_path=Caminhos.path_yolo, 
    #                                 dataset_path="",
    #                                 saida_csv_path="")
    
    # keypoints = extrair_kps.processar_video(Caminhos.path_teste1)
    # salvar_csv = ConverterKeypointsCSV()

    # salvar_csv.keypoints2csv(path_saida=os.path.join(Caminhos.path_teste0, "soco.csv"),
    #                          lista_keypoints_video=keypoints)
    
    # inferencia = InferenciaLSTM(bidirecional=True, num_camadas=3)
    
    # saida = inferencia.prever(path_csv=Caminhos.path_teste2)
    # print(saida)

    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

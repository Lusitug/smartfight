import os
import uvicorn
from time import time
from fastapi import FastAPI
# from utils.utilidades import Utilidades
# from ml.lstm.teinar_lstm import TreinadorLSTM
from api.inferencia_api import app as inferencia_app
# from ml.inferencia.inferencia_lstm import InferenciaLSTM
# from pose.extracao.extracao_keypoints import ExtracaoKeypoints
# from pose.preprocessamento.data_augumentation import DataAugumentation
# from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

app.mount("/", inferencia_app)
 # uvicorn main:app --reload

if __name__ == "__main__":
    tempo_inicial = time()
    
    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    # extrator = ExtracaoKeypoints( dataset_path= Utilidades.path_videos2estimate,
    #                              modelo_yolo_path=Utilidades.path_yolo,
    #                              saida_csv_path=Utilidades.path_keypoints2csv)
    
    # extrator.extrair_keypoints_dataset()
    
    # treinador = TreinadorLSTM(neuronios_ocultos=128,
    #                         num_epocas=70, 
    #                         batch_size=3,
    #                         bidirecional=True,
    #                         num_camadas=3)
    
    # treinador.treinar_modelo(plotar_grafico_perdas=True)


    # extrair_kps = ExtracaoKeypoints(modelo_yolo_path=Utilidades.path_yolo, 
    #                                 dataset_path="",
    #                                 saida_csv_path="")
    
    # keypoints = extrair_kps.processar_video(Utilidades.path_teste1)
    # salvar_csv = ConverterKeypointsCSV()

    # salvar_csv.keypoints2csv(path_saida=os.path.join(Utilidades.path_teste0, "soco.csv"),
    #                          lista_keypoints_video=keypoints)
    
    # inferencia = InferenciaLSTM(bidirecional=True, num_camadas=3)
    
    # saida = inferencia.prever(path_csv=Utilidades.path_teste2)
    # print(saida)

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

import os
from time import time

from ml.inferencia.inferencia_lstm import InferenciaLSTM
from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
# from ml.lstm.teinar_lstm import TreinadorLSTM
from utils.utilidades import path_keypoints2csv,path_teste0,  path_teste1 , path_teste2,path_yolo, path_videos2estimate, path_modelo_treinado
# from pose.preprocessamento.data_augumentation import DataAugumentation

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    tempo_inicial = time()
    
    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    # extrator = ExtracaoKeypoints( dataset_path= path_videos2estimate,
    #                              modelo_yolo_path=path_yolo,
    #                              saida_csv_path=path_keypoints2csv)
    
    # extrator.extrair_keypoints_dataset()
    
    # treinador = TreinadorLSTM(neuronios_ocultos=128,
    #                         num_epocas=70, 
    #                         batch_size=3,
    #                         bidirecional=True,
    #                         num_camadas=3)
    
    # treinador.treinar_modelo(plotar_grafico_perdas=True)


    extrair_kps = ExtracaoKeypoints(modelo_yolo_path=path_yolo, 
                                    dataset_path="",
                                    saida_csv_path="")
    
    keypoints = extrair_kps.processar_video(path_teste1)
    salvar_csv = ConverterKeypointsCSV()

    salvar_csv.keypoints2csv(path_saida=os.path.join(path_teste0, "soco.csv"),
                             lista_keypoints_video=keypoints)
    
    inferencia = InferenciaLSTM()
    
    saida =inferencia.prever(path_csv=path_teste2)
    print(saida)
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

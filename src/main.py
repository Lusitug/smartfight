from time import time
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from utils.utilidades import path_keypoints2csv, path_yolo,path_videos2estimate
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from pose.preprocessamento.data_augumentation import DataAugumentation

if __name__ == "__main__":
    tempo_inicial = time()
    
    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    extrator = ExtracaoKeypoints( dataset_path= path_videos2estimate,
                                 modelo_yolo_path=path_yolo,
                                 saida_csv_path=path_keypoints2csv)
    
    extrator.extrair_keypoints_dataset()
    
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

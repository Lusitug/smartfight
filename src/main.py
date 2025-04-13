from time import time
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from utils.utilidades import path_keypoints2csv, path_yolo,path_videos2estimate
# from pose.preprocessamento.data_augumentation import DataAugumentation
import torch
from torch.utils.data import DataLoader
from ml.dataset_custom import DatasetPersonalizado
import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    tempo_inicial = time()
    
    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    # extrator = ExtracaoKeypoints( dataset_path= path_videos2estimate,
                                #  modelo_yolo_path=path_yolo,
                                #  saida_csv_path=path_keypoints2csv)
    
    # extrator.extrair_keypoints_dataset()
    
    dataset = DatasetPersonalizado(dataset_csv_path=path_keypoints2csv)
    print(f"TOTAL AMOSTRAS {len(dataset)}")
    
    x, y = dataset[8]

    print(f"\n shape do input (x): {x.shape}")  # Ex: torch.Size([28, 34])
    # print(f"\n valor (x): {x}") (x,y)'s em formato [x1, y1, x2, y2, x3, y3]
    print(f"\n rotulo (y): {y}") 
    nome_classe = dataset.indice_nome(y.item())
    print(f" rotulo nome Classe: {nome_classe}")

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for i, (x_batch, y_batch) in enumerate(loader):
            print(f"\n🔁 Lote {i+1}")
            print("Shape do x_batch:", x_batch.shape)
            print("Rótulos:", y_batch)

            # Suporte para qualquer batch_size (mesmo >1)
            for j in range(len(y_batch)):
                rotulo = y_batch[j].item()
                nome_classe = dataset.indice_nome(rotulo)
                print(f"Amostra {j+1}: rótulo={rotulo} -> Classe: {nome_classe}")

            break  # Apenas o primeiro lote
    
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

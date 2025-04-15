from time import time
# from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from utils.utilidades import path_keypoints2csv, path_yolo,path_videos2estimate
# from pose.preprocessamento.data_augumentation import DataAugumentation
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.preparar_entrada.preparar_entrada import DatasetPersonalizado
from ml.lstm.classificador_lstm import ClassificadorLSTM
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    tempo_inicial = time()
    
    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    
    # extrator = ExtracaoKeypoints( dataset_path= path_videos2estimate,
    #                              modelo_yolo_path=path_yolo,
    #                              saida_csv_path=path_keypoints2csv)
    
    # extrator.extrair_keypoints_dataset()
    
    dataset = DatasetPersonalizado(dataset_csv_path=path_keypoints2csv)
    print(f"TOTAL AMOSTRAS {len(dataset)}")
    # x, y = dataset[6]

    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_pad_batch)
    
    entrada = 34
    oculto = 128
    tamanho_saida = len(dataset.golpes_classe)
    print(tamanho_saida)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo_LSTM = ClassificadorLSTM(
        len_featrures_frames=entrada,
        len_neuronios_ocultos=oculto,
        len_tipos_golpes=tamanho_saida
    ).to(device=device)
    print("MODELO LSTM: ", modelo_LSTM)
    criteria = nn.CrossEntropyLoss()
    optimazodor = optim.Adam(modelo_LSTM.parameters(), lr=0.001)
    
    for batch_x, batch_y, real_batch_len in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        print(batch_x)
        print(batch_y)
        optimazodor.zero_grad()
        print("OPTM:", optimazodor)
        saida = modelo_LSTM(batch_x, real_batch_len)
        print("SAIDA:", saida)
        loss = criteria(saida, batch_y)
        loss.backward()
        optimazodor.step()
        
        print("loss:", loss)
        break

    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

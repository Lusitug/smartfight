import os
import pandas as pd

class Caminhos:
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'
    path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

    path_keypoints2csv =  os.path.join("src", "saidas", "keypoints_extraidos")

    path_videos2estimate = os.path.join("src", "dataset")
    dataset_csv = os.path.join("src", "DATASET_CSV")

    path_modelo_treinado = os.path.join("src", "saidas", "modelos_ml")
    
    teste_periodiciodade11 = os.path.join("src", "direto_EyHSY2mj.csv") #curto
    teste_periodiciodade12 = os.path.join("src", "direto-seg_H3GhvS2O.csv") #curto

    teste_periodiciodade7 = os.path.join("src", "direto-seg1_acL4ciwa.csv") #longo
    teste_periodiciodade8 = os.path.join("src", "dataset", "Direto", "direto-seg1_acL4ciwa.mov")
    
    teste_periodiciodade9 = os.path.join("src", "c.csv")#curt
    teste_periodiciodade10 = os.path.join("src", "CC.csv") #longo
    

    teste_periodiciodade13 =  os.path.join("src", "joelho-direito1_XN53OaK9.csv") #curtlongo
    teste_periodiciodade14 =  os.path.join("src", "joelho.csv") #curt

    teste_periodiciodade15 =  os.path.join("src", "direto4_cxPxW3KR.csv") #longo







    dataset_guarda = os.path.join("src", "data_guarda")
    saida_csv_guarda = os.path.join("src", "saidas", "definicao_guarda")
    saida_media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media")
    
    media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media", "guarda_media.npy")
    saida_plot_guarda = os.path.join("src", "saidas", "definicao_guarda", "plot")

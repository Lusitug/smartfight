import os
import pandas as pd

class Caminhos:
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'
    path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

    path_keypoints2csv =  os.path.join("src", "saidas", "keypoints_extraidos")

    path_videos2estimate = os.path.join("src", "dataset")

    path_modelo_treinado = os.path.join("src", "saidas", "modelos_ml")

    teste_periodiciodade7 = os.path.join("src", "direto-seg1_acL4ciwa.csv")
    teste_periodiciodade8 = os.path.join("src", "dataset", "Direto", "direto-seg1_acL4ciwa.mov")
    
    
    dataset_guarda = os.path.join("src", "data_guarda")
    saida_csv_guarda = os.path.join("src", "saida", "definicao_guarda")
    saida_media_guarda = os.path.join("src", "saida", "definicao_guarda", "media")
    media_guarda = os.path.join("src", "saida", "definicao_guarda", "media", "guarda_media.npy")
    saida_plot_guarda = os.path.join("src", "saida", "definicao_guarda", "plot")

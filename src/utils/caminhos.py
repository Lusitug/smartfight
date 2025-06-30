import os
import pandas as pd

class Caminhos:
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'
    path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

    path_keypoints2csv =  os.path.join("src", "saidas", "keypoints_extraidos")

    path_videos2estimate = os.path.join("src", "dataset")

    dataset_padrao = os.path.join("src", "DATASET_PADRAO")
    dataset_csv = os.path.join("src", "DATASET_CSV")

    dataset_loop = os.path.join("src", "DATASET_LOOP")
    dataset_loop_csv = os.path.join("src", "DATASET_LOOP_CSV")

    path_modelo_treinado = os.path.join("src", "saidas", "modelos_ml")
    
    videokk=  os.path.join("src", "arqs_testes", "videos_testes", "chute-alto-direito-canhoto.mp4")
    

    # com loop
    teste_peridiocidade_cloop = os.path.join("src", "jab-canhoto_loop.csv")
    teste_peridiocidade_cloop1 = os.path.join("src", "jab-destro_loop.csv")

    # sem loop
    teste_peridiocidade_sloop = os.path.join("src", "jab1_qXkWWJe2.csv")






    # teste definir posição de guarda

    dataset_guarda = os.path.join("src", "data_guarda")
    saida_csv_guarda = os.path.join("src", "saidas", "definicao_guarda")
    saida_media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media")
    
    media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media", "guarda_media.npy")
    saida_plot_guarda = os.path.join("src", "saidas", "definicao_guarda", "plot")

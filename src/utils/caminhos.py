import os
import pandas as pd

class Caminhos:
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'
    path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

    path_keypoints2csv =  os.path.join("src", "saidas", "keypoints_extraidos")

    path_videos2estimate = os.path.join("src", "dataset")
    dataset_csv = os.path.join("src", "DATASET_CSV")

    dataset_loop_csv = os.path.join("src", "DATASET_LOOP_CSV")

    path_modelo_treinado = os.path.join("src", "saidas", "modelos_ml")
    

    # corrigir caminhos dos testes de peridiocidade, acrescentar "csvs_testes entre src e <nome_arquivo>"
    
    teste_periodiciodade11 = os.path.join("src", "direto_EyHSY2mj.csv") #curto direto rw
    teste_periodiciodade12 = os.path.join("src", "direto-seg_H3GhvS2O.csv") #curto direto lw

    teste_periodiciodade7 = os.path.join("src", "direto-seg1_acL4ciwa.csv") #longo
    teste_periodiciodade8 = os.path.join("src", "dataset", "Direto", "direto-seg1_acL4ciwa.mov") # so pra nao quebrar anima_keypoints.py
    
    teste_periodiciodade9 = os.path.join("src", "c.csv")# curt  chute alto de esquerda
    teste_periodiciodade10 = os.path.join("src", "CC.csv") #longo
    

    teste_periodiciodade13 =  os.path.join("src", "joelho-direito1_XN53OaK9.csv") #longo lk
    teste_periodiciodade17 =  os.path.join("src", "joelho-direito.csv") #longo rk
    teste_periodiciodade14 =  os.path.join("src", "joelho.csv") #curt   rk joelho direito

    teste_periodiciodade15 =  os.path.join("src", "direto4_cxPxW3KR.csv") #longo

    teste_periodiciodade16 =  os.path.join("src", "cotovelo-circular-direito.csv") #longo

    teste_periodiciodade18 =  os.path.join("src", "cruzado-d_Hcz7pGmj.csv") #curto 
    teste_periodiciodade19 =  os.path.join("src", "jab_jMplKElc.csv") #curto  
    teste_periodiciodade20 =  os.path.join("src", "gancho-e_cWKREa8Z.csv") #curto
    # 14 rk  18 rw 19 lw 20 lw

    # teste peridiocidade com loop

    teste_periodiciodade_loop1 = os.path.join("src", "arqs_testes", "csvs_testes", "chute-frontal-direito-destro_loop.csv") #curto para loop

    teste_periodiciodade_loop2 = os.path.join("src", "arqs_testes", "csvs_testes", "chute-frontal-esquerdo-canhoto_loop.csv") #curto para loop

    teste_periodiciodade_loop3 = os.path.join("src", "arqs_testes", "csvs_testes", "chute-frontal-esquerdo-canhoto-normal.csv") #curto para loop


    # teste definir posição de guarda

    dataset_guarda = os.path.join("src", "data_guarda")
    saida_csv_guarda = os.path.join("src", "saidas", "definicao_guarda")
    saida_media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media")
    
    media_guarda = os.path.join("src", "saidas", "definicao_guarda", "media", "guarda_media.npy")
    saida_plot_guarda = os.path.join("src", "saidas", "definicao_guarda", "plot")

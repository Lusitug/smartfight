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

    
    extrator = ExtracaoKeypoints( dataset_path= Caminhos.path_videos2estimate,
                                 modelo_yolo_path=Caminhos.path_yolo,
                                 saida_csv_path=Caminhos.path_keypoints2csv)
    
    extrator.extrair_keypoints_dataset()



    # augumentador = DataAugumentation()

    # augumentador.aplicar_augumentation_dataset()
    

####################################

    # ( TESTES PAUSADOS ) #       # ( TESTES PAUSADOS ) #        # ( TESTES PAUSADOS ) #       # ( TESTES PAUSADOS ) #       # ( TESTES PAUSADOS ) #

# classificação dtw-knn (testar outros ml)

# dtw_knn_classfy = ClassificadorDTW_KNN()
# dtw_knn_classfy.carregar_dataset()
# # dtw_knn_classfy.fit()

# x_train, x_test, y_train, y_test = dtw_knn_classfy.train_test()
# dtw_knn_classfy.fit(x_train, y_train)

# # print("x_train", type(x_train), len(x_train)) # print("y_train",  type(y_train), len(y_train)) # print("x_test", type(x_test), len(x_test)) # print("y_test", type(y_test) , len(y_test))

# print("[DEBUG] x_train[0] type:", type(x_train[0]), "shape:", getattr(x_train[0], "shape", "sem shape"))
# print("[DEBUG] x_test[0] type:", type(x_test[0]), "shape:", getattr(x_test[0], "shape", "sem shape"))

# # avaliação

# y_pred = dtw_knn_classfy.predict(x_train=x_train, y_train=y_train, x_test=x_test)
# print(dtw_knn_classfy.evualuate(x_test=x_test, y_test=y_test,y_pred=y_pred))

# # predição

# # 2. carregar vídeo externo
# golpe = pd.read_csv(Caminhos.teste_periodiciodade11)
# golpe_conver = Globais.converter_array32(golpe)

# segmentos_teste = Globais.segmentar_com_janela_sliding(
#     golpe_conver,
#     janela=len(golpe_conver),
#     passo=len(golpe_conver)
# )

# # 3. Predição
# if not segmentos_teste:
#     print("⚠️ O vídeo externo é muito curto ou a janela está maior que o número de frames.")
# else:
#     y_pred_externo = dtw_knn_classfy.predict(
#         x_train=dtw_knn_classfy.x_train,
#         y_train=dtw_knn_classfy.y_train,
#         x_test=segmentos_teste
#     )
#     for i, pred in enumerate(y_pred_externo):
#         nome_classe = dtw_knn_classfy.nomes_golpes_classe[pred]
#         print(f"Predito como : {nome_classe}")

    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from ml.inferencia.inferencia_lstm import InferenciaLSTM
from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from utils.utilidades import Utilidades

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inferencia = InferenciaLSTM(bidirecional=True, num_camadas=3)

extrator = ExtracaoKeypoints(modelo_yolo_path=Utilidades.path_yolo,
                             dataset_path="", saida_csv_path="")

conversor = ConverterKeypointsCSV()

# teste de inferencia com api integrarda
@app.post("/inferencia-video")
async def inferencia_video(file: UploadFile = File(...)):

    video_path = os.path.join(Utilidades.path_teste0, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("[ARQUIVO EM ANALISE: \n\t TEMPORARIAMENTE SALVO: ", video_path,"]")

    csv_path = os.path.join(Utilidades.path_teste0, "soco.csv")

    try:
        keypoints = extrator.processar_video(path_videos=video_path)
        conversor.keypoints2csv(lista_keypoints_video=keypoints, path_saida=csv_path)
        classe_prevista = inferencia.prever(path_csv=Utilidades.path_teste2)
        print(classe_prevista)

        return {"classe_prevista": classe_prevista}
    
    except Exception as e:
        print("[ERRO: ", e,"]")
        return {"status": "erro", "mensagem": str(e)}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
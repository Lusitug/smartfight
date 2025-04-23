import shutil
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
from ml.inferencia.inferencia_lstm import InferenciaLSTM
# from pose.conversao.converter_keypoints_csv import ConverterKeypointsCSV
from pose.extracao.extracao_keypoints import ExtracaoKeypoints
from utils.utilidades import Utilidades

# classe teste para receber videos via upload pelo navegador

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

# conversor = ConverterKeypointsCSV()

# teste de inferencia com api integrarda
@app.post("/inferencia-video")
async def inferencia_video(file: UploadFile = File(...)):
    print("Endpoint /inferencia-video chamado")
    try:
        video_path = os.path.join(Utilidades.path_teste0, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("[ARQUIVO EM ANALISE: \n\t TEMPORARIAMENTE SALVO: ", video_path,"]")

        # csv_path = os.path.join(Utilidades.path_teste0, "soco.csv")

        keypoints = extrator.processar_video(path_videos=video_path)
        print("len kps: ", len(keypoints))
        
        # conversor.keypoints2csv(lista_keypoints_video=keypoints, path_saida=csv_path)
        classe_prevista = inferencia.prever_golpes_em_memoria(keypoints)
        print(classe_prevista)

        return {"classe_prevista": classe_prevista}
    
    except Exception as e:
        print("[ERRO: ", e,"]")
        return {"status": "erro", "mensagem": str(e)}
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as cleanup_error:
            print("[ERRO AO REMOVER TEMPORÁRIOS]:", cleanup_error)

            
    """
# não é o foco do projeto, mas é um teste de inferencia em tempo real
@app.websocket("/inferencia-tempo-real")
async def inferencia_tempo_real(websocket: WebSocket):   
    print(" Tentando conectar WebSocket ")
    await websocket.accept()
    print("WebSocket conectado")

    try:
        buffer = []
        while True:
            data = await websocket.receive_bytes()
            print(f"[INFO] Dados recebidos: {len(data)} bytes")

            frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                  cv2.IMREAD_COLOR)
            if frame is None:
                print("[ERRO] Frame inválido recebido")
                continue

            keypoints = extrator.extrair_keypoints(frame)
            if not keypoints:
                print("[ERRO] Nenhum keypoint detectado no frame")
                continue
            
            buffer.append(keypoints)
            
            classe_prevista = inferencia.prever_golpes_em_memoria(buffer)
            print("Classe prevista: ", classe_prevista)
            await websocket.send_json({"classe_prevista": classe_prevista})

    except Exception as e:
        print("[ERRO: ", e,"]")
        await websocket.send_text(f"Erro: {str(e)}")
    finally:
        await websocket.close()
        print("WebSocket desconectado")
"""
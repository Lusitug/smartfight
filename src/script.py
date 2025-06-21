from utils.caminhos import Caminhos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from dtw import dtw, dtwPlot
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw as dtw2

# pre processamento
def ast_func(points):
    return ast.literal_eval(points)

def converter_frame_vetor(frame: pd.Series) -> np.ndarray:
    vetor = []
    for coluna in frame.index:
        if coluna == "frame":
            continue
        valor = frame[coluna]

        if isinstance(valor, str):
            try:
                ponto = ast.literal_eval(valor)
                if isinstance(ponto, (list, tuple)) and len(ponto) == 2:
                    vetor.extend(ponto)
                else:
                    # Caso seja uma lista malformada
                    vetor.extend([0.0, 0.0])
            except Exception:
                vetor.extend([0.0, 0.0])
        else:
            vetor.extend([0.0, 0.0])

    return np.array(vetor, dtype=np.float32)

def converter_array32(df: pd.DataFrame) -> np.ndarray:
    vetores = df.apply(converter_frame_vetor, axis=1).values.tolist()
    return np.array(vetores, dtype=np.float32)

# df = pd.read_csv(Caminhos.teste_periodiciodade10)
# df = pd.read_csv(Caminhos.teste_periodiciodade7)
df = pd.read_csv(Caminhos.teste_periodiciodade11)
articulação_l_w = np.array([ast_func(point) for point in df["r_w"].values])
# print(articulação_l_w.shape) # os frames e x,y
articulação_l_w_1d = np.linalg.norm(articulação_l_w, axis=1) if articulação_l_w.ndim > 1 else articulação_l_w
# print(articulação_l_w_1d.shape) # so os frames

# analisar_periodos(df, 15)

df2 = pd.read_csv(Caminhos.teste_periodiciodade12)
# df2 = pd.read_csv(Caminhos.teste_periodiciodade11)
articulação_r_w = np.array([ast_func(point) for point in df2["l_w"].values])
# print(articulação_r_w.shape) # os frames e x,y
articulação_r_w_1d = np.linalg.norm(articulação_r_w, axis=1) if articulação_r_w.ndim > 1 else articulação_r_w
# print(articulação_r_w_1d.shape) # so os frames

# df = converter_array32(df) # soco do dataset / as vezes outro soco curto
# df2 = converter_array32(df2) # soco curto

# print(len(articulação_l_w_1d))

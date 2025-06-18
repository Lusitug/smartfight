import pandas as pd
import numpy as np
import ast
from utils.caminhos import Caminhos
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def carregar_csv_golpe() -> pd.DataFrame:
    df = pd.read_csv(Caminhos.teste_periodiciodade7)
    return df


def carregar_modelo_guarda() -> np.ndarray:
    m_guarda = np.load(Caminhos.media_guarda)
    return m_guarda

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


def converter_array34(df: pd.DataFrame) -> np.ndarray:
    vetores = df.apply(converter_frame_vetor, axis=1).values.tolist()
    return np.array(vetores, dtype=np.float32)

def distancia_euclidiana(vetores: np.ndarray, modelo: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vetores - modelo, axis=1)


def salvar_distancias_txt(distancias: np.ndarray, path_saida: str):
    with open(path_saida, 'w') as f:
        for d in distancias:
            f.write(f"{d}\n")

def plotar_distancias(distancias: np.ndarray):
    plt.figure(figsize=(12, 4))
    plt.plot(distancias, label="Distância à guarda", color="blue")
    plt.xlabel("Frame")
    plt.ylabel("Distância euclidiana")
    plt.title("analise de periodos comparando com a media da guarda")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


df = carregar_csv_golpe()

df_transform = converter_array34(df=df)

modelo = carregar_modelo_guarda()

distancias = distancia_euclidiana(
                    vetores=df_transform,
                    modelo=modelo)

# salvar_distancias_txt(distancias, "distancias.txt")

plotar_distancias(distancias)

# eixo x - movimento lateral - pessoa se deslocando para os lados
# eixo y - movimento vertical-  pessoa se deslocando para cima/baixo (ex: pulo, agacho)


peaks, _ = find_peaks(distancias)
vales, _ = find_peaks(-distancias)

# coord_x é um vetor com valores de (por exemplo) r_w em X
# peaks, _ = find_peaks(coord_x, distance=20, prominence=0.03)
# vales, _ = find_peaks(-coord_x, distance=20, prominence=0.03)




"""
lixo


peaks, _ = find_peaks(distancias)
vales, _ = find_peaks(-distancias)

print("Índices dos picos:", peaks)
print("Índices dos vales:", vales)

coord_x = df['l_w'].apply(lambda s: ast.literal_eval(s)[0])  # X do quadril esquerdo
peaks, _ = find_peaks(coord_x, distance=10)  # Ajuste o parâmetro 'distance' conforme necessário
plt.plot(coord_x)
plt.plot(peaks, coord_x[peaks], "x")
plt.show()
"""
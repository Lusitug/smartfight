from gettext import find

from cv2 import norm
from utils.caminhos import Caminhos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dtw 
import ast
from scipy.signal import find_peaks

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

# fft
def analisar_periodos(keypoints: np.ndarray, ponto: int = 0):
    # fft
    idx_y = ponto * 2
    y = keypoints[:,idx_y]

    fft_result = np.fft.fft(y - np.mean(y))
    fft_freq = np.fft.fftfreq(len(y))

    print(f"resultado fft : {fft_result} ")
    print(f"frequencia associada ao valor fft : {fft_freq} ")
   
    # verifica o periodo em que há picos - a quantos frames há picos
    half = len(fft_freq) // 2
    fft_freq_pos = fft_freq[1:half]
    fft_result_pos = np.abs(fft_result[1:half])

    peak_idx = np.argmax(fft_result_pos)
    freq_dominante = fft_freq_pos[peak_idx]
    periodo = 1 / freq_dominante if freq_dominante != 0 else np.nan

    print(f"Frequência dominante: {freq_dominante:.5f} ciclos/frame")
    print(f"Período dominante: {periodo:.2f} frames por ciclo")

    # ff2
    plt.figure(figsize=(16, 8))  # Mais largo e alto

    # Gráfico 1: Sinal no tempo
    plt.subplot(2, 1, 1)  # 2 linhas, 1 coluna, posição 1
    plt.plot(y)
    plt.title("Sinal no tempo (coordenada Y do ponto)")
    plt.xticks(np.arange(0, len(y), 150))

    # Gráfico 2: FFT
    plt.subplot(2, 1, 2)  # 2 linhas, 1 coluna, posição 2
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title("FFT - Frequências dominantes")
    plt.xlabel("Frequência")
    plt.ylabel("Amplitude")
    plt.xticks(np.arange(0, 0.5, 0.01))

    plt.tight_layout()
    plt.show()
    """
    # fft0 e fft1
    # plt.figure(figsize=(14, 4))
    

    # plt.subplot(1, 2, 1)
    # plt.plot(y)
    # plt.title("Sinal no tempo (coordenada Y do ponto)")

    # plt.subplot(1, 2, 1)
    # plt.plot(y)
    # plt.title("Sinal no tempo (coordenada Y do ponto)")
    # plt.xticks(np.arange(0, len(y), 150))  # <-- Adiciona ticks a cada 150 frames
    
    # plt.subplot(1, 2, 2)
    # plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    # plt.title("FFT - Frequências dominantes")
    # plt.xlabel("Frequência")
    # plt.ylabel("Amplitude")

    # plt.subplot(1, 2, 2)
    # plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    # plt.title("FFT - Frequências dominantes")
    # plt.xlabel("Frequência")
    # plt.ylabel("Amplitude")
    # plt.xticks(np.arange(0, 0.5, 0.01))  # Ajuste conforme necessário para o seu eixo de frequência
   
    # plt.tight_layout()
    # plt.show()
    """


def testar_dtw(kps1: np.ndarray, kps2: np.ndarray):
    distance, cost_matrix, acc_cost_matrix, path = dtw(
        kps1,
        kps2,
        dist=lambda x, y: norm(x-y))


    print(f"Distância DTW: {distance:.2f}")

    plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')  # Caminho ótimo
    plt.title('Matriz de Custo Acumulado (DTW)')
    plt.xlabel('Seq 1')
    plt.ylabel('Seq 2')
    plt.show()

df = pd.read_csv(Caminhos.teste_periodiciodade7)
df2 = pd.read_csv(Caminhos.teste_periodiciodade11)

kps = np.array([ast_func(point) for point in df["l_w"].values])

# diffs = np.diff(kps, axis=0)
# speed = np.linalg.norm(diffs, axis=1)
# peaks, _ = find_peaks(speed, height=np.mean(speed)+2*np.std(speed))

df = converter_array32(df)
df2 = converter_array32(df2)

analisar_periodos(df, 9)
# testar_dtw(df, df2)

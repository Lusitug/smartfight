# import numpy as np
# import matplotlib.pyplot as plt
"""
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

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(y)
    plt.title("Sinal no tempo (coordenada Y do ponto)")
    
    plt.subplot(1, 2, 2)
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title("FFT - Frequências dominantes")
    plt.xlabel("Frequência")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
"""

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

    plt.figure(figsize=(16, 8))  # Mais largo e alto

    # Gráfico 1: Sinal no tempo
    plt.subplot(2, 1, 1)  # 2 linhas, 1 coluna, posição 1
    plt.plot(y)
    plt.title("Sinal no tempo (coordenada Y do ponto)")
    plt.xticks(np.arange(0, len(y), 150)) #  rotation=45,

   # Gráfico 2: FFT se pa nem preciso exibir esse graf
    plt.subplot(2, 1, 2)
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title("FFT - Frequências dominantes")
    plt.xlabel("Frequência")
    plt.ylabel("Amplitude")

    # Exibir ticks a cada 0.05 no eixo X
    plt.xticks(np.arange(0, 0.5, 0.05), ha='right') #  rotation=45,

    plt.tight_layout()
    plt.show()  



    
df = 0 # ler csv
# df = converter_array32(df) # soco do dataset / as vezes outro soco curto
analisar_periodos(df, 9)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.globais import Globais

#  converter df em float32,
# Analisa periodicidade usando FFT para um ponto específico
class AnalisarCiclosDataset:
    def __init__(self, golpe_csv: np.ndarray):
        self.df = golpe_csv
        self.df = Globais.converter_array32(self.df)
   
    # obtm os valores dos ciclos
    def verificar_periodo(self, idx_ponto: int= 0):
        idx_y = idx_ponto * 2
        y = self.df[:, idx_y]

        fft_resultado = np.fft.fft(y - np.mean(y))
        fft_frequencia = np.fft.fftfreq(len(y))

        half = len(fft_frequencia) // 2
        fft_frequencia_pos = fft_frequencia[1:half]
        fft_resultado_pos = np.abs(fft_resultado[1:half])

        idx_peak = np.argmax(fft_resultado_pos)
        frequencia_dominante = fft_frequencia_pos[idx_peak]
        periodo = 1 / frequencia_dominante if frequencia_dominante != 0 else np.nan

        # print(f"Frequência dominante: {frequencia_dominante:.5f} ciclos/frame")
        # print(f"Período dominante: {periodo:.2f} frames por ciclo")
        
        # Plot resultados
        plt.figure(figsize=(16, 8))
        
        # Sinal no tempo
        plt.subplot(2, 1, 1)
        plt.plot(y)
        plt.title("Sinal no tempo (coordenada Y do ponto)")
        plt.xticks(np.arange(0, len(y), 150))
        plt.grid(True)
        plt.legend()

        
        # FFT
        plt.subplot(2, 1, 2)
        plt.plot(fft_frequencia[:len(fft_frequencia)//2], np.abs(fft_resultado)[:len(fft_resultado)//2])
        plt.title("FFT - Frequências dominantes")
        plt.xlabel("Frequência")
        plt.ylabel("Amplitude")
        plt.xticks(np.arange(0, 0.5, 0.05), ha='right')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # ou outro valor, como 0.7
        plt.show()
        
        return {
            'dominant_frequency': frequencia_dominante,
            'period_frames': periodo,
            'fft_result': fft_resultado,
            'fft_frequencies': fft_frequencia
        }
    
    def verificar_periodo2(self, idx_ponto: int = 0):
        idx_y = idx_ponto * 2
        y = self.df[:, idx_y]

        fft_resultado = np.fft.fft(y - np.mean(y))
        fft_frequencia = np.fft.fftfreq(len(y))

        half = len(fft_frequencia) // 2
        fft_frequencia_pos = fft_frequencia[1:half]
        fft_resultado_pos = np.abs(fft_resultado[1:half])

        idx_peak = np.argmax(fft_resultado_pos)
        frequencia_dominante = fft_frequencia_pos[idx_peak]
        periodo = 1 / frequencia_dominante if frequencia_dominante != 0 else np.nan

        plt.figure(figsize=(16, 8))

        plt.subplot(2, 1, 1)
        plt.plot(y, label="Sinal Y")
        
        # Marcar os ciclos estimados no gráfico de tempo
        if not np.isnan(periodo):
            estimativas = np.arange(0, len(y), int(round(periodo)))
            for xc in estimativas:
                plt.axvline(x=xc, color='r', linestyle='--', alpha=0.4)
            plt.text(0.01, 0.95, f"Período estimado ≈ {periodo:.2f} frames", 
                    transform=plt.gca().transAxes, fontsize=12, color='red')

        plt.title("Sinal no tempo (coordenada Y do ponto)")
        plt.xticks(np.arange(0, len(y), 150))
        plt.grid(True)
        plt.legend()

        # Gráfico FFT
        plt.subplot(2, 1, 2)
        freqs = fft_frequencia[:len(fft_frequencia) // 2]
        amps = np.abs(fft_resultado)[:len(fft_resultado) // 2]
        plt.plot(freqs, amps, label="Espectro FFT")

        # Marca a frequência dominante
        freq_dom_x = frequencia_dominante
        freq_dom_y = np.abs(fft_resultado[int(len(y) * freq_dom_x)])
        plt.axvline(x=freq_dom_x, color='orange', linestyle='--', label='Freq. dominante')
        plt.text(freq_dom_x, max(amps)*0.7, f"{freq_dom_x:.3f} Hz", color='orange', fontsize=10)

        plt.title("FFT - Frequências dominantes")
        plt.xlabel("Frequência (ciclos/frame)")
        plt.ylabel("Amplitude")
        plt.xticks(np.arange(0, 0.5, 0.05))
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return {
            'dominant_frequency': frequencia_dominante,
            'period_frames': periodo,
            'fft_result': fft_resultado,
            'fft_frequencies': fft_frequencia
        }

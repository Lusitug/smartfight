import cv2
import numpy as np

class PreProcessamentoVideo:
    def __init__(self, largura_marca=115, altura_marca=30, tamanho_frame=(300,650)):
        self.largura_marca = largura_marca
        self.altura_marca = altura_marca
        self.tamanho_frame = tamanho_frame

    def reajustar_frame(self, frame):
        frame = cv2.resize(frame, self.tamanho_frame)
        frame = self.remover_marca_de_agua(frame)
        return frame

    def remover_marca_de_agua(self, frame): # quadradinho preto no canto inferior direito (alocado referente ao tamanho 300x650)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width - self.largura_marca, height - self.altura_marca), (width, height), (0, 0, 0), -1) # w 100 / h 25
        return frame

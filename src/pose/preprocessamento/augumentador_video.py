import numpy as np
import cv2

class AugumentadorVideo:
    @staticmethod
    def aplicar_ruido(frame, noise_level=0.11):
        ruido = np.random.randn(*frame.shape) * 255 * noise_level
        return np.clip(frame + ruido, 0, 255).astype(np.uint8)

    @staticmethod
    def translate_frame(frame, max_tx=40, max_ty=40):
        tx = np.random.randint(-max_tx,max_tx+1)
        ty = np.random.randint(-max_ty,max_ty+1)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
              
    # @staticmethod
    # def reduzir_brilho(frame, fator=0.2):
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)
    #     hsv_mod = cv2.merge((h, s, v))
    #     return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

    # @staticmethod
    # def aumentar_brilho(frame, fator=3.0):
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)
    #     v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)
    #     hsv_mod = cv2.merge((h, s, v))
    #     return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

    @staticmethod
    def borrao_gaussian(frame):
        return cv2.GaussianBlur(frame, (11, 11), sigmaX=3.0)

    @staticmethod
    def scale(frame, scale_factor=1.5):
        h, w = frame.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized = cv2.resize(frame, (new_w, new_h))
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        return resized[y_start:y_start + h, x_start:x_start + w]
    
    @staticmethod
    def flip_h(frame):
        return cv2.flip(frame, 1)
    
    @staticmethod
    def flip_v(frame):
        return cv2.flip(frame, 0)
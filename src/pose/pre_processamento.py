import cv2
import numpy as np

# frames
marca_largura = 115  
marca_altura = 30

def reajustar_frame(frame):
    frame = cv2.resize(frame, (300, 650))
    frame = remover_marca_de_agua(frame)
    return frame

def remover_marca_de_agua(frame): # quadradinho preto no canto inferior direito (alocado referente ao tamanho 300x650)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (width - marca_largura, height - marca_altura), (width, height), (0, 0, 0), -1) # w 100 / h 25
    return frame

# arrays
def converter_float32(video_keypoints):
    video_keypoints = np.array(video_keypoints, dtype=np.float32)
    return video_keypoints

# remove camada de pessoas detectadas [1] para melhorar estrutura
def espremer_estrutura_keypoint(video_keypoints):
    if video_keypoints.shape[1] == 1:
        video_keypoints = np.squeeze(video_keypoints, axis=1)
    return video_keypoints
# de: video_keypoints[frame][person][keypoint][coord] # para: video_keypoints[frame][keypoint][coord]

def aplicar_ruido(frame, noise_level=0.11):
    ruido = np.random.randn(*frame.shape) * 255 * noise_level
    frame_ruido = np.clip(frame + ruido, 0, 255).astype(np.uint8)
    return frame_ruido

def translate_frame(frame, max_tx=40, max_ty=40):
    tx = np.random.randint(-max_tx,max_tx+1)
    ty = np.random.randint(-max_ty,max_ty+1)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    alterado = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return alterado

def reduzir_brilho(frame, fator=0.2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)
    hsv_mod = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

def aumentar_brilho(frame, fator=3.0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)
    hsv_mod = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

def borrao_gaussian(frame):
    return cv2.GaussianBlur(frame, (11, 11), sigmaX=3.0)

def scale(frame, scale_factor=2.0):
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized = cv2.resize(frame, (new_w, new_h))
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    zoom = resized[y_start:y_start + h, x_start:x_start + w]
    return zoom
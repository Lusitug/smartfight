import cv2
import numpy as np

# frames
marca_largura = 115  
marca_altura = 30

def pre_process(frame):
    frame = cv2.resize(frame, (300, 650))
    frame = remover_marca_de_agua(frame)
    return frame

def remover_marca_de_agua(frame): # quadradinho preto no canto inferior direito (alocado referente ao tamanho 300x650)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (width - marca_largura, height - marca_altura), (width, height), (0, 0, 0), -1) # w 100 / h 25
    return frame

# arrays
def tipagem_compativel(video_keypoints):
    video_keypoints = np.array(video_keypoints, dtype=np.float32)
    return video_keypoints

# remove camada de pessoas detectadas [1] para melhorar estrutura
def espremer_estrutura_keypoint(video_keypoints):
    if video_keypoints.shape[1] == 1:
        video_keypoints = np.squeeze(video_keypoints, axis=1)
    return video_keypoints
# de: video_keypoints[frame][person][keypoint][coord] 
# para: video_keypoints[frame][keypoint][coord]



def add_noise(frame, noise_level=0.1):
    noise = np.random.randn(*frame.shape) * 255 * noise_level
    noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy_frame

# def scale_frame(frame, scale_factor=2.0):
#     h, w = frame.shape[:2]
#     new_w, new_h = int(w * scale_factor), int(h * scale_factor)
#     resized = cv2.resize(frame, (new_w, new_h))
#     x_start = (new_w - w) // 2
#     y_start = (new_h - h) // 2
#     cropped = resized[y_start:y_start + h, x_start:x_start + w]
#     return cropped

def translate_frame(frame, tx=50, ty=100):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return shifted

def flip_frame(frame):
    return cv2.flip(frame, 1)

def gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (11, 11), sigmaX=3.0)


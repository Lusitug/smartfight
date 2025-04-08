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
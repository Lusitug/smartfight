import os
import cv2
import numpy as np
from src.utils.utilidades import path_videos2estimate

def add_noise(frame, noise_level=0.1):
    noise = np.random.randn(*frame.shape) * 255 * noise_level
    noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy_frame

def translate_frame(frame, max_tx=40, max_ty=40):
    tx = np.random.randint(-max_tx, max_tx +1)
    ty = np.random.randint(-max_ty, max_ty +1)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return shifted

def reduzir_brilho(img, fator=0.2):
  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)

    hsv_mod = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

def aumentar_brilho(img, fator=3.0):
  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = np.clip(v.astype(np.float32) * fator, 0, 255).astype(np.uint8)

    hsv_mod = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

def scale(frame, scale_factor=2.0):
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized = cv2.resize(frame, (new_w, new_h))
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    cropped = resized[y_start:y_start + h, x_start:x_start + w]
    return cropped

def gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (11, 11), sigmaX=3.0)


AUGMENTATIONS = {
    'noise': add_noise,
    'translate': translate_frame,
    'shiny-': reduzir_brilho,
    'shiny+': aumentar_brilho,
    'gaussian': gaussian_blur,
    'scale':scale
}



def salvar_videos_augmentados(path_video, output_dir="videos_augmentados", fps_override=None):
    
    if not os.path.isfile(path_video):
        print(f"❌ [ERRO: {path_video}]")
        return

    nome_arquivo = os.path.splitext(os.path.basename(path_video))[0]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print(f"❌ [ERRO: {path_video}]")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)

    # writer
    writers = {}
    for nome, func in AUGMENTATIONS.items():
        nome_saida = f"{nome_arquivo}_{nome}.mp4"
        caminho_saida = os.path.join(output_dir, nome_saida)
        writers[nome] = cv2.VideoWriter(
            caminho_saida,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

    print(f"🎬 [PROCESSANDO: {nome_arquivo} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} FRAMES])")

    while True:
            ret, frame = cap.read()
            if not ret:
                break
            # augmumentations em lote
            for nome, func in AUGMENTATIONS.items():
                frame_aug = func(frame.copy())
                writers[nome].write(frame_aug)

    cap.release()
    for nome, writer in writers.items():
        writer.release()
        print(f"✅ [SALVO: {os.path.join(output_dir, f'{nome_arquivo}_{nome}.mp4')}]")

if __name__ == "__main__":
    path_video = os.path.join(path_videos2estimate, "Direto", "direto_EyHSY2mj.mp4")   
    salvar_videos_augmentados(path_video)



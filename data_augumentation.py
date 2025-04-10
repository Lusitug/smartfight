import os
import cv2
import numpy as np
from src.utils.utilidades import path_videos2estimate
# ------------------ AUGMENTAÇÕES ------------------

def add_noise(frame, noise_level=0.1):
    """Adiciona ruído gaussiano à imagem"""
    noise = np.random.randn(*frame.shape) * 255 * noise_level
    noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy_frame

def translate_frame(frame, tx=50, ty=100):
    """Translada a imagem no eixo x e y"""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return shifted

def flip_frame(frame):
    """Espelha a imagem horizontalmente"""
    return cv2.flip(frame, 1)

def gaussian_blur(frame):
    """Aplica desfoque gaussiano"""
    return cv2.GaussianBlur(frame, (11, 11), sigmaX=3.0)

# ------------------ DICIONÁRIO DE AUGMENTAÇÕES ------------------

AUGMENTATIONS = {
    'noise': add_noise,
    'translate': translate_frame,
    'flip': flip_frame,
    'gaussian': gaussian_blur
}

# ------------------ FUNÇÃO PRINCIPAL ------------------

def salvar_videos_augmentados(path_video, output_dir="videos_augmentados", fps_override=None):
    """Aplica técnicas de aumento de dados a um vídeo e salva um vídeo para cada técnica"""
    
    if not os.path.isfile(path_video):
        print(f"❌ Arquivo não encontrado: {path_video}")
        return

    nome_arquivo = os.path.splitext(os.path.basename(path_video))[0]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir o vídeo: {path_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)

    # Inicializa um writer para cada técnica
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

    print(f"🎬 Processando vídeo: {nome_arquivo} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames)")

    while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aplica cada técnica e grava o frame correspondente
            for nome, func in AUGMENTATIONS.items():
                frame_aug = func(frame.copy())
                writers[nome].write(frame_aug)

    cap.release()
    for nome, writer in writers.items():
        writer.release()
        print(f"✅ Vídeo salvo: {os.path.join(output_dir, f'{nome_arquivo}_{nome}.mp4')}")

    # ------------------ TESTE MANUAL ------------------

if __name__ == "__main__":
        # Substitua este caminho por qualquer vídeo local
    path_video = os.path.join(path_videos2estimate, "Direto", "direto_EyHSY2mj.mp4")   
    salvar_videos_augmentados(path_video)



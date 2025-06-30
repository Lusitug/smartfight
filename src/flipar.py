import cv2
from utils.caminhos import Caminhos

# Carrega o vídeo
video_path = "src\joelho-esquerdo-destro.mp4"
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi aberto corretamente
if not cap.isOpened():
    raise ValueError("Não foi possível abrir o vídeo de entrada")

# Obtém as propriedades do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Configurações para MP4 com melhor compatibilidade
output_path = 'output_flipped.mp4'
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    # Fallback para MP4V se AVC1 não estiver disponível
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise RuntimeError("Não foi possível inicializar o VideoWriter")

# Processa cada frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    flipped_frame = cv2.flip(frame, 1)
    out.write(flipped_frame)

# Garante que todos os frames foram escritos
out.release()
cap.release()
cv2.destroyAllWindows()

print(f"Vídeo processado salvo com sucesso em: {output_path}")
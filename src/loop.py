import cv2
import os
from utils.caminhos import Caminhos

# Carrega o vídeo original
input_path = Caminhos.teste_periodiciodade_loop2
cap = cv2.VideoCapture(input_path)

# Obtém as propriedades do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define a duração desejada em segundos (1 minuto = 60 segundos)
output_duration = 60  

# Calcula quantas vezes o vídeo original deve ser repetido
original_duration = 1  # 1 segundo
num_loops = int(output_duration / original_duration)

# Gera o nome do arquivo de saída com '_loop' antes da extensão
base_name = os.path.basename(input_path)
name_without_ext = os.path.splitext(base_name)[0]
output_path = f"{name_without_ext}_loop.mp4"

# Define o codec e o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop para repetir os frames
for _ in range(num_loops):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia para o primeiro frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

# Libera os recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo em loop salvo como: {output_path}")
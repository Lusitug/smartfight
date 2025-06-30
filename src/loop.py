import cv2
import os
from utils.caminhos import Caminhos  # Altere se necessário

# Duração total desejada em segundos
OUTPUT_DURATION = 60
ORIGINAL_DURATION = 1
NUM_LOOPS = int(OUTPUT_DURATION / ORIGINAL_DURATION)

# Caminho da pasta raiz do dataset
dataset_path = Caminhos.dataset_loop  # Exemplo: "caminho/para/dataset"

# Percorre recursivamente todas as subpastas
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if not file.endswith('.mp4'):
            continue

        if file.endswith('_loop.mp4'):
            print(f"Pulando (já tem _loop): {file}")
            continue

        input_path = os.path.join(root, file)

        # Carrega o vídeo original
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {input_path}")
            continue

        # Propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Nome de saída com _loop antes da extensão
        name_without_ext, ext = os.path.splitext(file)
        output_filename = f"{name_without_ext}_loop{ext}"
        output_path = os.path.join(root, output_filename)

        # 🔹 Criação do __init__.py se ainda não existir
        init_path = os.path.join(root, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("# Inicialização de pacote\n")
            print(f"Arquivo __init__.py criado em: {root}")

        print(f"Gerando vídeo: {output_path}")

        # Define codec e VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Escreve o vídeo em loop
        for _ in range(NUM_LOOPS):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

        # Libera recursos
        cap.release()
        out.release()

        print(f"Vídeo salvo: {output_path}")

cv2.destroyAllWindows()



# import cv2
# import os
# from utils.caminhos import Caminhos

# # Carrega o vídeo original
# input_path = Caminhos.teste_periodiciodade_loop2
# cap = cv2.VideoCapture(input_path)

# # Obtém as propriedades do vídeo
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define a duração desejada em segundos (1 minuto = 60 segundos)
# output_duration = 60  

# # Calcula quantas vezes o vídeo original deve ser repetido
# original_duration = 1  # 1 segundo
# num_loops = int(output_duration / original_duration)

# # Gera o nome do arquivo de saída com '_loop' antes da extensão
# base_name = os.path.basename(input_path)
# name_without_ext = os.path.splitext(base_name)[0]
# output_path = f"{name_without_ext}_loop.mp4"

# # Define o codec e o objeto VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # Loop para repetir os frames
# for _ in range(num_loops):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia para o primeiro frame
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)

# # Libera os recursos
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Vídeo em loop salvo como: {output_path}")
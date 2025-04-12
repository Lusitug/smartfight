import os

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aponta pra pasta 'src'

path_yolo =  os.path.join("src", "yolo_pesos", "yolo11x-pose.pt")

path_keypoints2csv =  os.path.join("src", "keypoints_extraidos")

path_videos2estimate = os.path.join("src", "dataset")

def gerar_init(caminho_pasta):
    init_path = os.path.join(caminho_pasta, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            pass

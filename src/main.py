import os
import sys
from pose.processar_keypoints import extrair_keypoints_dataset
# from pose.data_augument import aplicar_data_augmentation_dataset
from time import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    tempo_inicial = time()
    # aplicar_data_augmentation_dataset()
    extrair_keypoints_dataset() 
    tempo_final = time()
    print(f"\n\t⏰ [DURAÇÃO TOTAL: {tempo_final - tempo_inicial:.2f}]")

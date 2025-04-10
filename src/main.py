import os
import sys
from pose.processar_keypoints import extrair_keypoints_dataset
from pose.data_augument import aplicar_data_augmentation
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # aplicar_data_augmentation()
    extrair_keypoints_dataset() 
from ultralytics import YOLO
import cv2

HEIGHT_BASE = 640
WIDHT_BASE = 640

# classe principal
class DeteccaoPose():
    def __init__(self, modelo_path):
        pass

    def __del__(self):
        # Destruir classe
        pass

    def detectar_pontos_video(self, path):
        pass


    def criar_sequencias_temporais(self):
        pass


    def __str__(self):
        return f"{self.__class__.__name__} : {' || '.join([f'{chave}={valor}' for chave, valor in self.__dict__.items()])}"
    







        """
        # Dica

        Você pode transformar esse código em uma classe encapsulando as funções e variáveis necessárias. Aqui está uma versão em que a detecção de pontos-chave do vídeo é organizada dentro de uma classe:

```python
from ultralytics import YOLO

class DetecçãoPose:
    def __init__(self, modelo_path):
        # Inicializa o modelo com o caminho fornecido
        self.modelo_pose = YOLO(modelo_path)

    def detectar_pontos_video(self, path):
        # Inicializa a lista para armazenar os pontos-chave
        lista_pontos_chave = []

        # Faz a detecção no vídeo (aqui você precisaria de código para processar frames do vídeo)
        resultado = self.modelo_pose(path)
        
        # Aqui você pode adicionar a lógica para extrair os pontos-chave do resultado e adicioná-los à lista
        # Exemplo fictício de como você pode adicionar pontos-chave à lista
        for frame_resultado in resultado:
            pontos = frame_resultado.keypoints  # Supondo que o resultado tenha um atributo 'keypoints'
            lista_pontos_chave.append(pontos)

        return lista_pontos_chave

# Exemplo de uso
modelo_path = "yolov8n-pose.pt"
detector = DetecçãoPose(modelo_path)
video_path = "caminho/para/o/video"
pontos_chave = detector.detectar_pontos_video(video_path)

print(pontos_chave)
```

### Explicação:
1. **`__init__`**: Inicializa a classe com o caminho do modelo.
2. **`detectar_pontos_video`**: Função para processar o vídeo e detectar os pontos-chave.
3. **Uso da classe**: Criação de um objeto da classe `DetecçãoPose` e uso do método `detectar_pontos_video` para obter os pontos-chave.

Você pode expandir e adaptar esse código dependendo de como deseja tratar o vídeo e os pontos-chave extraídos.
        """
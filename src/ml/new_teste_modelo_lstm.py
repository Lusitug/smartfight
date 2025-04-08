import tensorflow as tf
# from tensorflow.ke
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# # Exemplo de dados simulados
# n_amostras = 100   # Número de sequências de treino
# n_frames = 30  # Número de frames por sequência
# n_features = 34  # 17 pontos-chave, cada um com coordenadas x e y

# # Criando dados de entrada simulados (X_train)
# X_train = np.random.rand(n_amostras, n_frames, n_features)  # Dados de exemplo aleatórios

# # Criando rótulos simulados (y_train) com 3 classes (por exemplo, 0, 1, 2)
# n_classes = 3
# y_train = np.random.randint(0, n_classes, n_amostras)  # Rótulos aleatórios

# # Convertendo os rótulos para one-hot encoding
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)

# # Dados de teste simulados
# X_test = np.random.rand(10, n_frames, n_features)  # Simulação de dados de teste
# y_test = np.random.randint(0, n_classes, 10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)


# batch_size = 32            # Quantidade de amostras no batch
# sequence_length = 10       # Quantidade de frames por sequência
# num_keypoints = 17         # Pontos-chave por frame
# num_features = num_keypoints * 2  # (x, y) para cada ponto
    # LSTM(64, input_shape=(sequence_length, num_features), return_sequences=False),

n_frames = [] 
n_classes = []

modelo_lstm = Sequential([
    LSTM(128, input_shape=(n_frames, 34), return_sequences=True), # 64
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'), # 16
    Dense(n_classes, activation='softmax'),
])

modelo_lstm.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


# # Treinar o modelo
# # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# modelo_lstm.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2) # , callbacks=[early_stopping]


# # Avaliar o modelo no conjunto de teste
# score = modelo_lstm.evaluate(X_test, y_test)
# print(f"Test loss: {score[0]}")
# print(f"Test accuracy: {score[1]}")


"""
    Estrutura LSTM
    (n_amostras, n_frames, n_features), onde:
    n_amostras: número total de sequências de treino,
    n_frames: número de frames por sequência (por exemplo, 30),
    n_features: número de características por frame (número de pontos-chave * 2,
    já que cada ponto-chave tem coordenadas x e y).
"""
# pre processamento
def ast_func(points):
    return ast.literal_eval(points)

def converter_frame_vetor(frame: pd.Series) -> np.ndarray:
    vetor = []
    for coluna in frame.index:
        if coluna == "frame":
            continue
        valor = frame[coluna]

        if isinstance(valor, str):
            try:
                ponto = ast.literal_eval(valor)
                if isinstance(ponto, (list, tuple)) and len(ponto) == 2:
                    vetor.extend(ponto)
                else:
                    # Caso seja uma lista malformada
                    vetor.extend([0.0, 0.0])
            except Exception:
                vetor.extend([0.0, 0.0])
        else:
            vetor.extend([0.0, 0.0])

    return np.array(vetor, dtype=np.float32)

def converter_array32(df: pd.DataFrame) -> np.ndarray:
    vetores = df.apply(converter_frame_vetor, axis=1).values.tolist()
    return np.array(vetores, dtype=np.float32)



##########################################################
# OBTER LABELS DO DATASET classe
class DatasetKeypoints:
    def __init__(self, dataset_csv_path: str):
        self.dataset_csv_path = dataset_csv_path

        self.nome_golpes_classe = sorted([
            pasta.strip()
            for pasta in os.listdir(dataset_csv_path)
            if os.path.isdir(os.path.join(dataset_csv_path, pasta)) and not pasta.startswith("__")
        ])

        self.golpe_idx = {golpe: idx for idx, golpe in enumerate(self.nome_golpes_classe)}
        self.paths_csv = []
        self.rotulos = []

        print("📁 [CLASSES DETECTADAS: ", self.nome_golpes_classe, "]")

        for nome_classe in self.nome_golpes_classe:
            classe_path = os.path.join(dataset_csv_path, nome_classe)
            for arquivo in os.listdir(classe_path):
                if arquivo.endswith(".csv"):
                    self.paths_csv.append(os.path.join(classe_path, arquivo))
                    self.rotulos.append(self.golpe_idx[nome_classe])

# KNN DTW CLASSE
class KNN_DTW:
    def __init__(self, k = 5):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for amostra in X_test:
            distancia = [dtw2.distance(amostra, amostra_treino)
                         for amostra_treino in self.X_train]
            indices = np.argsort(distancia)[:self.k]
            votos = [self.y_train[i] for i in indices]
            classe_comum = max(set(votos), key=votos.count)
            y_pred.append(classe_comum)
            return y_pred
##########################################################

## CORTAR CSV testes
def segmentar_com_janela_sliding(array: np.ndarray, janela=5, passo=20) -> List[np.ndarray]:
    segmentos = []
    for i in range(0, len(array) - janela + 1, passo):
        trecho = array[i:i+janela]
        if trecho.shape[0] == janela:
            segmentos.append(trecho)
    return segmentos


##########################################################
# Exemplo: articulação_l_w_1d tem shape (N,)
segmentos = segmentar_com_janela_sliding(articulação_l_w_1d, janela=59, passo=59)
print(f"Total de segmentos: {len(segmentos)}")
print(f"Shape de um segmento: {segmentos[0].shape}")


for i in range(15):
    plt.plot(segmentos[i])
    plt.title(f"Segmento {i}")
    plt.show()


############################# nem ta sendo usado
## PCA testes
def pca_teste(X_segmentado: List[np.ndarray], n_components: int = 80):
    X_flat = np.array([segmento.flatten() for segmento in X_segmentado])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)
    return X_pca, pca

aa, bb = pca_teste(segmentos, n_components=50)

print(aa)
print(bb)

plt.figure(figsize=(8, 6))
plt.scatter(aa[:, 0], aa[:, 1], c='blue', alpha=0.7)
plt.title('PCA - Componentes Principais dos Segmentos')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(aa[:, 0], aa[:, 1], c='blue', alpha=0.7)
for i, (x, y) in enumerate(zip(aa[:, 0], aa[:, 1])):
    plt.text(x, y, str(i), fontsize=8)  # Mostra o índice do segmento
plt.title('PCA - Componentes Principais dos Segmentos')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

######################

# KNN-DTW testes
dataset = DatasetKeypoints(Caminhos.dataset_csv)
X_segmentado, y = [], []

for path_csv, rotulo in zip(dataset.paths_csv, dataset.rotulos):
    df = pd.read_csv(path_csv)
    keypoints = converter_array32(df)
    segmentos = segmentar_com_janela_sliding(keypoints, janela=62, passo=62)
    X_segmentado.extend(segmentos)
    y.extend([rotulo]*len(segmentos))

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_segmentado, y, stratify=y, test_size=0.3)

# Treinar e avaliaro dataset com modelo KNN com DTW
modelo = KNN_DTW(k=5)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=dataset.nome_golpes_classe))

# testando video curto
golpe = pd.read_csv(Caminhos.teste_periodiciodade11)
golpe_conver = converter_array32(golpe)

segmentos_teste = segmentar_com_janela_sliding(golpe_conver, janela=len(golpe_conver), passo=len(golpe_conver))

if not segmentos_teste:
    print("⚠️ O vídeo externo é muito curto ou a janela está maior que o número de frames.")
else:
    y_pred_externo = modelo.predict(segmentos_teste)
    for i, pred in enumerate(y_pred_externo):
        nome_classe = dataset.nome_golpes_classe[pred]
        print(f"Predito como ➤    {nome_classe}")
from email import header
import pandas as pd
import numpy as np
from src.utils.caminhos import Caminhos
import ast

csv = pd.read_csv(Caminhos.teste_periodiciodade7)

print(csv.shape, )
print(csv.columns)
print("s",csv.dtypes)
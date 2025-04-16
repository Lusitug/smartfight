import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class ClassificadorLSTM(nn.Module):
        #numero de feature por frame/numero de neuronios ocultos lstm/numero de classes a serem classificadas
    def __init__(self, len_featrures_frames, #input_size
                len_neuronios_ocultos, #hidden_size
                len_tipos_golpes, #num_classes
                len_camadas=1,
                bidirecional=False,
                dropout=0.2):
        super(ClassificadorLSTM, self).__init__()
        self.bidirecional = bidirecional
        self.direcoes = 2 if bidirecional else 1
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=len_featrures_frames,
            hidden_size=len_neuronios_ocultos,
            batch_first=True,
            num_layers=len_camadas,
            bidirectional=bidirecional,
            dropout=dropout,
        )

        self.classificador = nn.Sequential(
            nn.Linear(len_neuronios_ocultos * self.direcoes, 64), #128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len_tipos_golpes)
        )

    def forward(self, x, len_frames_reais):
        # recebe o tamanho real dos frames obtidos com a collate_fn
        x_pack = pack_padded_sequence(x, len_frames_reais, batch_first=True, enforce_sorted=False)  # print("xpack: ", x_pack)
        # informar partes validas das sequencias com padding
        _, (estado_oculto, _) = self.lstm(x_pack)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        if self.bidirecional:
            estado_oculto_final = torch.cat((estado_oculto[-2], estado_oculto[-1]), dim=1)
        else:
            estado_oculto_final = estado_oculto[-1]    # última camada estado oculto
        out = self.dropout(estado_oculto_final)
        return self.classificador(out)
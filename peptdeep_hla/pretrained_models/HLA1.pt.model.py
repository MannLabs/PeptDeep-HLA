import torch
import peptdeep.model.building_block as building_block
from peptdeep.model.model_shop import *
class Model(torch.nn.Module):
    def __init__(self, *,
        hidden_dim=256,
        input_dim=128,
        n_lstm_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.nn = torch.nn.Sequential(
            torch.nn.Embedding(input_dim, hidden_dim//4),
            building_block.SeqCNN(hidden_dim//4),
            self.dropout,
            building_block.SeqLSTM(hidden_dim, hidden_dim, rnn_layer=n_lstm_layers),
            building_block.SeqAttentionSum(hidden_dim),
            self.dropout,
            torch.nn.Linear(hidden_dim,64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.nn(x).squeeze(-1)

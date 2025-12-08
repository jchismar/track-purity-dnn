import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.elu = nn.ELU()
        
    def forward(self, x):
        return self.elu(x + self.block(x))

class TrackPurityDNN(nn.Module):
    def __init__(self, input_dim=45, hidden_dims=[256, 128, 64, 32], residual_dim=32, dropout=0.1, n_res_blocks=3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.initial_layers = nn.Sequential(*layers)
        
        self.fc_in = nn.Linear(prev_dim, residual_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(residual_dim, dropout) for _ in range(n_res_blocks)
        ])
        
        self.out = nn.Linear(residual_dim, 1)
        
        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial_layers(x)
        x_in = F.elu(self.fc_in(x))
        
        for block in self.res_blocks:
            x_in = block(x_in)
        
        out = self.out(x_in)
        return out
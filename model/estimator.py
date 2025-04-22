from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim

class QFunction(nn.Module):
    """
    Représente un DQN.
    """
    def __init__(self, inputsize=8, hiddensize = 12, outputsize=3):
        super().__init__()
        self.T1 = nn.Linear(inputsize, hiddensize, dtype=torch.float64)   
        self.T2 = nn.Linear(hiddensize, outputsize, dtype=torch.float64)
    
    def copy(self, other: QFunction):
        """
        Copie un autre DQN.
        """
        self.T1 = other.T1
        self.T2 = other.T2

    def forward(self, state: list[int]) -> torch.Tensor:
        """
        Fait un forward.
        """
        X = torch.tensor(state, dtype=torch.float64)
        with torch.no_grad():
            Z1 = self.T1(X)
            A1 = torch.relu(Z1)
            Z2 = self.T2(A1)
        return Z2

    def getMove(self, state: list[int]) -> int:
        """
        Renvoie l'indice du meilleur coup.
        """
        tensor = self.forward(state)
        return torch.argmax(tensor).item()

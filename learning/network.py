from __future__ import annotations
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Représente un DQN.
    """
    def __init__(self, inputsize=8, hiddensize = 12, outputsize=3):
        super().__init__()
        self.T1 = nn.Linear(inputsize, hiddensize, dtype=torch.float)   
        self.T2 = nn.Linear(hiddensize, outputsize, dtype=torch.float)

    def save(self, filename):
        """
        Enregistre les poids dans un fichier.
        """
        weights = self.state_dict()
        torch.save(weights, filename)
    
    def load(self, filename):
        """
        Télécharge les poids depuis un fichier.
        """
        weights = torch.load(filename)
        self.load_state_dict(weights)
        
    def copy(self, other: QNetwork):
        """
        Copie un autre DQN.
        """
        self.T1 = other.T1
        self.T2 = other.T2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fait un forward.
        """
        Z1 = self.T1(X)
        A1 = torch.relu(Z1)
        Z2 = self.T2(A1)
        return Z2

    def getMove(self, state: list[int]) -> int:
        """
        Renvoie l'indice du meilleur coup.
        """
        X = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            tensor = self.forward(X)
        return torch.argmax(tensor).item()



        
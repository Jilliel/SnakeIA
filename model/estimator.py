from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim

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


class Qtrainer:
    """
    Cette classe se charge d'entrainer 
    """
    def __init__(self, network: QNetwork, learning_rate=0.01, discount_factor=0.9):
        self.Qnet: nn.Module = network
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.optimizer: optim.Optimizer = optim.Adam(params=network.parameters(), lr=learning_rate)
        self.discount: float = discount_factor

    def train(self, batch: list[tuple]):
        """
        Entraine le réseau sur une itération.
        """
        batch_phi0, batch_action, batch_reward, batch_phi1, batch_terminal = zip(*batch)
        states0 = torch.tensor(batch_phi0, dtype=torch.float)
        states1 = torch.tensor(batch_phi1, dtype=torch.float)

        actions = torch.tensor(batch_action, dtype=torch.int64)
        rewards = torch.tensor(batch_reward, dtype=torch.float)
        mask = torch.tensor(batch_terminal, dtype=torch.bool)

        values = torch.gather(input=self.Qnet(states0), dim=1, index=actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            newvalues = torch.where(
                mask,
                rewards,
                rewards + self.discount * torch.max(self.Qnet(states1), dim=1)[0]
            )

        loss: torch.Tensor = self.criterion(values, newvalues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
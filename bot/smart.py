from learning.network import QNetworkCNN
from bot.abstract import AbstractSnake
from random import randint
import numpy as np

class CleverSnake(AbstractSnake):
    """
    Ce Snake override les méthodes principales pour permettre l'entrainement du DQN.
    """
    def __init__(self, width=15, height=15):
        super().__init__(width, height)
        # Q-function variables
        self.Qnet: QNetworkCNN = QNetworkCNN()
        # Env variables
        self.maxround = 800
        # Player variables
        self.move: int = None

    def save(self, filename) -> None:
        """
        Enregistre le Qnetwork cible dans un fichier .pth
        """
        self.Qnet.save(filename)
    
    def load(self, filename) -> None:
        """
        Récupère les poids du Qnetwork depuis un fichier .pth
        """
        self.Qnet.load(filename)

    def getState(self) -> np.ndarray:
        """
        Renvoie une liste représentant l'état actuel.
        """
        state = np.zeros((self.height, self.width), dtype=np.int8)
        for pos in self.body:
            state[pos] = 1
        state[self.apple] = 3
        state[self.getHead()] = 2

        return state

    def getReward(self) -> int:
        """
        Renvoie la reward du coup.
        """
        if self.getHead() == self.apple:
            reward = 1
        elif self.checkCollision():
            reward = -1
        elif self.checkLimit():
            reward = -1
        else:
            reward = 0
        return reward
    
    def randplay(self) -> None:
        """
        Joue de manière aléatoire.
        """
        self.move = randint(0, 2)
        match self.move:        
            case 0:
                self.right()
            case 1:
                self.left()
            case _:
                pass

    def play(self) -> None:
        """
        Joue selon la politique.
        """
        state = self.getState()
        self.move = self.Qnet.getMove(state)
        match self.move:        
            case 0:
                self.right()
            case 1:
                self.left()
            case _:
                pass
    

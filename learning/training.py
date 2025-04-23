import torch
import torch.nn as nn
import torch.optim as optim

from bot.clever import CleverSnake
from learning.network import QNetwork

from collections import deque
from random import random, sample

class Qtrainer:
    """
    Cette classe se charge d'entrainer le Snake
    """
    def __init__(self, snake: CleverSnake):
        # Snake here
        self.snake: CleverSnake = snake
        # Buffer here
        self.buffersize: int = 50000 #Taille du buffer
        self.buffer: deque = deque([], maxlen=self.buffersize) # Buffer
        self.batchsize: int = 64 #Taille du batch
        # Training constants here
        self.frequency: int = 1000 #Fréquence de sauvegarde
        self.iterations: int = 0 # .
        self.gamma: float = 0.99 #Discount factor
        self.alpha: float = 0.0005 #Learing rate
        self.epsilon: float = 1 #Pour la politique epsilon greedy
        self.epsilon_inf: float = 0.01 # .
        self.epsilon_factor: float = 0.995 # .
        # Networks here
        self.Qnet: QNetwork = snake.Qnet
        self.Qtarget: QNetwork = QNetwork()
        # Optimizer here
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.optimizer: optim.Optimizer = optim.Adam(params=self.Qnet.parameters(), lr=self.alpha)

    def learn(self, batch: list[tuple]):
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
                rewards + self.gamma * torch.max(self.Qnet(states1), dim=1)[0]
            )

        loss: torch.Tensor = self.criterion(values, newvalues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, filename):
        """
        Sauvegarde l'entrainement.
        """
        weights = self.Qtarget.state_dict()
        torch.save(weights, filename)
    
    def play(self):
        """
        Joue une partie et entraine
        """
        # Snake param here
        self.snake.clear()
        terminal = False

        while not terminal:

            phi0 = self.snake.getState() 

            # Politique epsilon-greedy 
            if random() <= self.epsilon:
                self.snake.randplay()
            else:
                self.snake.play()
            self.snake.extend()

            #Maj de epsilon
            self.epsilon = max(self.epsilon * self.epsilon_factor, self.epsilon_inf)

            # On sauvegarde la transition 
            action = self.snake.move
            reward = self.snake.getReward()
            terminal = self.snake.checkTerminal()
            print(action, reward, terminal)
            phi1 = self.snake.getState()
            transition = (phi0, action, reward, phi1, terminal)
            self.buffer.append(transition)

            # On entraine le réseau
            if len(self.buffer) >= self.batchsize:
                batch = sample(self.buffer, self.batchsize)
                self.learn(batch)

            # On met à jour les poids du réseau cible.
            self.iterations += 1
            if self.iterations % self.frequency == 0:
                self.Qtarget.copy(self.Qnet)
            
            self.snake.round += 1
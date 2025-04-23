from model.estimator import QNetwork, Qtrainer
from random import random, randint, sample
from bot.abstract import AbstractSnake
from overrides import overrides
from collections import deque

class TrainingSnake(AbstractSnake):
    """
    Ce Snake override les méthodes principales pour permettre l'entrainement du DQN.
    """
    def __init__(self, width=15, height=15, batchsize=20, buffersize=100, copyfreq=20, learning_rate=0.01, discount_factor=0.8):
        super().__init__(width, height)
        # Epsilon-greedy policy
        self.epsilon: float = 1
        # Q-function variables
        self.N: int = 0
        self.copyfreq: int = copyfreq
        self.Qfunc: QNetwork = QNetwork(inputsize=9)
        self.Qtarget: QNetwork = QNetwork(inputsize=9)
        self.Qtrainer: Qtrainer = Qtrainer(self.Qfunc, learning_rate=learning_rate, discount_factor=discount_factor)
        # Buffers variables
        self.move: int = None
        self.batchsize: int = batchsize
        self.buffer: deque = deque([], maxlen=buffersize)
        # Env variables
        self.maxround = 300

    def save(self, filename) -> None:
        """
        Enregistre le Qnetwork cible dans un fichier .pth
        """
        self.Qtarget.save(filename)
    
    def load(self, filename) -> None:
        """
        Récupère les poids du Qnetwork depuis un fichier .pth
        """
        self.Qfunc.load(filename)

    def getState(self) -> list[int]:
        """
        Renvoie une liste représentant l'état actuel.
        [
        - Pomme devant: 0-1
        - Pomme droite: 0-1
        - Pomme derrière: 0-1
        - Pomme gauche: 0-1
        - Up: 0-1
        - Right: 0-1
        - Down: 0-1
        - Left: 0-1
        - Distance mur: 1->max(width-height)
        ]
        """
        yh, xh = self.getHead()
        ya, xa = self.apple

        #Partie pomme
        dy = ya-yh
        dx = xa-xh
        state_apple = [int(dy <0),
                 int(dx > 0),
                 int(dy > 0), 
                 int(dx < 0)]
        
        #Partie direction
        state_dir = [0, 0, 0, 0]
        state_dir[self.direction.index] = 1

        #Partie distance
        distances = [yh, self.width-xh, self.height-yh, xh]
        state_dist = [distances[self.direction.index] / max(self.height, self.width)]

        return state_apple + state_dir + state_dist

    def playMove(self):
        """
        Joue le coup passé en argument:
        - 0: Droite
        - 1: Gauche
        - 2: Devant
        """
        match self.move:
            case 0:
                self.right()
            case 1:
                self.left()
            case 2:
                pass
            case _:
                raise NotImplementedError

    def play(self, state: list[int]) -> None:
        """
        Permet au Snake de changer de direction.
        """
        # On applique une politique epsilon-greedy
        if random() <= self.epsilon:
            self.move = randint(0, 2)
        else:
            self.move = self.Qfunc.getMove(state)

        self.epsilon = max(self.epsilon*0.995, 0.01)
        self.playMove()
    
    @overrides
    def clear(self) -> None:
        """
        Remet à zéro le Snake.
        """
        self.body.clear()
        self.body.append(self.start)
        self.addApple()
        self.round = 0
        self.score = 0
        self.epsilon = 0.1
    
    @overrides
    def run(self):
        """
        Joue une partie
        """
        self.clear()

        # Preprocesses phi1
        phi0 = None
        phi1 = self.getState()
        terminal = False

        while not terminal and self.round < self.maxround:

            phi0 = phi1 
            self.play(state=phi0)
            self.extend()

            # Puis on s'occupe d'attribuer une reward
            if self.getHead() == self.apple:
                self.score += 1
                self.maxround += 30
                self.addApple()
                # On encourage fortement la collecte de pommes.
                reward = 1
            elif not self.checkLimit():
                # On pénalise cette erreur.
                terminal = True
                reward = -1
            elif self.checkCollision():
                # On pénalise cette erreur.
                terminal = True
                reward = -1
            else:
                reward = 0
                self.retract()

            phi1 = self.getState()

            # On gère l'entrainement
            transition = (phi0, self.move, reward, phi1, terminal)
            self.buffer.append(transition)
            if len(self.buffer) >= self.batchsize:
                batch = sample(self.buffer, self.batchsize)
                self.Qtrainer.train(batch)

            # On gère la sauvegarde du Q network pour permettre plus de stabilité.
            self. N += 1
            if self.N % self.copyfreq == 0:
                self.Qtarget.copy(self.Qfunc)
            
            # Enfin on gère l'affichage
            if not terminal:
                self.show()
            self.round += 1
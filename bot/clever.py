from learning.network import QNetwork
from bot.abstract import AbstractSnake
from random import randint

class CleverSnake(AbstractSnake):
    """
    Ce Snake override les méthodes principales pour permettre l'entrainement du DQN.
    """
    def __init__(self, width=15, height=15):
        super().__init__(width, height)
        # Q-function variables
        self.Qnet: QNetwork = QNetwork(inputsize=16)
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
        - Distance mur haut
        - Distance mur droit
        - Distance mur bas
        - Distance mur gauche
        - Distance corps haut
        - Distance corps droit
        - Distance corps bas
        - Distance corps gauche
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
        max_distance = max(self.height, self.width) # Pour la normalisation
        state_dist_wall = []
        state_dist_body = []

        wall_distance = (yh, self.width-xh, self.height-yh, xh)
        directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        for (dy, dx), dist in zip(directions, wall_distance):
            state_dist_wall.append(dist / max_distance)
            dmin = dist
            for i in range(1, dist):
                if (yh + i * dy, xh + i * dx) in self.body:
                    dmin = i
                    break
            state_dist_body.append(dmin / max_distance)
            
        state_dist = state_dist_wall + state_dist_body

        return state_apple + state_dir + state_dist

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
    

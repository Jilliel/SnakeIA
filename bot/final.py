from bot.abstract import AbstractSnake    
from model.estimator import QNetwork

class FinalSnake(AbstractSnake):
    def __init__(self, width = 15, height = 15) -> None:
        super().__init__(width, height)
        self.Qnet: QNetwork = QNetwork(inputsize=9)
        self.move: int = None

    def load(self, filename) -> None:
        """
        Récupère les poids depuis un fichier.
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

    def playMove(self) -> None:
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

    def play(self) -> None:
        """
        Permet au Snake de changer de direction.
        """
        # On applique une politique epsilon-greedy
        state = self.getState()
        self.move = self.Qnet.getMove(state)
        self.playMove()
from abc import ABC, abstractmethod
from collections import deque

position = tuple[int, int]

class Direction:
    """
    Représente une direction.
    """
    def __init__(self, default=0) -> None:
        self.directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.index = default
    
    def right(self) -> None:
        """
        Tourne à droite
        """
        self.index += 1
        self.index %= 4

    def left(self) -> None:
        """
        Tourne à droite
        """
        self.index -= 1
        self.index %= 4
    
    def follow(self, pos) -> position:
        """
        Renvoie la case suivante.
        """
        y, x = pos
        dy, dx = self.directions[self.index]
        return y+dy, x+dx


class AbstractSnake(ABC):
    """
    Représente un Snake.
    """
    def __init__(self, head) -> None:
        self.body = deque([head])
        self.direction: Direction = Direction()

    def clear(self, head):
        """
        Remet à zéro le Snake.
        """
        self.body.clear()
        self.body.append(head)

    def getHead(self) -> position:
        """
        Renvoie la tête du Snake.
        """
        return self.body[-1]
    
    def getBody(self) -> list[position]:
        """
        Renvoie le corps du Snake.
        """
        return list(self.body)[:-1]
    
    def move(self) -> None:
        """
        Déplace le Snake.
        """
        head = self.getHead()
        newhead = self.direction.follow(head)
        self.body.append(newhead)
    
    def retract(self) -> None:
        """
        Fait avancer l'arrière du Snake.
        """
        self.body.popleft()

    def left(self) -> None:
        """
        Tourne à gauche.
        """
        self.direction.left()

    def right(self) -> None:
        """
        Tourne à droite.
        """
        self.direction.right()

    @abstractmethod
    def play(self) -> None:
        """
        Permet au Snake de changer de direction.
        """
        pass

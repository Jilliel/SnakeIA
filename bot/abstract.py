from abc import ABC, abstractmethod
from collections import deque
from random import randint
from time import sleep
import os

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
    def __init__(self, width: int, height: int) -> None:
        # Map
        self.width: int = width
        self.height: int = height
        # Snake
        self.direction: Direction = Direction()
        self.start: position = (self.height//2 + 1, self.width//2 + 1)
        self.body: deque = deque([self.start])
        # Env
        self.score: int = 0
        self.apple: position = None
        self.debug: bool = False

    def checkLimit(self) -> bool:
        """
        Indique si la tête est en dehors du terrain.
        """
        y, x = self.getHead()
        return 0 <= y < self.height and 0 <= x < self.width
    
    def checkCollision(self) -> bool:
        """
        Indique si la tête touche le corps
        """
        head = self.body.pop()
        ret = head in self.body
        self.body.append(head)
        return ret

    def addApple(self) -> None:
        """
        Crée une pomme.
        """
        y = randint(0, self.height-1)
        x = randint(0, self.width-1)
        newapple = (y, x)
        if newapple in self.body:
            self.addApple()
        else:
            self.apple = newapple

    def clear(self) -> None:
        """
        Remet à zéro le Snake.
        """
        self.body.clear()
        self.body.append(self.start)
        self.addApple()
        self.score = 0

    def getHead(self) -> position:
        """
        Renvoie la tête du Snake.
        """
        return self.body[-1]
    
    def extend(self) -> None:
        """
        Déplace le Snake dans sa direction.
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
    
    def run(self) -> None:
        """
        Joue une partie
        """
        self.clear()
        while True:
            # Fait avancer le Snake
            self.play()
            self.extend()
            # Puis s'occupe des évènements courants.
            if self.getHead() == self.apple:
                self.score += 1
                self.addApple()
            elif not self.checkLimit():
                break
            elif self.checkCollision():
                break
            else:
                self.retract()

            if self.debug:
                self.show()
                sleep(0.15)

    def show(self) -> None:
        """
        Affiche le Snake.
        """
        os.system("clear")
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.apple:
                    symbole = "0"
                elif (i, j) == self.getHead():
                    symbole = "X"
                elif (i, j) in self.body:
                    symbole = "*"
                else:
                    symbole = "-"
                print(symbole, end=" ")
            print()
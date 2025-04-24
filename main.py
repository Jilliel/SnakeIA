from learning.training import Qtrainer
from bot.clever import CleverSnake
import matplotlib.pyplot as plt

# Objects definition
snake = CleverSnake()
snake.load("weights.pth")
snake.maxround = 1500
trainer = Qtrainer(snake)
history = []

# Iterations data
epoch = 200
epochsize = 20
# Does main thing
for i in range(epoch):
    mscore = 0
    for _ in range(epochsize):
        mscore += trainer.game() / epochsize
    print(f"Epoch {i}: {mscore:0.1f}")
    history.append(mscore)

#Stores the weights
trainer.save("weights.pth")

#Création d'un graphe
#plt.plot(history)
#plt.xlabel("N° Epoch")
#plt.ylabel("Score moyen")
#plt.savefig("history.png")

#Shows the result
input("Ready ?")
snake.debug = True
snake.run()

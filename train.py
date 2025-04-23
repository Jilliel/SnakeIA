from bot.student import TrainingSnake
import matplotlib.pyplot as plt
from collections import deque

snake = TrainingSnake(batchsize=64, 
                    buffersize=50000, 
                    copyfreq=1000,
                    learning_rate=0.0005,
                    discount_factor=0.99)

snake.load("weights.pth")

simsize = 100
epochsize = 50

score_buffer = deque([], maxlen=epochsize)
round_buffer = deque([], maxlen=epochsize)
score_history = []
round_history = []

for i in range(simsize):
    for _ in range(epochsize):
        snake.run()
        score_buffer.append(snake.score)
        round_buffer.append(snake.round)
    mscore = sum(score_buffer)/epochsize
    mround = sum(round_buffer)/epochsize
    print(f"Epoch {i}:\n\t-Score moyen={mscore}\n\t-Durée moyenne={mround} rounds.")
    score_history.append(mscore)
    round_history.append(mround)

plt.subplot(1, 2, 1)
plt.plot(score_history)
plt.xlabel("Epoch")
plt.ylabel("Score moyen")
plt.subplot(1, 2, 2)
plt.plot(round_history)
plt.xlabel("Epoch")
plt.ylabel("Nombre de rounds moyen")

plt.savefig("history.png")
snake.save("weights.pth")
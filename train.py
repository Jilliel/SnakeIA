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

buffer = deque([], maxlen=epochsize)
history = []

for i in range(simsize):
    for _ in range(epochsize):
        snake.run()
        buffer.append(snake.score)
    accuracy = sum(buffer)/epochsize
    print(f"Epoch {i}: accuracy={accuracy}")
    history.append(accuracy)

plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Score moyen")
plt.savefig("history.png")

snake.save("weights.pth")
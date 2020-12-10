#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
people = ('Farrah', 'Fred', 'Felicia')
fruit_names = ('Apples', 'Bananas', 'Oranges', 'Peaches')
app = plt.bar(range(3), fruit[0, :], color='r', width=0.5)
ban = plt.bar(range(3), fruit[1, :], color='yellow', width=0.5, bottom=fruit[0, :])
ora = plt.bar(range(3), fruit[2, :], color='#ff8000', width=0.5, bottom=fruit[0, :] + fruit[1, :])
pea = plt.bar(range(3), fruit[3, :], color='#ffe5b4', width=0.5, bottom=fruit[0, :] + fruit[1, :] + fruit[2, :])
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.xticks(range(3), people)
plt.yticks(np.arange(0, 81, 10))
plt.legend(fruit_names)
plt.show()

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.ylim(0, 10)
plt.plot(range(len(y)), y, '-r')
plt.show()

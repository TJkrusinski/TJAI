


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10)

with plt.style.context('Solarize_Light2'):
    plt.plot(x, np.sin(x) + 2 + np.random.randn(50))

    plt.title('A random line')
    plt.xlabel('X label', fontsize=14)
    plt.ylabel('Y label', fontsize=14)

plt.show()
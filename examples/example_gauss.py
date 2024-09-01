import numpy as np
import praktikum_tools as tl

x = np.linspace(-1, 3, 100)
x = tl.ErrValue(x, 0.1)

y = gauss(x, {'A': [4], 'sigma': [1.5], 'mu': [1]})

p = Plot()
p.add(x, y)
p.show()

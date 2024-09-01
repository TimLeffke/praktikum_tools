import numpy as np
import praktikum_tools as tl

x = np.linspace(-1, 5   , 100)
x = tl.ErrValue(x, 0.05)

y = tl.gauss(x, {'A': [4, 2], 'sigma': [0.5, 0.75], 'mu': [1, 2], 'background': []}) + np.random.normal(loc = 0, scale = 0.1, size = len(x))
y.error = 0.1

params = tl.fit_gauss(x, y, n = 2)

print(params)

p = tl.Plot(legend = True)
p.add(x, y, label = "Messwerte")
p.gauss(x, params, draw_individual = False, label = "Anpassung")
p.gauss(x, params, draw_individual = True)
p.show()

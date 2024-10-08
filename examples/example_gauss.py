import numpy as np
import praktikum_tools as tl

# Generate synthetic data
x = np.linspace(-1, 5   , 100)
x = tl.ErrValue(x, 0.05)
y = tl.gauss(x, {'A': [4, 2], 'sigma': [0.5, 0.75], 'mu': [1, 2], 'background': []}) + np.random.normal(loc = 0, scale = 0.1, size = len(x))
y.error = 0.1

# Fit the sum of two gauss curves without background
params = tl.fit_gauss(x, y, n = 2)

print(params)

# Plot the data and results
p = tl.Plot(legend = True)
p.add(x, y, label = "Data")
p.gauss(x, params, draw_individual = False, label = "Fit")
p.gauss(x, params, draw_individual = True)
p.show()

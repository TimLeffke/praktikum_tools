import numpy as np
import praktikum_tools as tl

# Define custom function
def custom_sin(x, A, omega, phase):
    return A * tl.sin(x * omega + phase)

# Generate synthetic data
x = np.linspace(-1, 1, 10)
x = tl.ErrValue(x, 0.01)
y = custom_sin(x, 2.5, 1.2, 0.3) + np.random.normal(loc = 0, scale = 0.1, size = len(x))
y.error = 0.1

# Fit custom function
A, omega, phase = tl.fit_func(custom_sin, x, y)

# Plot data and results
p = tl.Plot(xlabel = "x", ylabel = r"$A\sin(\varphi_0 + \omega x)$", legend = True)
p.add(x, y, label = "Data")
p.func(custom_sin, x, [A, omega, phase], label = "Fit")
p.show()

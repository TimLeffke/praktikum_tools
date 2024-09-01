import numpy as np
import praktikum_tools as tl

def sin(x, A, omega, phase):
    return A * tl.sin(x * omega + phase)


x = np.linspace(-1, 1, 10)
x = tl.ErrValue(x, 0.01)

y = sin(x, 2.5, 1.2, 0.3) + np.random.normal(loc = 0, scale = 0.1, size = len(x))
y.error = 0.1

A, omega, phase = tl.fit_func(sin, x, y)

p = tl.Plot(xlabel = "x", ylabel = r"$A\sin(\varphi_0 + \omega x)$")
p.add(x, y)
p.func(sin, x, [A, omega, phase], autoscale = False)
p.show()

import numpy as np
import praktikum_tools as tl

# Create x values
x = np.linspace(0, 3*np.pi, 100)

# x values should have an error of 0.05
x = tl.ErrValue(x, 0.05)
y = tl.sin(x)


# Create Plot object
p = tl.Plot(title = "Best Plot", xlabel = "x", ylabel = "y")
# Add data
p.add(x, y)
# Show plot)
p.show()

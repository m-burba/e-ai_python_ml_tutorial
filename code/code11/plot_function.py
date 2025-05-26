import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-10, 10, 400)

# Calculate the corresponding y values using the function f(x)
y = f(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)', color='blue')
plt.title('Plot of f(x) over x in [-10, 10]')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid()
plt.legend()

# Save the figure
plt.savefig('plot.png')
plt.close()
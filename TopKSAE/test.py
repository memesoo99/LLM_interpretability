import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define the custom pastel blue colormap
blue_pastel_cmap = LinearSegmentedColormap.from_list(
    "blue_pastel", ["#D0E8F2", "#A9D6E5", "#89C2D9", "#61A5C2", "#468FAF"]
)

# Create a gradient to visualize the colormap
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# Display the colormap
plt.figure(figsize=(8, 2))
plt.imshow(gradient, aspect='auto', cmap=blue_pastel_cmap)
plt.axis('off')  # Remove axes for better visualization
plt.title("Blue Pastel Colormap", fontsize=14, pad=10)
plt.show()

import numpy as np
from matplotlib import pyplot as plt

n = 10
mat1 = np.zeros((n+1))
mat2 = np.zeros((n+1))

for i in range(n+1):
    mat1[i] = i / n
    mat2[i] = 1 - i / n


print(mat1)
print(mat2)

room = np.vstack((mat1+mat2, mat1 * mat2))

fig, ax = plt.subplots()
display = ax.matshow(room, cmap='viridis')
cbar = plt.colorbar(display, ax=ax, orientation='vertical')

color = ['black', 'white']

# Add text annotations for each cell
for (i, j), val in np.ndenumerate(room):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color[i])
    ax.text(j - 0.4, i - 0.4, round(mat1[j], 2), ha='left', va='top', color=color[i], fontsize=8)
    ax.text(j + 0.4, i - 0.4, round(mat2[j], 2), ha='right', va='top', color=color[i], fontsize=8)

plt.show()
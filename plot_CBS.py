from itertools import product
import json

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

model_size = 'large'
num_heads = {'small':12, 'medium':24, 'large':36, 'xl':48}[model_size]
with open(f'./rome_results_{model_size}.json', 'r') as f:
    data = json.load(f)


#########


prob_array = np.zeros((7,num_heads))

for num_heads, pos in product(range(num_heads), range(7)):
    prob_array[pos, num_heads] = data['hidden'][str(num_heads)][str(pos)]

plt.figure(figsize=(10,6))
plt.xticks(np.arange(0,num_heads,5)+0.5, np.arange(0,num_heads,5))
plt.yticks()
plt.yticks(np.arange(0,7)+0.5, ["The*", "Big*", "Bang*", "Theory*", "premie", "res", "on"])
# im = plt.imshow(prob_array, cmap="Purples")
im = plt.pcolormesh(prob_array, cmap="Purples")
plt.gca().invert_yaxis()
cbar = plt.colorbar(im)
cbar.ax.set_title("p(CBS)", y=-0.07)
plt.savefig(f'./{model_size}_hidden.png')


#########

prob_array = np.zeros((6,num_heads))

for num_heads, pos in product(range(num_heads), range(6)):
    prob_array[pos, num_heads] = data['mlp'][str(num_heads)][str(pos)]

plt.figure(figsize=(10,6))
plt.xticks(np.arange(0,num_heads,5)+0.5, np.arange(0,num_heads,5))
plt.yticks()
plt.yticks(np.arange(0,7)+0.5, ["The*", "Big*", "Bang*", "Theory*", "premie", "res", "on"])
# im = plt.imshow(prob_array, cmap="Purples")
im = plt.pcolormesh(prob_array, cmap="Greens")
plt.gca().invert_yaxis()
cbar = plt.colorbar(im)
cbar.ax.set_title("p(CBS)", y=-0.07)
plt.savefig(f'./{model_size}_mlp.png')

#########

prob_array = np.zeros((6,num_heads))

for num_heads, pos in product(range(num_heads), range(6)):
    prob_array[pos, num_heads] = data['attn'][str(num_heads)][str(pos)]

plt.figure(figsize=(10,6))
plt.xticks(np.arange(0,num_heads,5)+0.5, np.arange(0,num_heads,5))
plt.yticks()
plt.yticks(np.arange(0,7)+0.5, ["The*", "Big*", "Bang*", "Theory*", "premie", "res", "on"])
# im = plt.imshow(prob_array, cmap="Purples")
im = plt.pcolormesh(prob_array, cmap="Reds")
plt.gca().invert_yaxis()
cbar = plt.colorbar(im)
cbar.ax.set_title("p(CBS)", y=-0.07)
plt.savefig(f'./{model_size}_attn.png')
from itertools import product
import json

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

model_size = 'xl'
num_heads = {'small':12, 'medium':24, 'large':36, 'xl':48}[model_size]
with open(f'./rome_results_{model_size}.json', 'r') as f:
    data = json.load(f)

prob_array = np.zeros((6,num_heads))

for num_heads, pos in product(range(num_heads), range(6)):
    prob_array[pos, num_heads] = data[str(num_heads)][str(pos)]

plt.figure()
im = plt.imshow(prob_array, cmap="Purples")
plt.colorbar(im)
plt.savefig(f'./{model_size}.png')
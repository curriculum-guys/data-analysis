import numpy as np

n_conditions = 5 ** 6

raw = np.load('../data/trialsS1.npy', allow_pickle=True)
data = [t[0] for t in raw]

linear_data = []
for r in data:    
    for c in r:
        linear_data.append(c)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(16,12)})

# sns.histplot(linear_data)
# plt.xlabel("conditions combination index")
# plt.show()

pop_size = 250

gen_data = [[] for _ in range(len(linear_data) // (pop_size*10))]
for i in range(len(linear_data)):
    gen = (i // (pop_size*10))
    gen_data[gen].append(linear_data[i])

conditions_map_data = []
for gen in gen_data:
    conditions_map = [0 for _ in range(n_conditions)]
    for r in gen:
        conditions_map[r] += 1
    conditions_map_data.append(conditions_map)

print(len(conditions_map_data))
sns.heatmap(np.transpose(conditions_map_data))
plt.show()
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter
import copy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str)
parser.add_argument("--neuron_file",
                    type=str,
                    help="Flie to be plotted.")
parser.add_argument("--num_layers",
                    type=int,
                    help="Number of layers in the model.")
args = parser.parse_args()

fig_dir = 'eplot_results/neuron_distribution'


model_color_dict={
    "gemma-2b": "#fed070", #f6e093
    "lama2-7b": "#e58b7b",
    "lama2-13b": "#97af19",
    "llama3-8b": "#386795",#97af19
}
# =========== stat kn_bag ig ==============
y_points = []
tot_bag_num = 0
tot_rel_num = 0
tot_neurons = 0
neuron_bag_counter = Counter()
# for filename in os.listdir(kn_dir):
#     if not filename.startswith('kn_bag-'):
#         continue

with open(args.neuron_file, 'r') as f:
    neuron_bag_list = json.load(f)
    # neuron_bag_list: 所有example所产生的neurons 
    # neuron_bag: 某example所产生的neurons，如[[11, 2891, 0.0002630653325468302], [10, 1845, 0.0002512137289159], [10, 1154, 0.00021400029072538018], ...]
    for neuron_bag in neuron_bag_list:
        for neuron in neuron_bag:
            neuron_bag_counter.update([neuron[0]]) # 计数：哪层有几个神经元
            y_points.append(neuron[0])
    # tot_num = len(kn_bag_list)

# neuron_bag_counter:
# 层: 该层的神经元个数
            
kn_bag_counter_ori = copy.deepcopy(neuron_bag_counter)

for k, v in neuron_bag_counter.items():
    tot_neurons += neuron_bag_counter[k]
for k, v in neuron_bag_counter.items():
    neuron_bag_counter[k] /= tot_neurons

# neuron_bag_counter:
# 层: 该层的神经元个数/总神经元个数

# average # neurons
print('total # neurons:', tot_neurons)

plt.figure(figsize=(8.5, 3))

x = np.array([i + 1 for i in range(args.num_layers)])
y = np.array([neuron_bag_counter[i] for i in range(args.num_layers)])
# plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i for i in range(4, args.num_layers+1, 4)], labels=[i for i in range(4, args.num_layers+1, 4)], fontsize=20)
# plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=10)
plt.yticks(np.arange(0, 0.3, 0.1), labels=[f'{np.abs(i)}%' for i in range(0, 30, 10)], fontsize=10)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=20)

plt.ylim(y.min() - 0.006, y.max() + 0.02)

plt.xlim(0.3, args.num_layers+0.7)
# bottom = -y
# y = y * 2
# plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.bar(x, y, width=1.02, color=model_color_dict[args.model_name])

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
plt.savefig(os.path.join(fig_dir, f'{args.model_name}_kneurons_distribution.pdf'), dpi=100)
plt.close()
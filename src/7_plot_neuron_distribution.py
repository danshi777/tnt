import json
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter
import copy
import argparse
from trl.trainer.utils import pos_list2str, pos_str2list
# trl/trainer/utils.py
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str)
parser.add_argument("--neuron_file",
                    type=str,
                    help="Flie to be plotted.")
parser.add_argument("--num_layers",
                    type=int,
                    help="Number of layers in the model.")
parser.add_argument("--neuron_num",
                    type=int,
                    help="Number of neurons to be plotted.")
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
neurons = []
neuron_counter = Counter()
for neuron_bag in neuron_bag_list:
    for neuron in neuron_bag:
        neuron_counter.update([pos_list2str(neuron[:2])])
most_common_neuron = neuron_counter.most_common(args.neuron_num)
neurons = [pos_str2list(ne_str[0]) for ne_str in most_common_neuron]

# neurons，如: [[3, 9217], [0, 13112], [8, 13122], [7, 13705], [7, 2338], [4, 7622], [3, 4584], [17, 7401], [2, 12812], [13, 8216], [1, 279], [0, 15622], [2, 16355], [6, 16325], [7, 7293], [16, 2807], [0, 13912], [17, 15726], [6, 5944], [17, 7178]]
tot_num = len(neurons)

neuron_counter_per_layer = Counter()
for n in neurons:
    neuron_counter_per_layer.update([n[0]]) # 计数：哪层有几个神经元

print(neuron_counter_per_layer)
ori_neuron_counter_per_layer = copy.deepcopy(neuron_counter_per_layer)
# neuron_counter_per_layer:
# 层: 该层的神经元个数
for k, v in neuron_counter_per_layer.items():
    neuron_counter_per_layer[k] /= tot_num
# neuron_counter_per_layer:
# 层: 该层的神经元个数/总神经元个数

print('total # neurons:', tot_num)

plt.figure(figsize=(8.5, 3))

x = np.array([i + 1 for i in range(args.num_layers)])
y = np.array([neuron_counter_per_layer[i] for i in range(args.num_layers)])
# plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i for i in range(4, args.num_layers+1, 4)], labels=[i for i in range(4, args.num_layers+1, 4)], fontsize=20)
# plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=10)
plt.yticks(np.arange(0, 0.9, 0.1), labels=[f'{np.abs(i)}%' for i in range(0, 90, 10)], fontsize=10)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=20)

plt.ylim(y.min() - 0.006, y.max() + 0.05)

plt.xlim(0.3, args.num_layers+0.7)
# bottom = -y
# y = y * 2
# plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)

# 绘制柱状图
bars = plt.bar(x, y, width=1.02, color=model_color_dict[args.model_name])
# 在每个柱子上方添加数值
for bar, layer in zip(bars, range(args.num_layers)):
    height = bar.get_height()  # 获取柱子的高度
    if layer in ori_neuron_counter_per_layer:
        value = ori_neuron_counter_per_layer[layer]
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # 横坐标
            height + 0.005,  # 纵坐标，稍微高于柱子高度
            f'{value}',  # 显示的原始计数值
            ha='center',  # 水平居中
            va='bottom',  # 垂直对齐到底部
            fontsize=10,  # 字体大小
            color='black'  # 字体颜色
        )

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
plt.savefig(os.path.join(fig_dir, f'{args.model_name}_kneurons_distribution.png'), dpi=100)
plt.close()
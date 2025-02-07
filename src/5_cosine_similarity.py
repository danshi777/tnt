import json
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM
from collections import Counter
from trl.trainer.utils import pos_list2str, pos_str2list


def load_neurons(model, adversarial_method_name, model_name, top_n, neuron_num=500):
    neuron_file = f"results/neurons/{adversarial_method_name}_{model_name}_cross_entropy_intermediate_top{top_n}_neuron.json"
    print(f"Load neurons from: {neuron_file}")
    
    with open(neuron_file, "r") as f:
        neuron_bag_list = json.load(f) # neuron_bag_list: 所有example所产生的neurons
    
    neurons = []
    neuron_counter = Counter()
    for neuron_bag in neuron_bag_list: # neuron_bag: 某example所产生的neurons，如[[11, 2891, 0.0002630653325468302], [10, 1845, 0.0002512137289159], [10, 1154, 0.00021400029072538018], ...]
        for neuron in neuron_bag:
            neuron_counter.update([pos_list2str(neuron[:2])])
    most_common_neuron = neuron_counter.most_common(neuron_num)
    neurons = [pos_str2list(ne_str[0]) for ne_str in most_common_neuron] 
    # neurons，如: [[3, 9217], [0, 13112], [8, 13122], [7, 13705], [7, 2338], [4, 7622], [3, 4584], [17, 7401], [2, 12812], [13, 8216], [1, 279], [0, 15622], [2, 16355], [6, 16325], [7, 7293], [16, 2807], [0, 13912], [17, 15726], [6, 5944], [17, 7178]]
    return neurons

def get_neuron_vector(model, layer_id, position_id):
    model.eval()
    # model.model.layers[layer].mlp.down_proj.weight: (hidden_size, intermediate_size)
    return model.model.layers[layer_id].mlp.down_proj.weight[:,position_id] # (hidden_size)

def get_task_matrix(model, neurons):
    neuron_vectors = [get_neuron_vector(model, layer, pos) for layer, pos in neurons]
    # for layer, pos in neurons:
    #     print(layer, pos)
    #     print(get_neuron_vector(model, layer, pos).shape)
    # print(neuron_vectors[0].shape)
    # print(neuron_vectors[499].shape)
    neuron_vectors = torch.stack(neuron_vectors)
    return np.array(neuron_vectors.detach().cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_or_path", type=str, default=50)
    parser.add_argument("--model_name", type=str, default=50)
    # parser.add_argument("--adversarial_method_name", type=str)
    parser.add_argument("--top_n", type=int, default=50)

    args = parser.parse_args()
    return args

def cosine_similarity(A, B):
    # # 归一化
    # A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    # print(A_norm)
    # B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    # # 计算余弦相似度矩阵
    # return np.dot(A_norm, B_norm.T)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.dot(A_norm, B_norm.T)

def overlap_coefficient(set1, set2):
    """计算两个任务神经元集合的 Overlap Coefficient"""
    set1, set2 = set(map(tuple, set1)), set(map(tuple, set2))  # 确保转换为元组
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_id_or_path, low_cpu_mem_usage=True, device_map="auto")
    model.eval()

    adversarial_method_name1 = "DAP"
    adversarial_method_name2 = "DeepInception"
    adversarial_method_name3 = "MultiJail"
    # adversarial_method_name4 = "JailBroken"

    task1_neurons = load_neurons(model, adversarial_method_name1, args.model_name, args.top_n)
    print(task1_neurons)
    task2_neurons = load_neurons(model, adversarial_method_name2, args.model_name, args.top_n)
    print(task2_neurons)
    task3_neurons = load_neurons(model, adversarial_method_name3, args.model_name, args.top_n)
    print(task3_neurons)
    # task4_neurons = load_neurons(model, adversarial_method_name4, args.model_name, args.top_n)

    # 提取每个任务的参数矩阵
    M1 = get_task_matrix(model, task1_neurons)
    M2 = get_task_matrix(model, task2_neurons)
    M3 = get_task_matrix(model, task3_neurons)
    # M4 = get_task_matrix(model, task4_neurons)
    print(M1.shape, M2.shape, M3.shape)

    similarity_12 = cosine_similarity(M1, M2).mean()
    similarity_13 = cosine_similarity(M1, M3).mean()
    similarity_23 = cosine_similarity(M2, M3).mean()
    # similarity_14 = cosine_similarity(M1, M4).mean()
    similarity_33 = np.diag(cosine_similarity(M3, M3)).mean() # np.diag: 只计算对角线元素（即相同神经元之间的相似度）

    print("Task 1 vs Task 2 Cosine Similarity:", similarity_12)
    print("Task 1 vs Task 3 Cosine Similarity:", similarity_13)
    print("Task 2 vs Task 3 Cosine Similarity:", similarity_23)
    # print("Task 1 vs Task 4:", similarity_14)
    print("Task 3 vs Task 3 Cosine Similarity:", similarity_33)



    # 计算任务之间的 Overlap Coefficient
    overlap_12 = overlap_coefficient(task1_neurons, task2_neurons)
    overlap_13 = overlap_coefficient(task1_neurons, task3_neurons)
    overlap_23 = overlap_coefficient(task2_neurons, task3_neurons)

    print("Task 1 vs Task 2 Overlap:", overlap_12)
    print("Task 1 vs Task 3 Overlap:", overlap_13)
    print("Task 2 vs Task 3 Overlap:", overlap_23)

if __name__ == "__main__":
    main()
import torch
import json
import random
import os
from collections import Counter
from itertools import product


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def enhance_neurons(args, model):
    # ======================== calculate and enhance neurons =================================
    if args.do_random_neurons:
        # neurons = []
        # for i in range(args.enhance_neuron_num):
            # layer = random.randint(0, model.config.num_hidden_layers-1)
            # pos = random.randint(0, model.config.intermediate_size-1)
            # neurons.append([layer, pos])
        # 以上生成的有可能重复！故采用下面的方式生成
        # 生成笛卡尔积，即所有可能的 (layer, pos) 组合，
        # 其中layer在[0,model.config.num_hidden_layers-1]之间，layer在[0,model.config.intermediate_size-1]之间
        all_pairs = list(product(range(0, model.config.num_hidden_layers), range(0, model.config.intermediate_size)))
        neurons = random.sample(all_pairs, args.enhance_neuron_num)
        print(f'The number of changed random neurons: {len(neurons)}')
        print(neurons)
    elif args.do_random_last_layer_neurons:
        layer = model.config.num_hidden_layers-1
        # neurons = []
        # for _ in range(to_be_trained_neuron_num):
        #     pos = random.randint(0, model.config.intermediate_size-1)
        #     neurons.append([layer, pos])
        # 以上生成的有可能重复！故采用下面的方式生成
        poses = random.sample(range(0, model.config.intermediate_size), args.enhance_neuron_num)
        neurons = [[layer, pos] for pos in poses]
        print(f'The number of changed random neurons in the last layer: {len(neurons)}')
        print(neurons)
    else: # get important neurons
        # neuron_file = os.path.join(args.cn_dir, f'{args.dataset_name}-{args.model_name}-cn_bag-context.json')
        neuron_file = args.neuron_file
        print(f"Change context neurons in: {neuron_file}")
        with open(neuron_file, 'r') as fr:
            neuron_bag_list = json.load(fr)
        neurons = []
        neuron_counter = Counter()
        # cn_bag_list: 所有example所产生的context keurons 
        # cn_bag: 某example所产生的context keurons，如[[11, 2891, 0.0002630653325468302], [10, 1845, 0.0002512137289159], [10, 1154, 0.00021400029072538018], ...]
        for neuron_bag in neuron_bag_list:  
            for neuron in neuron_bag:
                neuron_counter.update([pos_list2str(neuron[:2])])
        most_common_neuron = neuron_counter.most_common(args.enhance_neuron_num)
        neurons = [pos_str2list(cn_str[0]) for cn_str in most_common_neuron]
        # print('model.config.num_hidden_layers:', model.config.num_hidden_layers)
        print(f'The number of changed important neurons: {len(neurons)}')
        print(most_common_neuron)
        # print('neurons:', neurons)
    
    ### enhance weights of the neurons
    # unk_emb = model.model.embeddings.word_embeddings.weight[100]
    for layer, pos in neurons:
        with torch.no_grad():
            ori_emb = model.model.layers[layer].mlp.down_proj.weight[:, pos]
            # model.model.layers[layer].mlp.down_proj.weight: (hidden_size, intermediate_size) 即(4096, 11008)
            # model.model.layers[layer].mlp.down_proj.weight[:, pos] = args.enhance_strength * ori_emb
            model.model.layers[layer].mlp.down_proj.weight[:, pos] *= args.enhance_strength
    if not args.do_random_neurons and not args.do_random_last_layer_neurons:
        return model, most_common_neuron
    else:
        return model, neurons
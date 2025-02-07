import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from data_utils import prepare_dataset
from model_utils import model_name_to_path_dict


# 对固定的（在破坏神经元时的）activation求梯度
# hook取outp，即hidden_size维的作为神经元


# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# 把wight变为 1/m*wight, 2/m*wight, ..., k/m*weight, ...
def scaled_input_from_zero(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(1, num_points + 1)], dim=0)  # (num_points, ffn_size)
    # return res, step[0]
    return res, num_points


def scaled_input(minimum_weight, maximum_weights, batch_size, num_batch):

    num_points = batch_size * num_batch
    step = (maximum_weights - minimum_weight) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(minimum_weight, step * i) for i in range(1, num_points + 1)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet(ig_list, threshold=0.1):
    ig_triplet = []
    ig = np.array(ig_list)
    max_ig = ig.max() # maximum attribution score
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * threshold:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def process_input(input, response, tokenizer, device, args):
    # input_text = question + safe_generation
    inputs = tokenizer(tokenizer.bos_token + input + "\n\n", add_special_tokens=False)
    responses = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)

    input_ids = torch.tensor(inputs["input_ids"] + responses["input_ids"]).to(device)
    attention_mask = torch.tensor(inputs["attention_mask"] + responses["attention_mask"]).to(device)
    label_ids = torch.tensor([-100] * len(inputs["input_ids"]) + responses["input_ids"]).to(device) # 只计算在response上的梯度

    answer_index = - len(responses["input_ids"]) - 1

    # if len(input_ids) > args.max_length:
    #     input_ids = input_ids[:args.max_length]
    #     attention_mask = attention_mask[:args.max_length]
    #     label_ids = label_ids[:args.max_length]
    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "answer_index": answer_index,
        }


def get_toxic_attr(idx, question, example, args, model, tokenizer, device, min_cosine_value):

    # u_tokenized_inputs = tokenizer(unsafe_generation, return_tensors="pt")
    # u_input_ids = u_tokenized_inputs["input_ids"].to(device)
    # u_attention_mask = u_tokenized_inputs["attention_mask"].to(device)
    
    # s_tokenized_inputs = tokenizer(safe_generation, return_tensors="pt")
    # s_input_ids = s_tokenized_inputs["input_ids"].to(device)
    # s_attention_mask = s_tokenized_inputs["attention_mask"].to(device)

    q_tokenized_inputs = tokenizer(question, return_tensors="pt") # 会在最前面加<bos> token
    q_input_ids = q_tokenized_inputs["input_ids"].to(device)
    q_attention_mask = q_tokenized_inputs["attention_mask"].to(device)
    
    # generate a response
    output = model.generate(
        input_ids=q_input_ids,
        num_beams=1,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        # output_hidden_states=True,  # 确保生成时保留隐藏状态，确保hook得到整个序列最后一层的hidden states
        # return_dict_in_generate=True,  # 返回结构化信息
    )
    ori_response = tokenizer.decode(output[0, q_input_ids.shape[-1]:], skip_special_tokens=True)
    # ori_response = tokenizer.batch_decode(output[:, q_input_ids.shape[-1]:], skip_special_tokens=True)[0]
    
    # record results
    res_dict = {
        'id': idx,
        'unsafe_all_ffn_activations': [],
        'safe_all_ffn_activations': [],
        'all_attr_score': [],
    }

    # input_text = f"Q: {question} A: {unsafe_generation}"
    input_text = f"{question}{ori_response}"
    if idx < 1:
        print("="*100)
        print(input_text)
        print("="*100)
        print(f"Use {args.criterion} as criterion")
    
    if args.criterion == "cosine_similarity":
        qa_tokenized_inputs = tokenizer(input_text, return_tensors="pt")
        qa_input_ids = qa_tokenized_inputs["input_ids"][:, :-1].to(device)
        
        # hook取inp，即intermediate_size维的作为神经元
        # hook取outp，即hidden_size维的作为神经元
        ref_last_hidden_states_dict = dict()
        if args.get_intermediate_size_or_hidden_size == "intermediate":
            def get_ref_hidden_state_forward_hook(module, inp, outp):
                ref_last_hidden_states_dict["input"] = inp[0]
        if args.get_intermediate_size_or_hidden_size == "hidden":
            def get_ref_hidden_state_forward_hook(module, inp, outp):
                ref_last_hidden_states_dict["output"] = outp
        last_layer_idx = model.model.config.num_hidden_layers - 1
        handel_forward_last_layer = model.model.layers[last_layer_idx].register_forward_hook(get_ref_hidden_state_forward_hook)
        outputs = model(input_ids=qa_input_ids) 
        handel_forward_last_layer.remove()
    # original pred prob
    # if args.get_pred:
    #     u_outputs = model(input_ids=u_input_ids, attention_mask=u_attention_mask)  # (batch, n_vocab)
    #     u_logits = u_outputs.logits[:, -1, :] # (batch,, n_vocab)
    #     u_base_pred_prob = F.softmax(u_logits, dim=1)  # (batch,, n_vocab)
    #     res_dict['u_pred'].append(u_base_pred_prob.tolist())

    #     s_outputs = model(input_ids=s_input_ids, attention_mask=s_attention_mask)
    #     s_logits = s_outputs.logits[:, -1, :]
    #     s_base_pred_prob = F.softmax(s_logits, dim=1)
    #     res_dict['s_pred'].append(s_base_pred_prob.tolist())
        
    for tgt_layer in range(model.model.config.num_hidden_layers):
        # print("example idx:", idx)
        # print("tgt_layer:", tgt_layer)
        
        ffn_activations_dict = dict()
        
        # def u_forward_hook_fn(module, inp, outp):
        #     ffn_activations_dict['u_input'] = inp[0]  # inp type is Tuple

        # def s_forward_hook_fn(module, inp, outp):
        #     ffn_activations_dict['s_input'] = inp[0]

        if args.get_intermediate_size_or_hidden_size == "intermediate":
            def q_forward_hook_fn(module, inp, outp):
                ffn_activations_dict['q_input'] = inp[0]
        if args.get_intermediate_size_or_hidden_size == "hidden":
            def q_forward_hook_fn(module, inp, outp):
                ffn_activations_dict['q_output'] = outp
    

        # # ========================== get hidden states of unsafe generation =========================
        # u_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(u_forward_hook_fn)
        # with torch.no_grad():
        #     u_outputs = model(input_ids=u_input_ids, attention_mask=u_attention_mask)
        # u_ffn_activations = ffn_activations_dict['u_input']
        # u_ffn_activations = u_ffn_activations[:, -1, :]
        # u_logits = u_outputs.logits[:, -1, :]
        # u_hook.remove()
        
        # # =========================== get hidden states of safe generation ============================
        # s_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(s_forward_hook_fn)
        # with torch.no_grad():
        #     s_outputs = model(input_ids=s_input_ids, attention_mask=s_attention_mask)
        # s_ffn_activations = ffn_activations_dict['s_input']
        # s_ffn_activations = s_ffn_activations[:, -1, :]
        # s_logits = s_outputs.logits[:, -1, :]
        # s_hook.remove()
        
        # =========================== get hidden states of question ============================
        q_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(q_forward_hook_fn)
        with torch.no_grad():
            q_outputs = model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        
        if args.get_intermediate_size_or_hidden_size == "intermediate":
            q_ffn_activations = ffn_activations_dict['q_input']
        if args.get_intermediate_size_or_hidden_size == "hidden":
            q_ffn_activations = ffn_activations_dict['q_output']
        q_ffn_activations = q_ffn_activations[:, -1, :]
        q_logits = q_outputs.logits[:, -1, :]
        q_hook.remove()
        
        # scaled_activations, activations_step = scaled_input(u_ffn_activations, s_ffn_activations, args.batch_size, args.num_batch)
        scaled_activations, num_points = scaled_input_from_zero(q_ffn_activations, args.batch_size, args.num_batch)
        scaled_activations.requires_grad_(True)


        if args.get_intermediate_size_or_hidden_size == "intermediate":
            def forward_hook_fn(module, inp, outp):
                # 在前向传播时调用
                forward_cache.append(inp[0])

            def backward_hook_fn(module, grad_input, grad_output):
                # 在反向传播时调用
                backward_cache.append(grad_input[0])
        if args.get_intermediate_size_or_hidden_size == "hidden":
            def forward_hook_fn(module, inp, outp):
                # 在前向传播时调用
                forward_cache.append(outp)

            def backward_hook_fn(module, grad_input, grad_output):
                # 在反向传播时调用
                backward_cache.append(grad_output[0])

        # integrated grad at the gold label for each layer
        attribution_score = None
        for batch_idx in range(args.num_batch):       
            grad = None
            all_importance = None
            forward_cache = []
            backward_cache = []
            for i in range(0, args.batch_size, args.batch_size_per_inference):
                # print(i, i + args.batch_size_per_inference)
                batch_activations = scaled_activations[i: i + args.batch_size_per_inference] # (batch_size_per_inference, intermediate_size) for "intermediate" setting, (batch_size_per_inference, hidden_size) for "hidden" setting,
                # batch_s_activations = s_ffn_activations.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, ffn_size)
                # batched_s_input_ids = s_input_ids.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                # batched_s_attention_mask = s_attention_mask.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                    
                # inputs = process_input(input=question, response=unsafe_generation, tokenizer=tokenizer, device=device, args=args)
                inputs = tokenizer(input_text, return_tensors="pt") # 会在最前面加<bos> token
                # label是input向后shift一位
                input_ids = inputs["input_ids"][:, 0:-1]
                labels = inputs["input_ids"][:, 1:]

                batched_input_ids = input_ids.repeat(args.batch_size_per_inference, 1).to(device) # (batch_size_per_inference, seq_len)
                batched_labels = labels.repeat(args.batch_size_per_inference, 1).to(device) # (batch_size_per_inference, seq_len)
                # batched_labels = inputs["labels"][0].repeat(args.batch_size_per_inference, 1).to(device) # (batch_size_per_inference, seq_len)
                
                golden_token_ids = tokenizer(ori_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device) # (1, answer_len)
                # 由于input_ids少了一位，因此计算时golden_token_ids也少一位，又由于本来梯度的计算要向前一位，即为 -(golden_token_ids.shape[1]-1) -1
                answer_index = - golden_token_ids.shape[1]
     
                if args.get_intermediate_size_or_hidden_size == "intermediate":
                    def i_forward_hook_change_fn(module, inp):
                        inp0 = inp[0]
                        inp0[:, answer_index, :] = batch_activations
                        inp = tuple([inp0])
                        return inp
                    handle_change = model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(i_forward_hook_change_fn) # 在模块的forward函数执行前被调用
                
                if args.get_intermediate_size_or_hidden_size == "hidden":
                    def i_forward_hook_change_fn(module, inp, outp):
                        outp[:, answer_index, :] = batch_activations
                        return outp
                    handle_change = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(i_forward_hook_change_fn) # 在模块的forward过程中执行
                    
                handle_forward = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(forward_hook_fn)
                handle_backward = model.model.layers[tgt_layer].mlp.down_proj.register_backward_hook(backward_hook_fn)
            
                if args.criterion == "cross_entropy":
                    with torch.autograd.set_detect_anomaly(True):
                        outputs = model(input_ids=batched_input_ids, labels=batched_labels)  # (batch, n_vocab), (batch, ffn_size)
                        logits = outputs.logits

                        criterion = torch.nn.CrossEntropyLoss(reduction='none')
                        # 调整 logits 和 labels 的形状
                        logits_flat = logits.view(-1, logits.shape[-1])  # (batch_size_per_inference * seq_length, vocab_size)
                        labels_flat = batched_labels.view(-1)  # (batch_size_per_inference * seq_length)
                        losses_per_position = criterion(logits_flat, labels_flat) # (batch_size_per_inference * seq_length)
                        # 恢复到 (batch_size, seq_length) 的形状
                        losses_per_position = losses_per_position.view(args.batch_size_per_inference, -1) # (batch_size_per_inference, seq_length)
                        # 只取在response上的loss，shape变为(batch_size_per_inference, response_len)
                        losses_per_position = losses_per_position[:, answer_index:]
                        # 计算每个样本的总损失（如对序列中的位置求平均）
                        loss_per_sample = losses_per_position.mean(dim=1)  # (batch_size_per_inference)
                    del outputs, logits_flat, labels_flat, losses_per_position
                    torch.cuda.empty_cache()
                
                if args.criterion == "cosine_similarity":
                    gen_last_hidden_states_dict = dict()
                    if args.get_intermediate_size_or_hidden_size == "intermediate":
                        def get_gen_hidden_state_forward_hook(module, inp, outp):
                            gen_last_hidden_states_dict["input"] = outp
                        handel_i_forward_last_layer = model.model.layers[last_layer_idx].register_forward_hook(get_gen_hidden_state_forward_hook)
                        outputs = model(input_ids=batched_input_ids)
                        handel_i_forward_last_layer.remove()
                        
                        ref = ref_last_hidden_states_dict["input"][0][:, answer_index:] # (1, response_len, intermediate_size or hidden_size)
                        ref = ref.expand(args.batch_size_per_inference, -1, -1) # (batch_size_per_inference, response_len, intermediate_size or hidden_size)
                        gen = gen_last_hidden_states_dict["input"][0][:, answer_index:] # (batch_size_per_inference, response_len, intermediate_size or hidden_size)
             
                    if args.get_intermediate_size_or_hidden_size == "hidden":
                        def get_gen_hidden_state_forward_hook(module, inp, outp):
                            gen_last_hidden_states_dict["output"] = outp
                        handel_i_forward_last_layer = model.model.layers[last_layer_idx].register_forward_hook(get_gen_hidden_state_forward_hook)
                        outputs = model(input_ids=batched_input_ids)
                        handel_i_forward_last_layer.remove()
                        
                        ref = ref_last_hidden_states_dict["output"][0][:, answer_index:] # (1, response_len, intermediate_size or hidden_size)
                        ref = ref.expand(args.batch_size_per_inference, -1, -1) # (batch_size_per_inference, response_len, intermediate_size or hidden_size)
                        gen = gen_last_hidden_states_dict["output"][0][:, answer_index:] # (batch_size_per_inference, response_len, intermediate_size or hidden_size)
                        
                    cosine_similarity_per_position = F.cosine_similarity(ref, gen, dim=-1) # (batch_size_per_inference, response_len)
                    if torch.min(cosine_similarity_per_position) < min_cosine_value:
                        min_cosine_value = torch.min(cosine_similarity_per_position)
                    # 计算每个样本的总损失（如对序列中的位置求平均）
                    cosine_similarity_per_sample = cosine_similarity_per_position.mean(dim=-1) # (batch_size_per_inference)
                    loss_per_sample = -cosine_similarity_per_sample  # (batch_size_per_inference)
                    del outputs, ref, gen, cosine_similarity_per_position
                    torch.cuda.empty_cache()

                # 逐样本计算 loss 相对于 batch_activations 的梯度
                grads = []
                for j in range(args.batch_size_per_inference):
                    grad = torch.autograd.grad(
                        outputs=loss_per_sample[j],   # 单个样本的损失
                        inputs=batch_activations,    # 对整个 batch_activations 求导
                        retain_graph=True
                    )[0] # 得到的grad是一个tuple，所以[0]  
                    # 上面得到的 grad 是一个张量，形状与 batch_activations 相同
                    # 每次提取梯度张量中的第 j 行（grad[j]），即 loss_per_sample[j] 对 batch_activations[j] 的梯度。
                    grads.append(grad[j])
                
                # 堆叠为一个张量 (batch_size_per_inference, hidden_size) 或 (batch_size_per_inference, intermediate_size)
                grad_to_be_mul = torch.stack(grads)
                # batch_activations: (batch_size_per_inference, hidden_size)
                activation_to_be_mul = batch_activations
                importance_matrix = (activation_to_be_mul * grad_to_be_mul).abs()

                # shape变为(batch_size_per_inference, response_len, hidden_size)
                # importance_matrix = importance_matrix[:, answer_index:, :]

                # 按照seq_len维度取平均值，shape变为(batch_size_per_inference, hidden_size)
                # importance_matrix = importance_matrix.mean(dim=1)

                all_importance = importance_matrix if all_importance is None else torch.cat((all_importance, importance_matrix), dim=0)

                handle_change.remove()
                handle_forward.remove()
                handle_backward.remove()
                del grads, grad_to_be_mul, activation_to_be_mul
                torch.cuda.empty_cache()
            
            # all_importance: (num_points, intermediate_size) or (num_points, hidden_size)
            attr_score = all_importance.sum(dim=0) # (intermediate_size) or (hidden_size)
            attr_score = attr_score * num_points # (intermediate_size) or (hidden_size)
        
        attribution_score = attr_score if attribution_score is None else torch.add(attribution_score, attr_score) # (hidden_size)

        # res_dict['all_attr_score'].append(attribution_score.tolist()) # (intermediate_size) or (hidden_size)
        # 将 attribution_score 转为 CPU 数据并释放 GPU 显存
        cpu_attr_score = attribution_score.cpu().tolist()  # 转移到 CPU
        res_dict['all_attr_score'].append(cpu_attr_score)  # 只存储到 CPU 的 res_dict

        # 删除 GPU 张量，释放显存
        del attribution_score
        torch.cuda.empty_cache()  # 清空缓存以释放显存

    return res_dict, min_cosine_value


def analysis_attribution_file(filename, args, metric='all_attr_score'):
    print(f'===========> parsing important neurons in {os.path.join(args.output_attribution_dir, filename)}')
    rlts_bag = []
    with open(os.path.join(args.output_attribution_dir, filename), 'r') as fr:
        for idx, line in enumerate(tqdm(fr.readlines())):
            try:
                example = json.loads(line)
                rlts_bag.append(example)
            except Exception as e:
                print(f"Exception {e} happend in line {idx}")

    print(f"Total examples: {len(rlts_bag)}")
    
    ave_neuron_num = 0
    neuron_bag_list = []

    for rlt in rlts_bag:
        metric_triplets = rlt[metric]
        metric_triplets.sort(key=lambda x: x[2], reverse=True)
        cn_bag = metric_triplets[:args.top_n]
        neuron_bag_list.append(cn_bag)

    for neuron_bag in neuron_bag_list:
        ave_neuron_num += len(neuron_bag)
    ave_neuron_num /= len(neuron_bag_list)

    return ave_neuron_num, neuron_bag_list


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    # parser.add_argument("--data_path",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data path. Should be .json file. ")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Model name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_id_or_path", 
                        default=None, 
                        type=str,
                        help="Path to local pretrained model or model identifier from huggingface.co/models or local.")
    # parser.add_argument("--dataset_name",
    #                     type=str,
    #                     default=None,
    #                     help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--adversarial_method_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    
    parser.add_argument("--output_attribution_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the attribution scores will be written.")
    parser.add_argument("--output_neuron_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the identified neurons will be written.")

    # Other parameters
    # parser.add_argument("--max_seq_length",
    #                     default=128,
    #                     type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                         "Sequences longer than this will be truncated, and sequences shorter \n"
    #                         "than this will be padded.")
    parser.add_argument("--max_new_tokens",
                        default=300,
                        type=int,
                        help="The maximum number of tokens generated by the `generate()` method.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_id",
                        type=str,
                        default='0',
                        help="available gpu id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # parameters about integrated grad
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--num_of_examples",
                        default=500,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")
    parser.add_argument("--batch_size_per_inference",
                        default=2,
                        type=int,
                        choices=[1,2,4,5,10,20],
                        help="The batch size for each inference, you can choose an appropriate value from 1, 2, 4, 5, 10, 20 according to the model size and CUDA memory.")
    parser.add_argument("--criterion",
                        default=None,
                        required=True,
                        choices=["cross_entropy", "cosine_similarity"],
                        type=str,
                        help="The criterion")
    parser.add_argument("--get_intermediate_size_or_hidden_size",
                        default=None,
                        required=True,
                        choices=["intermediate", "hidden"],
                        type=str,
                        help="Take intermediate dim as neurons or take hidden dim as neurons")
    parser.add_argument("--add_chat_template",
                        default=False,
                        help="Set this flag if you want to use chat template.")
    
    parser.add_argument("--only_process_neurons",
                        action='store_true',
                        help="Whether to process the attribution score and identify neurons")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.05)
    parser.add_argument("--top_n",
                        type=int,
                        default=20)

    # parse arguments
    args = parser.parse_args()

    args.model_id_or_path = model_name_to_path_dict[args.model_name]
    if args.adversarial_method_name in ["Jailbroken", "SafeEditWithAdv"]:
        args.add_chat_template = False
    else:
        args.add_chat_template = True

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # TODO: To implement multi gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))


    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    output_prefix = f"{args.adversarial_method_name}_{args.model_name}_{args.criterion}_{args.get_intermediate_size_or_hidden_size}_top{args.top_n}"
    os.makedirs(args.output_attribution_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_attribution_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    ############################# 开始计算梯度归因分数 #############################
    if not args.only_process_neurons:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path)

        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(args.model_id_or_path, low_cpu_mem_usage=True, device_map=device)
        
        # if "llama" in args.model_name.lower():
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     model.resize_token_embeddings(len(tokenizer))
        
        model.eval()

        # ============================== prepare dataset ==============================
        data = prepare_dataset(args, tokenizer, "train")
        data = data[:args.num_of_examples]
        # from datasets import load_dataset
        # datafiles = {
        #     'train': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_train.json',
        #     'test': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_test.json',
        #     'validation': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_val.json'
        # }
        # dataset = load_dataset('json', data_files=datafiles, cache_dir='data/')
        
        tic = time.perf_counter()
        count = 0
        print(f"Start to calculate the attribution score, dataset size: {len(data)}")
        min_cosine_value_all = 100
        with open(os.path.join(args.output_attribution_dir, output_prefix + '.rlt.json'), 'w') as fw:
            for idx, example in enumerate(tqdm(data)):
                # adversarial_prompt = example["adversarial prompt"]
                # unsafety_category = example["unsafety category"]
                question = example["question"]
                # safe_generation = example["safe generation"]
                # unsafe_generation = example["unsafe generation"]
                # unsafe_generation = example["unsafe generation"].split(": ")[-1]
                
                if idx == 0:
                    # print(safe_generation)
                    # print(unsafe_generation)
                    prompt_dict = {
                        "question": question,
                        # "safe_generation" : safe_generation,
                        # "unsafe_generation" : unsafe_generation,
                    }
                    json.dump(prompt_dict, open(os.path.join(args.output_attribution_dir, output_prefix + '.prompt.json'), 'w'), indent=2)
                
                res_dict, min_cosine_value = get_toxic_attr(idx, question, example, args, model, tokenizer, device, min_cosine_value=99)
                if min_cosine_value < min_cosine_value_all:
                    min_cosine_value_all = min_cosine_value
                # if args.criterion == "cross_entropy":
                #     res_dict["all_attr_score"] = convert_to_triplet(res_dict["all_attr_score"], 0.05)
                # if args.criterion == "cosine_similarity":
                #     res_dict["all_attr_score"] = convert_to_triplet(res_dict["all_attr_score"], 0.01)
                res_dict["all_attr_score"] = convert_to_triplet(res_dict["all_attr_score"], args.threshold)

                fw.write(json.dumps(res_dict) + "\n")
                count += 1
                res_dict['all_attr_score'] = []
                torch.cuda.empty_cache()
        print(f"min_cosine_value_all: {min_cosine_value_all}")
        print(f"Saved in {os.path.join(args.output_attribution_dir, output_prefix + '.rlt.json')}")

        toc = time.perf_counter()
        time_str = f"***** Costing time: {toc - tic:0.4f} seconds *****"
        print(time_str)
        json.dump(time_str, open(os.path.join(args.output_attribution_dir, output_prefix + '.time.json'), 'w'))


    ############################# 开始处理并识别重要神经元 #############################
    print(f"Start to process and identify neurons.")
    if not os.path.exists(args.output_neuron_dir):
        os.makedirs(args.output_neuron_dir)
        
    filename = output_prefix + '.rlt.json'
    ave_neuron_num, neuron_bag_list = analysis_attribution_file(filename, args)
    
    output_prefix = f"{args.adversarial_method_name}_{args.model_name}_{args.criterion}_{args.get_intermediate_size_or_hidden_size}_top{args.top_n}"

    stat_str = f'{len(data)} examples in total.\nOn average, {ave_neuron_num} important neurons were identified per example.'
    print(stat_str)
    json.dump(stat_str, open(os.path.join(args.output_neuron_dir, output_prefix + '_statistics.json'), 'w'))

    output_neuron_path = os.path.join(args.output_neuron_dir, output_prefix + '_neuron.json')
    with open(output_neuron_path, 'w') as fw:
        json.dump(neuron_bag_list, fw, indent=2)
    print(f"Neurons have been saved in {output_neuron_path}")


if __name__ == "__main__":
    main()
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


# baseline hidden states: unsafe generation
# target hidden states: safe generation
# 在输入为：f"Q: {question} A: "时
# 一次推理得出“safe_generation”的每个位置的梯度，再平均


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
    return res, step[0]


def scaled_input(minimum_weight, maximum_weights, batch_size, num_batch):

    num_points = batch_size * num_batch
    step = (maximum_weights - minimum_weight) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(minimum_weight, step * i) for i in range(1, num_points + 1)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)
    max_ig = ig.max() # maximum attribution score
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet

def process_input(input, response, tokenizer, args):
    # input_text = question + safe_generation
    inputs = tokenizer(input)
    responses = tokenizer(response + tokenizer.eos_token)

    input_ids = torch.tensor(inputs["input_ids"] + responses["input_ids"]).cuda()
    attention_mask = torch.tensor(input["attention_mask"] + responses["attention_mask"]).cuda()
    label_ids = torch.tensor([-100] * len(inputs["input_ids"]) + responses["input_ids"]).cuda() # 只计算在response上的梯度

    # if len(input_ids) > args.max_length:
    #     input_ids = input_ids[:args.max_length]
    #     attention_mask = attention_mask[:args.max_length]
    #     label_ids = label_ids[:args.max_length]
    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def get_toxic_attr(idx, question, safe_generation, unsafe_generation, example, args, model, tokenizer, device):
    # # llama2-7b-chat: '_A'
    # # llama3-8b-instruct: 'ĠA'
    # # gemma-2b-it 和 llama2-13b-chat: 'A'
    # if args.model_name == 'llama2-7b-chat':
    #     # gold_label_id = tokenizer.convert_tokens_to_ids("▁" + answer_obj)
    #     tokens = tokenizer.tokenize(answer_obj) # ['▁A']
    #     gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])
    # elif args.model_name == 'llama3-8b-instruct':
    #     # gold_label_id = tokenizer.convert_tokens_to_ids("Ġ" + answer_obj)
    #     tokens = tokenizer.tokenize(f" {answer_obj}") # ['ĠA']
    #     gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])
    # else:
    #     gold_label_id = tokenizer.convert_tokens_to_ids(answer_obj) # ['A']

    u_tokenized_inputs = tokenizer(unsafe_generation, return_tensors="pt")
    u_input_ids = u_tokenized_inputs["input_ids"].to(device)
    u_attention_mask = u_tokenized_inputs["attention_mask"].to(device)
    
    s_tokenized_inputs = tokenizer(safe_generation, return_tensors="pt")
    s_input_ids = s_tokenized_inputs["input_ids"].to(device)
    s_attention_mask = s_tokenized_inputs["attention_mask"].to(device)

    q_tokenized_inputs = tokenizer(question, return_tensors="pt")
    q_input_ids = q_tokenized_inputs["input_ids"].to(device)
    q_attention_mask = q_tokenized_inputs["attention_mask"].to(device)
    
    # record results
    res_dict = {
        'id': idx,
        'u_pred': [],
        's_pred': [],
        'u_all_ffn_activations': [],
        's_all_ffn_activations': [],
        'all_attr_gold': [],
    }
    
    # original pred prob
    if args.get_pred:
        u_outputs = model(input_ids=u_input_ids, attention_mask=u_attention_mask)  # (batch, n_vocab)
        u_logits = u_outputs.logits[:, -1, :] # (batch,, n_vocab)
        u_base_pred_prob = F.softmax(u_logits, dim=1)  # (batch,, n_vocab)
        res_dict['u_pred'].append(u_base_pred_prob.tolist())

        s_outputs = model(input_ids=s_input_ids, attention_mask=s_attention_mask)
        s_logits = s_outputs.logits[:, -1, :]
        s_base_pred_prob = F.softmax(s_logits, dim=1)
        res_dict['s_pred'].append(s_base_pred_prob.tolist())

    for tgt_layer in range(model.model.config.num_hidden_layers):
        ffn_activations_dict = dict()
        
        def u_forward_hook_fn(module, inp, outp):
            ffn_activations_dict['u_input'] = inp[0]  # inp type is Tuple

        def s_forward_hook_fn(module, inp, outp):
            ffn_activations_dict['s_input'] = inp[0]


        def q_forward_hook_fn(module, inp, outp):
            ffn_activations_dict['q_input'] = outp
    

        # ========================== get hidden states of unsafe generation =========================
        u_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(u_forward_hook_fn)
        with torch.no_grad():
            u_outputs = model(input_ids=u_input_ids, attention_mask=u_attention_mask)
        u_ffn_activations = ffn_activations_dict['u_input']
        u_ffn_activations = u_ffn_activations[:, -1, :]
        u_logits = u_outputs.logits[:, -1, :]
        u_hook.remove()
        
        # =========================== get hidden states of safe generation ============================
        s_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(s_forward_hook_fn)
        with torch.no_grad():
            s_outputs = model(input_ids=s_input_ids, attention_mask=s_attention_mask)
        s_ffn_activations = ffn_activations_dict['s_input']
        s_ffn_activations = s_ffn_activations[:, -1, :]
        s_logits = s_outputs.logits[:, -1, :]
        s_hook.remove()
        
        # =========================== get hidden states of question ============================
        q_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(q_forward_hook_fn)
        with torch.no_grad():
            q_outputs = model(input_ids=q_input_ids, attention_mask=q_attention_mask)
        q_ffn_activations = ffn_activations_dict['q_input']
        q_ffn_activations = q_ffn_activations[:, -1, :]
        q_logits = q_outputs.logits[:, -1, :]
        q_hook.remove()
        
        # scaled_activations, activations_step = scaled_input(u_ffn_activations, s_ffn_activations, args.batch_size, args.num_batch)
        scaled_activations, activations_step = scaled_input_from_zero(q_ffn_activations, args.batch_size, args.num_batch)
        # scaled_activations.requires_grad_(True)

        def forward_hook_fn(module, inp, outp):
            # 在前向传播时调用
            forward_cache.append(outp)

        def backward_hook_fn(module, grad_input, grad_output, module_name):
            # 在反向传播时调用
            backward_cache.append(grad_output[0])

        # integrated grad at the gold label for each layer
        ig_gold = None
        for batch_idx in range(args.num_batch):       
            grad = None
            all_grads = None
            forward_cache = []
            backward_cache = []
            for i in range(0, args.batch_size, args.batch_size_per_inference):
                # print(i, i + args.batch_size_per_inference)
                batch_activations = scaled_activations[i: i + args.batch_size_per_inference] # (batch_size_per_inference, ffn_size)
                # batch_s_activations = s_ffn_activations.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, ffn_size)
                # batched_s_input_ids = s_input_ids.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                # batched_s_attention_mask = s_attention_mask.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                
                # input_text = f"Q: {question} A: {safe_generation}"
                input_text = question + safe_generation
                # if batch_idx < 1:
                    # print("="*100)
                    # print(input_text)
                
                inputs = process_input(input=question, response=safe_generation, tokenizer=tokenizer, args=args)
    
                batched_input_ids = inputs["input_ids"].repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                batched_attention_mask = inputs["attention_mask"].repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                batched_labels = inputs["labels"].repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                
                # golden_token_ids = tokenizer(safe_generation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device) # (1, answer_len)
                # answer_index = -golden_token_ids.shape[1]-1

                handle_forward = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(forward_hook_fn)
                handle_backward = model.model.layers[tgt_layer].mlp.down_proj.register_backward_hook(backward_hook_fn)
            
                outputs = model(input_ids=batched_input_ids, attention_mask=batched_attention_mask, labels=batched_labels)  # (batch, n_vocab), (batch, ffn_size)
                loss = outputs.loss
                loss.backward()

                # compute final grad at a layer at the safe_generation tokens' positions
                # tgt_logits = outputs.logits[:, -1, :] # (batch_size_per_inference, n_vocab)
                # tgt_probs = outputs.logits
                # backward_cache[len(backward_cache) - k - 1] [1, sent_len, hidden_states]
                tgt_probs = F.softmax(outputs.logits, dim=-1) # (batch_size_per_inference, seq_len, n_vocab)
                tgt_probs = torch.log(tgt_probs)

                answer_tokens_log_probs = tgt_probs[:, answer_index:-1, :] # (batch_size_per_inference, answer_len, n_vocab)
                
                # 拿到answer tokens' ids 的 log_probs
                batched_golden_token_ids = golden_token_ids.repeat(args.batch_size_per_inference, 1)
                # batched_golden_token_ids.unsqueeze(-1): (batch_size_per_inference, answer_len, 1)
                answer_tokens_log_probs = torch.gather(answer_tokens_log_probs, -1, batched_golden_token_ids.unsqueeze(-1)) # (batch_size_per_inference, answer_len, 1)
                # 取完后 answer_tokens_log_probs的shape: (batch_size_per_inference, answer_len, 1)
                
                # sum
                answer_seq_prob = torch.sum(answer_tokens_log_probs, dim=1) # (batch_size_per_inference, 1)
                
                answer_seq_prob = torch.unbind(answer_seq_prob) # tuple, 包含 batch_size_per_inference 个张量，每个张量的形状为 (1)
                # 把seq_len这个维度sum起来
                
                # 相对于s_ffn_activations 计算answer_tokens_log_probs的梯度
                # s_ffn_activations: (1, ffn_size)
                grads_i = torch.autograd.grad(answer_seq_prob, s_ffn_activations, retain_graph=True) # grads_i[0].shape: (1, ffn_size)
                # grads_i[0]是 batch_size_per_inference条 求梯度再求和的结果
                del tgt_probs

                handle_forward.remove()
                handle_backward.remove()
                
                all_grads = grads_i[0] if all_grads is None else torch.cat((all_grads, grads_i[0]), dim=0)
            grad = all_grads.sum(dim=0)  # (ffn_size)
            ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
            
        ig_gold = ig_gold * activations_step  # (ffn_size)
        res_dict['u_all_ffn_activations'].append(u_ffn_activations.squeeze().tolist())
        res_dict['s_all_ffn_activations'].append(s_ffn_activations.squeeze().tolist())
        res_dict['all_attr_gold'].append(ig_gold.tolist())
        
    return res_dict


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file. ")
    parser.add_argument("--model_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models or local.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the results will be written.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name as output prefix to indentify each running of experiment.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
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


    # parse arguments
    args = parser.parse_args()

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

    output_prefix = f"{args.dataset_name}-{args.model_name}-context"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # prepare dataset
    with open(args.data_path, "r", encoding="utf-8") as fin:
        lines = fin.read() # 现在是json字符串类型
        data = json.loads(lines)
        # 先read()再loads()   就相当于   load()
    data = data[:200]
    # from datasets import load_dataset
    # datafiles = {
    #     'train': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_train.json',
    #     'test': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_test.json',
    #     'validation': '/data/dans/projects/IRCAN-Safety2/data/datasets--zjunlp--SafeEdit/snapshots/a88b7689a3744873805077e04edd1c19219fc7bf/SafeEdit_val.json'
    # }
    # dataset = load_dataset('json', data_files=datafiles, cache_dir='data/')

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, device_map=device)
    
    # if "llama" in args.model_name:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    

    tic = time.perf_counter()
    count = 0
    print(f"Start process, dataset size: {len(data)}")

    with open(os.path.join(args.output_dir, output_prefix + '.rlt.json'), 'w') as fw:
        for idx, example in enumerate(tqdm(data)):
            adversarial_prompt = example["adversarial prompt"]
            unsafety_category = example["unsafety category"]
            question = example["question"]
            safe_generation = example["safe generation"]
            # unsafe_generation = example["unsafe generation"]
            unsafe_generation = example["unsafe generation"].split(": ")[-1]
            
            # if args.model_name == 'llama2-7b-chat':
            #     prompt_without_context = f"Choose the correct option to answer the following question: \n{question}\n{choice_str}Answer: "
            #     prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{question}\n{choice_str}Answer: "
            # if args.model_name == 'llama2-13b-chat':
            #     prompt_without_context = f"Choose the correct option to answer the following question: \n{question}\n{choice_str}Answer: \n"
            #     prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{question}\n{choice_str}Answer: \n"
            # if args.model_name == 'llama3-8b-instruct':
            #     prompt_without_context = f"Choose the correct option to answer the following question: \n{question}\n{choice_str}The correct answer is "
            #     prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{question}\n{choice_str}The correct answer is "
            # if args.model_name == 'gemma-2b-it':
            #     prompt_without_context = f"Choose the correct option to answer the following question: \n{question}\n{choice_str}Answer: \nThe correct option is \n**"
            #     prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{question}\n{choice_str}Answer: \nThe correct option is \n**"
    
            if idx == 0:
                print(safe_generation)
                print(unsafe_generation)
                prompt_dict = {
                    "question": question,
                    "safe_generation" : safe_generation,
                    "unsafe_generation" : unsafe_generation,
                }
                json.dump(prompt_dict, open(os.path.join(args.output_dir, output_prefix + '.prompt.json'), 'w'), indent=2)
            
            res_dict = get_toxic_attr(idx, question, safe_generation, unsafe_generation, example, args, model, tokenizer, device)
            res_dict["all_attr_gold"] = convert_to_triplet_ig(res_dict["all_attr_gold"])
            
            fw.write(res_dict)
            count += 1
    
    print(f"Saved in {os.path.join(args.output_dir, output_prefix + '.rlt.jsonl')}")

    toc = time.perf_counter()
    time_str = f"***** Costing time: {toc - tic:0.4f} seconds *****"
    print(time_str)
    json.dump(time_str, open(os.path.join(args.output_dir, output_prefix + '.time.json'), 'w'))


if __name__ == "__main__":
    main()
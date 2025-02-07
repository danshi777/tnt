import pandas as pd
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

batch_size = 1
model_path = "/data1/dans/models/llama2-7b-chat-hf"
device = torch.device(f"cuda:2")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=True,
    low_cpu_mem_usage=True,
    device_map=device,
)

if tokenizer.pad_token == None:
    print("add_special_tokens")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.eval()

def generate_response(batch_questions):
    inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model.generate(**inputs,
                            num_beams=1,
                            do_sample=False,
                            max_new_tokens=512,
                            pad_token_id=tokenizer.pad_token_id)
    # generation = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    generations = tokenizer.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True) # list
    return generations

all_generations = []

for split in ["train", "test"]:
    file = f"data/AdvBench/harmful_{split}.csv"
    data = pd.read_csv(file)
    data_list = data.values.tolist()
    
    for idx in tqdm(range(0, len(data), batch_size)):
        batch_examples = data[idx:idx+batch_size]
        batch_questions = [example[0] for example in batch_examples]
        all_generations.extend(generate_response(batch_questions))
    
    print(all_generations[:8])
    data['safe generation'] = all_generations
    data.to_csv(f"data/AdvBench/harmful_behaviors_{split}.csv", index=False)
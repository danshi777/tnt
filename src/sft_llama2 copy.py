# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fine-Tune Llama2-7b on SE paired dataset
import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)

from trl import SFTConfig, SFTTrainer, TrlParser
from trl.trainer.utils import get_peft_config, get_neuron_training_config
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """
    model_id_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", 
        metadata={"help": "the model path"}
    )
    data_dir: Optional[str] = field(
        default="/data/dans/projects/trl/trl/data/SafeEdit/train/", 
        metadata={"help": "the location of the dataset"}
    )
    model_name: Optional[str] = field(default="llama2-7b", metadata={"help": "the model name"})
    data_name: Optional[str] = field(default="safeedit", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})

    # Neuron Training Config
    use_neuron_training: Optional[bool] = field(default=False, metadata={"help": "whether to use neuron training"})
    trainable_param_num: Optional[int] = field(default=None, metadata={"help": "number of trainable neurons"})
    neuron_file: Optional[str] = field(default=None, metadata={"help": "The file where the neuron files are stored"})



# def get_stack_exchange_paired(
#     data_dir: str = "/data/dans/projects/trl/trl/data/SafeEdit/",
#     split: str = None,
#     num_proc = None,
# ) -> Dataset:
#     """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

#     The dataset is converted to a dictionary with the following structure:
#     {
#         'prompt': List[str],
#         'chosen': List[str],
#         'rejected': List[str],
#     }

#     Prompts are structured as follows:
#       "Question: " + <prompt> + "\n\nAnswer: "
#     """
#     dataset = load_dataset(
#         "json",
#         # data_files=f"{data_dir}/{split}.json",
#         data_dir=data_dir,
#         split=split,
#         verification_mode="no_checks",
#     )
#     original_columns = dataset.column_names

#     def return_prompt_and_responses(samples):
#         return {
#             # "prompt": [
#             #     "Question: " + question + "\n\nAnswer: "
#             #     for question in samples["question"]
#             # ],
#             "prompt": samples["question"],
#             "chosen": samples["safe generation"],
#             "rejected": [
#                 unsafe_generation.split(": ")[-1]
#                 for unsafe_generation in samples["unsafe generation"]
#             ]
#         }
#     return dataset.map(
#         return_prompt_and_responses,
#         batched=True,
#         num_proc=num_proc,
#         remove_columns=original_columns,
#     )



if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    # parser = HfArgumentParser(ScriptArguments)
    # print(parser._option_string_actions.keys())

    script_args, training_args = parser.parse_args_into_dataclasses()
    # script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(training_args.seed)

    # 1. load a pretrained model
    # load the base model in 4-bit quantization
    # bnb_config = None
    # if script_args.use_bnb:
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id_or_path,
        # quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


    # 2. Load the datasets
    def ori_chars_token_ratio(dataset, tokenizer, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
            text = ori_prepare_sample_text(example)
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens
    
    def chars_token_ratio(dataset, tokenizer, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
            text = prepare_sample_text(example)
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens


    def ori_prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
        return text


    def prepare_sample_text(example):
        # converted_sample = [
        #     {"role": "user", "content": examples["prompt"]},
        #     {"role": "assistant", "content": examples["completion"]},
        # ]
        # return tokenizer.apply_chat_template(converted_sample, tokenize=False)
        """Prepare the text from a sample of the dataset."""
        text = f"Question: {example['question']}\n\nAnswer: {example['safe generation']}"
        return text


    def create_datasets(tokenizer, args):
        ori_dataset = load_dataset(
            "lvwerra/stack-exchange-paired",
            data_dir=args.subset,
            split=args.split,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )
        if args.streaming:
            print("Loading the dataset in streaming mode")
            ori_valid_data = ori_dataset.take(args.size_valid_set)
            ori_train_data = ori_dataset.skip(args.size_valid_set)
            # ori_train_data = ori_train_data.shuffle(buffer_size=args.shuffle_buffer, seed=42)
        else:
            ori_dataset = ori_dataset.train_test_split(test_size=0.005, seed=42)
            ori_train_data = ori_dataset["train"]
            ori_valid_data = ori_dataset["test"]
            print(f"Size of the train set: {len(ori_train_data)}. Size of the validation set: {len(ori_valid_data)}")
        ori_chars_per_token = ori_chars_token_ratio(ori_train_data, tokenizer)
        print(f"The character to token ratio of the dataset is: {ori_chars_per_token:.2f}")


        train_dataset = load_dataset(
            "json",
            data_dir=args.data_dir,
            split="train",
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )
        valid_dataset = load_dataset(
            "json",
            data_dir=args.data_dir,
            split="validation",
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )
        chars_per_token = chars_token_ratio(train_dataset, tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        # print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")

        ori_train_dataset = ConstantLengthDataset(
            tokenizer,
            ori_train_data,
            formatting_func=ori_prepare_sample_text,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=ori_chars_per_token,
        )
        ori_valid_dataset = ConstantLengthDataset(
            tokenizer,
            ori_valid_data,
            formatting_func=ori_prepare_sample_text,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=ori_chars_per_token,
        )

        train_datasetaa = ConstantLengthDataset(
            tokenizer,
            train_dataset,
            formatting_func=prepare_sample_text,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
        )
        valid_datasetaa = ConstantLengthDataset(
            tokenizer,
            valid_dataset,
            formatting_func=prepare_sample_text,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
        )
        a = 0
    train_dataset, eval_dataset = create_datasets(tokenizer, script_args)


    # 4. initialize training arguments:
    peft_config = get_peft_config(script_args)
    neuron_training_config = get_neuron_training_config(script_args)
    # peft_config = LoraConfig(
    #     r=script_args.lora_r,
    #     lora_alpha=script_args.lora_alpha,
    #     lora_dropout=script_args.lora_dropout,
    #     target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    if peft_config is not None:
        # 5. initialize the SFT trainer
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
        )
    elif neuron_training_config is not None:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            neuron_training_config=neuron_training_config,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
        )
    else:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
        )

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    # if script_args.gradient_checkpointing:
    #     raise ValueError("gradient_checkpointing not supported")



    # 6. train
    trainer.train()
    # trainer.save_model(training_args.output_dir)

    # # 7. save
    # output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    # trainer.model.save_pretrained(output_dir)

    # # Free memory for merging weights
    # del base_model
    # if is_torch_xpu_available():
    #     torch.xpu.empty_cache()
    # elif is_torch_npu_available():
    #     torch.npu.empty_cache()
    # else:
    #     torch.cuda.empty_cache()

    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()

    # output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    # model.save_pretrained(output_merged_dir, safe_serialization=True)
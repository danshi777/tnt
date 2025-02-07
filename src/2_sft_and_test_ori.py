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
from typing import Optional, Literal

from accelerate import Accelerator
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
from trl.trainer.utils import get_peft_config, get_neuron_training_config, get_other_methods_training_config
from trl.trainer import ConstantLengthDataset
from data_utils import load_dataset_for_train
from model_utils import model_name_to_path_dict
from test_utils import test

@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """
    model_name: Optional[str] = field(default="llama2-7b", metadata={"help": "the model name"})
    model_id_or_path: Optional[str] = field(default=None, metadata={"help": "the model path"})
    # data_dir: Optional[str] = field(
    #     default="/data/dans/projects/trl/trl/data/SafeEdit/train/", 
    #     metadata={"help": "the location of the dataset"}
    # )
    # data_name: Optional[str] = field(default="safeedit", metadata={"help": "the dataset name"})
    adversarial_method_name: Optional[str] = field(default="SafeEditWithAdv", metadata={"help": "the dataset name"})

    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "whether to use BitsAndBytes"})
    num_of_examples: Optional[int] = field(default=None, metadata={"help": "the size of the actual train set, if `None`, take all the examples as train set; else, take top `num_of_examples` examples as train set."})

    # LoraConfig
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})

    # Neuron Training Config
    use_neuron_training: Optional[bool] = field(default=False, metadata={"help": "whether to use neuron training"})
    trainable_param_num: Optional[int] = field(default=None, metadata={"help": "number of trainable neurons"})
    neuron_file: Optional[str] = field(default=None, metadata={"help": "The file where the neuron files are stored"})
    get_intermediate_size_or_hidden_size: Optional[Literal["intermediate", "hidden"]] = field(default=None, metadata={"help": "Take intermediate dim as neurons or take hidden dim as neurons."})

    # Random Neurons Training Config
    use_random_training: Optional[bool] = field(default=False, metadata={"help": "whether to train random neurons"})
    
    # Random Neurons of the Last Layer Training Config
    use_random_last_layer_training: Optional[bool] = field(default=False, metadata={"help": "whether to train random neurons in the last layer"})
    
    # Arguments for the test process
    few_data_for_debug: Optional[bool] = field(default=False, metadata={"help": "When debugging, only take a small amount of data."})
    generate_batch_size: Optional[int] = field(default=2, metadata={"help": "batch size of generation."})
    eval_safety_batch_size: Optional[int] = field(default=10, metadata={"help": "batch size of evaluation of safety."})
    safety_classifier_path: Optional[str] = field(default="/data/dans/projects/IRCAN-Safety2/SafeEdit-Safety-Classifier", metadata={"help": "Path to the trained safety classifier model."})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "The maximum number of tokens generated by the `generate()` method."})
    generation_dir: Optional[str] = field(default=None, metadata={"help": "The output directory where the model predictions will be written."})
    metric_dir: Optional[str] = field(default=None, metadata={"help": "The output directory where the implementation metrics will be written."})
    suffix_for_output: Optional[str] = field(default=None, metadata={"help": "The output suffix, which is used to form the output file name."})



if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    # parser = HfArgumentParser(ScriptArguments)
    # print(parser._option_string_actions.keys())

    script_args, training_args = parser.parse_args_into_dataclasses()
    # script_args = parser.parse_args_into_dataclasses()[0]
    
    script_args.model_id_or_path = model_name_to_path_dict[script_args.model_name]

    set_seed(training_args.seed)

    # 1. load a pretrained model
    # load the base model in 4-bit quantization
    bnb_config = None
    if script_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id_or_path,
        # quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    if "llama" in script_args.model_name.lower():
        print('Add pad token for llama models.')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_model.resize_token_embeddings(len(tokenizer)) # 调整embedding matrix. Increasing the size will add newly initialized vectors at the end（为新加入的 token 随机初始化其对应的嵌入向量（通常使用均匀分布或正态分布）. Reducing the size will remove vectors from the end.


    # 2. Load the datasets
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

    # def to_unpaired_preference(example):
    #     data = {
    #         "question": example['question'],
    #         "answer": example['safe generation'],
    #     }
    #     return data

    # def create_datasets(tokenizer, args):

        # train_datasetaa = ConstantLengthDataset(
        #     tokenizer,
        #     train_dataset,
        #     formatting_func=prepare_sample_text,
        #     infinite=True,
        #     seq_length=args.seq_length,
        #     chars_per_token=chars_per_token,
        # )
        # valid_datasetaa = ConstantLengthDataset(
        #     tokenizer,
        #     valid_dataset,
        #     formatting_func=prepare_sample_text,
        #     infinite=False,
        #     seq_length=args.seq_length,
        #     chars_per_token=chars_per_token,
        # )

    # train_dataset, eval_dataset = create_datasets(tokenizer, script_args)
    
    train_dataset, valid_dataset, prepare_sample_text = load_dataset_for_train(script_args)
    if script_args.num_of_examples:
        train_dataset = train_dataset.select(range(100))
        print("只训练100条数据")


    # 4. initialize training arguments:
    peft_config = get_peft_config(script_args)
    neuron_training_config = get_neuron_training_config(script_args)
    other_methods_training_config = get_other_methods_training_config(script_args)

    # peft_config = LoraConfig(
    #     r=script_args.lora_r,
    #     lora_alpha=script_args.lora_alpha,
    #     lora_dropout=script_args.lora_dropout,
    #     target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # 5. initialize the SFT trainer
    if peft_config is not None:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            peft_config=peft_config,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
            dataset_train_batch_size=training_args.per_device_train_batch_size,
            dataset_eval_batch_size=training_args.per_device_eval_batch_size,
        )
    elif neuron_training_config is not None:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            neuron_training_config=neuron_training_config,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
            dataset_train_batch_size=training_args.per_device_train_batch_size,
            dataset_eval_batch_size=training_args.per_device_eval_batch_size,
        )
    elif other_methods_training_config is not None:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            other_methods_training_config=other_methods_training_config,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
            dataset_train_batch_size=training_args.per_device_train_batch_size,
            dataset_eval_batch_size=training_args.per_device_eval_batch_size,
        )
    else:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            max_seq_length=None,
            formatting_func=prepare_sample_text,
            processing_class=tokenizer,
            args=training_args,
            dataset_train_batch_size=training_args.per_device_train_batch_size,
            dataset_eval_batch_size=training_args.per_device_eval_batch_size,
        )

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    # if script_args.gradient_checkpointing:
    #     raise ValueError("gradient_checkpointing not supported")


    # 6. train
    trainer.train()

    base_model.eval()
    test(script_args, base_model, tokenizer, training_args.seed)
        
    # Random Neurons of the Last Layer Training Config
    if not script_args.use_random_training and not script_args.use_random_last_layer_training:
        trainer.save_model(training_args.output_dir)




    # 7. save
    # output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    # trainer.model.save_pretrained(output_dir)
    # print(f"The trained model has been saved at {output_dir}")

    # if peft_config is not None:
    #     # Free memory for merging weights
    #     del base_model
    #     if is_torch_xpu_available():
    #         torch.xpu.empty_cache()
    #     elif is_torch_npu_available():
    #         torch.npu.empty_cache()
    #     else:
    #         torch.cuda.empty_cache()

    #     model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto")
    #     model = model.merge_and_unload()

    #     output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    #     model.save_pretrained(output_merged_dir, safe_serialization=True)
    #     print(f"The merged model has been saved at {output_merged_dir}")
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

# 0. imports
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from peft import AutoPeftModelForCausalLM

from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import get_peft_config, get_neuron_training_config


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_id_or_path: Optional[str] = field(
        default="../saved_models/sft/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    model_name: Optional[str] = field(default="llama2-7b", metadata={"help": "the model name"})
    # data_dir: Optional[str] = field(
    #     default="/data/dans/projects/trl/trl/data/SafeEdit/train/",
    #     metadata={"help": "the location of the dataset"}
    # )
    # data_name: Optional[str] = field(default="safeedit", metadata={"help": "the dataset name"})
    adversarial_method_name: Optional[str] = field(default="SafeEditWithAdv", metadata={"help": "the dataset name"})

    run_name: Optional[str] = field(default="dpo_safeedit_llama2-7b", metadata={"help": "the experiment name"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})

    # LoraConfig
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})

    # Neuron Training Config
    use_neuron_training: Optional[bool] = field(default=False, metadata={"help": "whether to use neuron training"})
    trainable_param_num: Optional[int] = field(default=None, metadata={"help": "number of trainable neurons"})
    neuron_file: Optional[str] = field(default=None, metadata={"help": "The dir where the neuron files are stored"})

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    num_proc: Optional[int] = field(default=2, metadata={"help": "the number of processes"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "the eval strategy"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


def get_stack_exchange_paired(
    data_dir: str = "/data/dans/projects/trl/trl/data/SafeEdit/",
    split: str = None,
    num_proc = None,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "json",
        # data_files=f"{data_dir}/{split}.json",
        data_dir=data_dir,
        split=split,
        verification_mode="no_checks",
    )
    original_columns = dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [
                "Question: " + question + "\n\nAnswer: "
                for question in samples["question"]
            ],
            "chosen": samples["safe generation"],
            "rejected": [
                unsafe_generation.strip()
                for unsafe_generation in samples["unsafe generation"]
            ]
        }
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16


    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     script_args.model_id_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch_dtype,
    #     device_map={"": Accelerator().local_process_index},
    # )
    

    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(data_dir=script_args.data_dir, split="train")
    print(f"train_dataset: {train_dataset[0]}")
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
        num_proc=script_args.num_proc,
    )

    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_dir=script_args.data_dir, split="validation")
    print(f"validation_dataset: {eval_dataset[0]}")
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
        num_proc=script_args.num_proc,
    )

    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy=script_args.eval_strategy,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    # 4. initialize training arguments:
    peft_config = get_peft_config(script_args)
    neuron_training_config = get_neuron_training_config(script_args)
    # peft_config = LoraConfig(
    #     r=script_args.lora_r,
    #     lora_alpha=script_args.lora_alpha,
    #     lora_dropout=script_args.lora_dropout,
    #     # target_modules=[
    #     #     "q_proj",
    #     #     "v_proj",
    #     #     "k_proj",
    #     #     "out_proj",
    #     #     "fc_in",
    #     #     "fc_out",
    #     #     "wte",
    #     # ],
    #     target_modules=["q_proj","v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # 5. initialize the DPO trainer
    if peft_config is not None:
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )
    elif neuron_training_config is not None:
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            neuron_training_config=neuron_training_config,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )
    else:
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

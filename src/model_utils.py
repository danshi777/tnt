import json
import pandas as pd
from datasets import Dataset, load_dataset

model_name_to_path_dict = {
    # "llama2-7b-chat": "/data1/dans/models/llama2-7b-chat-hf",
    "llama2-7b-chat": "~/jinrenren_starfs/jinrenren/pretrained_models/meta-llama/Llama-2-7b-chat-hf",
    "gemma-2b-it": "/data/cordercorder/pretrained_models/google/gemma-2b-it",
}
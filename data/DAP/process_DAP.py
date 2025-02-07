import json
import os
import random

random.seed(42)

adv_method = "DAP"
os.makedirs(f"data/{adv_method}/train", exist_ok=True)
os.makedirs(f"data/{adv_method}/test", exist_ok=True)

template_file = f"data/{adv_method}/jailbreak_template.json"
with open(template_file, "r", encoding="utf-8") as f:
    json_data = f.read()
    loaded_data = json.loads(json_data)
jailbreak_templates = loaded_data["templates"]



for split in ["train/train", "train/validation"]:
    query_file = f"data/AdvBench/{split}.json"
    with open(query_file, "r", encoding="utf-8") as f:
        json_data = f.read()
        loaded_data = json.loads(json_data)
    malicious_queries = [example["question"] for example in loaded_data]

    processed_data = []
    for template in jailbreak_templates:
        for example in loaded_data:
            question = template.replace("[INSERT PROMPT HERE]", example["question"])
            processed_data.append({"original question": example["question"], "question": question, "safe generation": example["safe generation"]})

    random.shuffle(processed_data)

    data_to_output = []
    file_output_file = f"data/{adv_method}/{split}.json"
    for new_id, data in enumerate(processed_data):
        data_to_output.append(
        {   
            "id": new_id,
            "original question": data["original question"],
            "question": data["question"],
            "safe generation": data["safe generation"]
        }
        )
    with open(file_output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_output, f, ensure_ascii=False, indent=2)
    print(f"更新后的数据已保存到 {file_output_file}")



for split in ["test/test"]:
    query_file = f"data/AdvBench/{split}.json"
    with open(query_file, "r", encoding="utf-8") as f:
        json_data = f.read()
        loaded_data = json.loads(json_data)
    malicious_queries = [example["question"] for example in loaded_data]

    processed_data = []
    for template in jailbreak_templates:
        for example in loaded_data:
            question = template.replace("[INSERT PROMPT HERE]", example["question"])
            processed_data.append({"original question": example["question"], "question": question})

    random.shuffle(processed_data)

    data_to_output = []
    file_output_file = f"data/{adv_method}/{split}.json"
    for new_id, data in enumerate(processed_data):
        data_to_output.append(
        {   
            "id": new_id,
            "original question": data["original question"],
            "question": data["question"],
        }
        )
    with open(file_output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_output, f, ensure_ascii=False, indent=2)
    print(f"更新后的数据已保存到 {file_output_file}")
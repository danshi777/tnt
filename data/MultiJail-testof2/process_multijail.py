import pandas as pd
import json
import random
import os

random.seed(42)


file = f"data/MultiJail/MultiJail.csv"
df = pd.read_csv(file)
data = []
for idx, row in df.iterrows():
    data.append(
        {
            "id": row["id"],
            "source": row["source"],
            "tags": row["tags"],
            "en": row["en"],
            "zh": row["zh"],
            "it": row["it"],
            "vi": row["vi"],
            "ar": row["ar"],
            "ko": row["ko"],
            "th": row["th"],
            "bn": row["bn"],
            "sw": row["sw"],
            "jv": row["jv"]
        }
    )

random.shuffle(data)

total_size = len(data)
test_size = 50
val_size = 50
data_test = data[:val_size]
data_val = data[test_size:test_size + val_size]
data_train = data[test_size + val_size:]

data_path = "data/MultiJail/"
os.makedirs(f"{data_path}/train", exist_ok=True)
os.makedirs(f"{data_path}/test", exist_ok=True)

with open(f"{data_path}/test/test.json", "w", encoding="utf-8") as f_test:
    json.dump(data_test, f_test, ensure_ascii=False, indent=2)

with open(f"{data_path}/train/validation.json", "w", encoding="utf-8") as f_valid:
    json.dump(data_val, f_valid, ensure_ascii=False, indent=2)

with open(f"{data_path}/train/train.json", "w", encoding="utf-8") as f_train:
    json.dump(data_train, f_train, ensure_ascii=False, indent=2)
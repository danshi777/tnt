import pandas as pd
import random
import json

# 读取CSV文件
data = pd.read_csv("data/AdvBench/harmful_behaviors.csv")

# 设置随机种子以确保结果可复现
random_seed = 42
random.seed(random_seed)

# 添加原始ID列
data = data.reset_index().rename(columns={"index": "original id"})

# 随机打乱数据
data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# 计算各数据集的大小
total_size = len(data)
test_size = int(total_size * 0.2)
val_size = int(total_size * 0.2)
train_size = total_size - test_size - val_size

# 划分数据集
data_test = data[:test_size]
data_val = data[test_size:test_size + val_size]
data_train = data[test_size + val_size:]

# 保存到CSV文件
data_train.to_csv("data/AdvBench/harmful_behaviors_train.csv", index=False)
data_val.to_csv("data/AdvBench/harmful_behaviors_validation.csv", index=False)
data_test.to_csv("data/AdvBench/harmful_behaviors_test.csv", index=False)


# 将CSV文件转换为JSON文件
def csv_to_json(input_csv, output_json):
    df = pd.read_csv(input_csv)
    json_data = []
    for idx, row in df.iterrows():
        json_data.append({
            "id": idx,
            "original id": row["original id"],
            "question": row["goal"],
            # "unsafe generation": row["target"]
        })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

# 转换训练、验证和测试集
csv_to_json("data/AdvBench/harmful_behaviors_train.csv", "data/AdvBench/train/train_ori.json")
csv_to_json("data/AdvBench/harmful_behaviors_validation.csv", "data/AdvBench/train/validation_ori.json")
csv_to_json("data/AdvBench/harmful_behaviors_test.csv", "data/AdvBench/test/test.json")

print("数据集划分完成！")
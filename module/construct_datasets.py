import json
import pandas
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer

# 加载json数据集
def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []


# 对数据进行分词并转化为 token_id，最终得到输入、mask、输出对应的 token_id
def data_preprocess(item, tokenizer, max_length=1024):
    system_message = "You are a helpful assistant."
    user_message = item["instruction"] + item["input"]
    assistant_message = item["output"]

    instruction = tokenizer(f"<|im_start|>system\n{system_message}<|im_end|>\n"
                            f"<|im_start|>user\n{user_message}<|im_end|>\n"
                            f"<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(assistant_message, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = len(input_ids) * [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length]
    }

# 加载训练、验证、测试数据集
def load_dataset(data_path, tokenizer):
    data_list = load_json_data(data_path)
    data_set = Dataset.from_pandas(pandas.DataFrame(data_list))
    model_data = data_set.map(lambda x: data_preprocess(x, tokenizer), remove_columns=data_set.column_names)

    return model_data

# 主函数
def main():
    # 数据集路径
    train_data_path = "../dataset/ZHMedical/train_dataset.json"
    eval_data_path = "../dataset/ZHMedical/eval_dataset.json"
    test_data_path = "../dataset/ZHMedical/test_dataset.json"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("../models/Qwen2-7B-Instruct", trust_remote_code=True,local_files_only=True)


    # 加载数据集
    train_dataset = load_dataset(train_data_path, tokenizer)
    eval_dataset = load_dataset(eval_data_path, tokenizer)
    test_dataset = load_dataset(test_data_path, tokenizer)

    # 调试时只使用部分数据
    debug_mode = True  # 设置为 False 以使用完整数据集
    if debug_mode:
        train_dataset = train_dataset.select(range(64))  # 使用前 64 条数据
        eval_dataset = eval_dataset.select(range(64))
        test_dataset = test_dataset.select(range(64))

    # 打印数据集大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

if __name__ == "__main__":
    main()
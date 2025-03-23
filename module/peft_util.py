import json
import pandas
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score
#from vllm_code.vllm_predict import ChatLLM








# 使用测试数据集对模型进行不分批次的评测
def test_without_batch(model, tokenizer, test_path):
    test_dataset = load_json_data(test_path)
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')
    for item in test_dataset:
        prompt = item["instruction"] + item["input"]
        pred_label = predict(model, tokenizer, prompt)
        real_label = item["output"]
        pred_label = pred_label.replace("'", '"')
        real_label = real_label.replace("'", '"')
        real_list = json.loads(real_label)
        try:
            pred_list = json.loads(pred_label)
            if len(pred_list) < len(real_list):
                pred_list += ["O"] * (len(real_list) - len(pred_list))
            elif len(pred_list) > len(real_list):
                pred_list = pred_list[:len(real_list)]
            f1 = f1_score(real_list, pred_list, average='micro')
            f1_score_list.append(f1)
        except json.JSONDecodeError:
            pass
        pbar.update(1)
    pbar.close()
    f1_avg_score = sum(f1_score_list) / len(f1_score_list)
    print(f"f1-score: {f1_avg_score:.4f}")


# 使用测试数据集对模型进行批次的评测
def test_with_batch(model, tokenizer, test_path, batch_size: int = 8):
    test_dataset = load_json_data(test_path)
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')
    for i in range(0, len(test_dataset), batch_size):
        batch_data = test_dataset[i:i + batch_size]
        prompt_batch = [item["instruction"] + item["input"] for item in batch_data]
        pred_label = predict_batch(model, tokenizer, prompt_batch)
        real_label = [item["output"] for item in batch_data]
        for pred_label, real_label in zip(pred_label, real_label):
            pred_label = pred_label.replace("'", '"')
            real_label = real_label.replace("'", '"')
            real_list = json.loads(real_label)
            try:
                pred_list = json.loads(pred_label)
                if len(pred_list) < len(real_list):
                    pred_list += ["O"] * (len(real_list) - len(pred_list))
                elif len(pred_list) > len(real_list):
                    pred_list = pred_list[:len(real_list)]
                f1 = f1_score(real_list, pred_list, average='micro')
                f1_score_list.append(f1)
            except json.JSONDecodeError:
                pass
        pbar.update(len(batch_data))
    pbar.close()
    f1_avg_score = sum(f1_score_list) / len(f1_score_list)
    print(f"f1-score: {f1_avg_score:.4f}")


# 使用vllm大模型加速推理框架对测试数据集进行评测
#
# def test_with_vllm(model_path, test_path, batch_size: int = 8, quantization=None, kv_cache_dtype='auto'):
#     test_dataset = load_json_data(test_path)
#     f1_score_list = []
#     pbar = tqdm(total=len(test_dataset), desc=f'progress')
#     for i in range(0, len(test_dataset), batch_size):
#         batch_data = test_dataset[i:i + batch_size]
#         prompt_batch = [item["instruction"] + item["input"] for item in batch_data]
#         # 调用vllm框架进行推理
#         llm = ChatLLM(model_path, quantization=quantization, kv_cache_dtype=kv_cache_dtype)
#         pred_label = llm.infer(prompt_batch)
#         real_label = [item["output"] for item in batch_data]
#         for pred_label, real_label in zip(pred_label, real_label):
#             pred_label = pred_label.replace("'", '"')
#             real_label = real_label.replace("'", '"')
#             real_list = json.loads(real_label)
#             try:
#                 pred_list = json.loads(pred_label)
#                 if len(pred_list) < len(real_list):
#                     pred_list += ["O"] * (len(real_list) - len(pred_list))
#                 elif len(pred_list) > len(real_list):
#                     pred_list = pred_list[:len(real_list)]
#                 f1 = f1_score(real_list, pred_list, average='micro')
#                 f1_score_list.append(f1)
#             except json.JSONDecodeError:
#                 pass
#         pbar.update(len(batch_data))
#     pbar.close()
#     f1_avg_score = sum(f1_score_list) / len(f1_score_list)
#     print(f"f1-score: {f1_avg_score:.4f}")



if __name__ == '__main__':
    model_path = "../models/Qwen2-7B-Instruct"
    model, tokenizer = load_model(model_path, checkpoint_path='saved/medical_lora_model/checkpoint-1000')
    instruction = "请对以下文本进行命名实体识别，输出每个字符的BIO标注。B表示实体的开始，I表示实体的内部，O表示非实体部分。最后以列表的格式输出结果：\n"
    input = "['药', '进', '１', '０', '帖', '，', '黄', '疸', '稍', '退', '，', '饮', '食', '稍', '增', '，', '精', '神', '稍', '振']"
    prompt = instruction + input
    output = predict(model, tokenizer, prompt)
    print(output)

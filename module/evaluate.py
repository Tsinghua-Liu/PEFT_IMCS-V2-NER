import json
import pandas
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score
from .construct_datasets import load_json_data
import ast
import re

def safe_eval(pred):
    """
    安全地解析预测结果，修复截断的列表格式并处理常见语法错误。
    :param pred: 预测结果（字符串）
    :return: 解析后的列表
    """
    # 初始清理：移除多余的空格和换行符
    pred = pred.strip()  # 去除多余空格
    if not pred:
        return []

    # 如果以 '[' 开头但未以 ']' 结尾，认为是截断的
    if pred.startswith('[') and not pred.endswith(']'):
        # 找到最后一个完整的元素
        last_comma_index = pred.rfind(',')
        if last_comma_index != -1:
            pred = pred[:last_comma_index] + ']'  # 删除不完整的元素并补全 ']'
        else:
            pred = '[]'  # 如果没有完整元素，返回空列表

    try:
        result = ast.literal_eval(pred)
        # 确保返回的结果是列表
        if isinstance(result, tuple):
            return list(result)
        elif isinstance(result, list):
            return result
        else:
            return [result]  # 如果是单个元素，包装为列表
    except (SyntaxError, ValueError):
        return []  # 如果解析失败，返回空列表


def build_prompt(content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    return messages

# 利用模型进行批量预测生成答案
def predict_batch(model, tokenizer, contents, device='cuda'):
    prompts = [build_prompt(content) for content in contents]

    text = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": 128, "do_sample": True}
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            output = outputs[i, model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
        return responses

# 使用测试数据集对模型进行批次的评测
def test_with_batch(model, tokenizer, test_path, batch_size: int = 8 ,sample_radio = 1 ,print_number =1):
    """
    Args:
        model:
        tokenizer:
        test_path:
        batch_size:
        sample_radio: 对数据集中的多少比例进行测试
        print_number: 对测试数据中的多少数据进行打印显示

    Returns:

    """
    test_dataset = load_json_data(test_path)
    test_dataset = test_dataset[:int(len(test_dataset)*sample_radio)]
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')

    for i in range(0, len(test_dataset), batch_size):
        batch_data = test_dataset[i:i + batch_size]
        prompt_batch = [item["instruction"] + item["input"] for item in batch_data]
        pred_label = predict_batch(model, tokenizer, prompt_batch)
        real_label = [item["output"] for item in batch_data]  # str
        for pred_label, real_label in zip(pred_label, real_label):
            pred_label = pred_label.replace("'", '"')
            real_label = real_label.replace("'", '"')

            if print_number>0:
                print_number = print_number-1
                print(f"pred_label : \n{pred_label}")
                print(f"real_label : \n{real_label}")
            real_list = eval(real_label)
            try:
                pred_list = safe_eval(pred_label)  # 通过填充 处理预测和lable不一致的问题
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
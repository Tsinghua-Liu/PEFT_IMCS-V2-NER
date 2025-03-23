import json
import logging
import os
import random
from typing import List

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_line(line):
    """
    解析单行数据，返回字符和标签。
    :param line: 输入行
    :return: (char, label) 或 None（如果格式不正确）
    """
    parts = line.strip().split()
    if len(parts) == 2:
        return parts[0], parts[1]
    logging.warning(f"数据格式不正确，跳过行：'{line}'")
    return None

def save_dataset(dataset, output_file_path):
    """
    将数据集保存到文件。
    :param dataset: 数据集
    :param output_file_path: 输出文件路径
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)
        logging.info(f"处理完成，新的数据集已保存到 {output_file_path}")
    except IOError as e:
        logging.error(f"保存文件时出错：{e}")
def add_few_shot_examples(dataset, num_examples):
    """
    为数据集添加 Few-Shot 示例。
    :param dataset: 原始数据集
    :param num_examples: 示例数量
    :return: 包含 Few-Shot 示例的新数据集
    """
    if num_examples == 0:
        return dataset  # 如果没有示例，直接返回原始数据

    new_dataset = []
    for data in dataset:
        # 随机抽取 N 个示例
        examples = random.sample(dataset, min(num_examples, len(dataset)))
        # 构建 Few-Shot 上下文
        few_shot_context = "\n".join(
            [f"输入：{ex['input']}\n输出：{ex['output']}" for ex in examples]
        )
        # 将 Few-Shot 上下文添加到 instruction 中
        new_data = {
            "instruction": f"{data['instruction']}\n{few_shot_context}",
            "input": data["input"],
            "output": data["output"]
        }
        new_dataset.append(new_data)
    return new_dataset

def process_data(input_file_path, output_file_path, instruction=None, few_shots :List =[0]):
    """
    处理输入文件中的数据，生成新的数据集并保存到输出文件。
    :param input_file_path: 输入文件路径
    :param output_file_path: 输出文件路径
    :param instruction: 可选参数，自定义指令文本
    :return: None
    """
    # 默认指令
    default_instruction = (
        "请对以下文本进行命名实体识别，输出每个字符的BIO标注。"
        "B表示实体的开始，I表示实体的内部，O表示非实体部分。"
        "最后以列表的格式输出结果：\n"
    )
    instruction = instruction or default_instruction

    # 验证输入文件路径
    if not os.path.exists(input_file_path):
        logging.error(f"输入文件不存在：{input_file_path}")
        return

    # 读取输入文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except IOError as e:
        logging.error(f"读取文件时出错：{e}")
        return

    # 初始化变量
    dataset = []
    current_input = []
    current_output = []

    for line in lines:
        parsed = parse_line(line)
        if parsed:
            char, label = parsed
            current_input.append(char)
            current_output.append(label)
        elif current_input and current_output:  # 遇到空行或格式错误行时保存当前组数据
            dataset.append({
                "instruction": instruction,
                "input": "".join(current_input),
                "output": current_output
            })
            current_input = []
            current_output = []

    # 处理最后一组数据（如果没有空行结尾）
    if current_input and current_output:
        dataset.append({
            "instruction": instruction,
            "input": "".join(current_input),
            "output": current_output
        })

    # 生成 Few-Shot 数据集并保存
    for num_examples in [0]:
        few_shot_dataset = add_few_shot_examples(dataset, num_examples)
        # 生成输出文件名
        base_name, ext = os.path.splitext(output_file_path)
        few_shot_output_file = f"{base_name}_shot{num_examples}{ext}"
        # 保存数据集
        save_dataset(few_shot_dataset, few_shot_output_file)


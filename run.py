from module.preprocess_data import process_data
from module.construct_datasets import load_dataset
from module.model_manager import load_model,merge_model
from module.peft_load import build_peft_model
from module.peft_train import build_trainer,build_train_arguments,cross_validation_train
from module.evaluate import test_with_batch
import os.path
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

from sklearn.model_selection import KFold
import json
import os

debug_mode = False  # 设置为 False 以使用完整数据集

# 处理原始的训练数据，生成微调的训练数据集
input_train_path = "dataset/ZHMedical/medical_train.txt"
output_train_path = "dataset/ZHMedical/train_dataset.json"
process_data(input_train_path, output_train_path)
# 处理原始的评估数据，生成微调的评估数据集
input_eval_path = "dataset/ZHMedical/medical_eval.txt"
output_eval_path = "dataset/ZHMedical/eval_dataset.json"
process_data(input_eval_path, output_eval_path)
# 处理原始的测试数据，生成模型评测的测试数据集
input_tset_path = "dataset/ZHMedical/medical_test.txt"
output_test_path = "dataset/ZHMedical/test_dataset.json"
process_data(input_tset_path, output_test_path)

model_path = "models/Qwen2-0.5B-Instruct"
train_data_path = "dataset/ZHMedical/train_dataset.json"
eval_data_path = "dataset/ZHMedical/eval_dataset.json"
test_data_path = "dataset/ZHMedical/test_dataset.json"


import module.construct_datasets
import imp
imp.reload(module.construct_datasets)
from module.construct_datasets import load_dataset

lora_output_path = "saved/Qwen2-0.5B-Instruct_ZhMedNER"

# 加载模型和分词器
model, tokenizer = load_model(model_path)

# 加载训练、验证数据集
train_dataset = load_dataset(train_data_path, tokenizer)
eval_dataset = load_dataset(eval_data_path, tokenizer)
test_dataset = load_dataset(test_data_path, tokenizer)


if debug_mode:
    train_dataset = train_dataset.select(range(64))  # 使用前 64 条数据
    eval_dataset = eval_dataset.select(range(64))
    test_dataset = test_dataset.select(range(64))
print(train_dataset)
tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True)

for name, parameter in model.named_parameters():
    print(name)
    break

peft_model = build_peft_model(model)
print(peft_model.device)
peft_model.print_trainable_parameters()

lora_args = build_train_arguments(lora_output_path,eval_steps = 100,per_device_train_batch_size = 4)

# trainer = build_trainer(peft_model, tokenizer, lora_args, train_dataset, eval_dataset)
# trainer.train()

def cross_validation_train(peft_model, tokenizer, lora_args, train_data, eval_data, n_splits=5):
    """
    使用交叉验证进行训练。
    :param peft_model: PEFT 模型
    :param tokenizer: 分词器
    :param lora_args: 训练参数
    :param train_data: 训练数据
    :param eval_data: 验证数据
    :param n_splits: 交叉验证的折数
    """
    # 合并训练集和验证集
    full_data = concatenate_datasets([train_data, eval_data])
    results = []
    # 使用 KFold 进行交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_data)):
        print(f"Training fold {fold + 1}/{n_splits}...")

        # 划分训练集和验证集
        train_fold = full_data.select(train_idx)
        val_fold = full_data.select(val_idx)

        # 为当前 fold 生成唯一的 output_dir
        fold_output_dir = os.path.join(lora_args.output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        # 复制训练参数并更新 output_dir
        fold_lora_args = TrainingArguments(
            **{**lora_args.to_dict(), "output_dir": fold_output_dir}
        )
        """
        **lora_args.to_dict()：解包原始参数字典。
        "output_dir": fold_output_dir：添加或覆盖 output_dir 的值。
        ** new_args_dict 将字典解包为关键字参数，传递给 TrainingArguments 的构造函数。
        """
        # 构建 Trainer
        trainer = build_trainer(peft_model, tokenizer, lora_args, train_fold, val_fold)

        # 开始训练
        train_result = trainer.train()

        print(f"fold={fold}, result = {train_result}")
        results.append(train_result)

cross_validation_train(peft_model, tokenizer, lora_args, train_dataset, eval_dataset, n_splits=5)

# 进行测试。
test_with_batch(model, tokenizer, test_data_path, batch_size=16, sample_radio=1.0)
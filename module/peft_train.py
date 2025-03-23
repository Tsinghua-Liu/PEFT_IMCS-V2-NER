import os.path
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import KFold
import json
import os
from datasets import concatenate_datasets

def build_train_arguments(output_path, **kwargs):
    """
    构建 Lora 微调的超参数。
    :param output_path: 输出模型的保存目录
    :param kwargs: 自定义参数，用于覆盖默认值
    :return: TrainingArguments 对象
    """
    # 默认参数
    default_args = {
        "output_dir": output_path,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "log_level": "info",
        "logging_steps": 200,
        "logging_first_step": True,
        "logging_dir": os.path.join(output_path, "logs"),
        "num_train_epochs": 1,
        "eval_strategy": "steps",
        "eval_on_start": False,
        "eval_steps": 200,
        "save_steps": 200,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "save_on_each_node": True,
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "dataloader_drop_last": True,
        "gradient_checkpointing": True,
    }
    # 合并默认参数和用户传入的参数
    final_args = {**default_args, **kwargs}

    return TrainingArguments(**final_args)

def build_trainer(model, tokenizer, args, train_dataset, eval_dataset, **kwargs):
    """
    构建 Trainer 对象。
    :param args: TrainingArguments 对象
    :param train_dataset: 训练数据集
    :param eval_dataset: 验证数据集
    :param kwargs: 自定义参数，用于覆盖默认值
    :return: Trainer 对象
    """
    # 默认参数
    default_trainer_args = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=5)],
        "save_strategy" : "epoch",  # 按 epoch 保存检查点
        "save_total_limit" : 10,  # 最多保留 3 个检查点（按需调整）
    }

    # 合并默认参数和用户传入的参数
    final_trainer_args = {**default_trainer_args, **kwargs}

    # 如果启用梯度检查点，执行额外操作
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    return Trainer(**final_trainer_args)


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
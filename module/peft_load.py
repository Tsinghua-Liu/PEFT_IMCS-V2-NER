from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,  # 用于 QLoRA
    PrefixTuningConfig,               # 用于 P-Tuning
)

class FineTuningMethod(Enum):
    """
    微调方式的枚举类。
    """
    QLORA = "qlora"
    LORA = "lora"
    PTUNING = "ptuning"

def get_peft_config(method, **kwargs):
    """
    根据微调方式生成对应的 PEFT 配置。
    :param method: 微调方式（FineTuningMethod）
    :param kwargs: 自定义配置参数
    :return: PEFT 配置

    请设置 target_modules="all-linear"（比根据名称指定单个模块更容易，单个模块名称可能因架构而异）。
    """
    default_configs = {
        FineTuningMethod.QLORA: {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "modules_to_save": ["word_embeddings"],
            "inference_mode": False,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.2,
            "quantization_config": {"load_in_4bit": True,"bnb_4bit_quant_type":"nf4",
            "bnb_4bit_use_double_quant":True},  # QLoRA 需要量化配置
        },
        FineTuningMethod.LORA: {
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "modules_to_save": ["word_embeddings"],
            "inference_mode": False,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.2,
        },
        FineTuningMethod.PTUNING: {
            "task_type": TaskType.CAUSAL_LM,
            "num_virtual_tokens": 10,  # 虚拟 token 数量
            "encoder_hidden_size": 128,  # 编码器隐藏层大小
        },
    }

    # 合并默认配置和自定义配置
    config = default_configs.get(method, {})
    config.update(kwargs)

    # 根据微调方式返回对应的配置对象
    if method == FineTuningMethod.QLORA:
        return LoraConfig(**config)
    elif method == FineTuningMethod.LORA:
        return LoraConfig(**config)
    elif method == FineTuningMethod.PTUNING:
        return PrefixTuningConfig(**config)
    else:
        raise ValueError(f"不支持的微调方式：{method}")

def build_peft_model(model, method=FineTuningMethod.LORA, **kwargs):
    """
    根据微调方式构建 PEFT 模型。
    :param model: 原始模型
    :param method: 微调方式（FineTuningMethod）
    :param kwargs: 自定义配置参数
    :return: PEFT 模型
    """
    # 获取 PEFT 配置
    config = get_peft_config(method, **kwargs)

    # 如果是 QLoRA，需要先对模型进行量化准备
    if method == FineTuningMethod.QLORA:
        model = prepare_model_for_kbit_training(model)

    # 返回 PEFT 模型
    return get_peft_model(model, config)

# 示例用法
if __name__ == "__main__":
    # 加载模型和分词器
    model_path = "path_to_your_model"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 使用默认配置构建 LoRA 模型
    peft_model = build_peft_model(model, method=FineTuningMethod.LORA)

    # 使用自定义配置构建 QLoRA 模型
    peft_model = build_peft_model(
        model,
        method=FineTuningMethod.QLORA,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    # 使用默认配置构建 P-Tuning 模型
    peft_model = build_peft_model(model, method=FineTuningMethod.PTUNING)
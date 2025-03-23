import json
import pandas
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from peft import PeftModel



# 加载模型和分词器
def load_model(model_path, checkpoint_path='', device='cuda',is_quant = False):
    """
    Args:
        model_path:
        checkpoint_path:
        device:
        is_quant:  是否开启量化

    Returns:
    """
    if not is_quant:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(
            device)
        if checkpoint_path:
            model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)
            for param in model.base_model.parameters(): #对base model 部分梯度进行禁止
                param.requires_grad = False
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True  #对lora部分梯度打开
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    else:
        from transformers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=nf4_config, device_map={"":0}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, padding_side="left")
    return model, tokenizer

def merge_model(base_model_path, save_path, lora_path):

    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)
    base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {save_path}")
    model.save_pretrained(save_path)
    base_tokenizer.save_pretrained(save_path)



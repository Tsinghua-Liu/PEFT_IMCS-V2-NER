


# 利用模型进行单次预测生成答案
def predict(model, tokenizer, prompt, device='cuda'):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


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
    gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            output = outputs[i, model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
        return responses
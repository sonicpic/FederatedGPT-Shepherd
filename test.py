import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def inference(text: str, model_path: str, tokenizer_path: str, max_length: int = 50):
    """
    使用联邦学习和LoRA微调后的模型进行文本生成推理。

    :param text: 输入文本，模型将基于此文本生成续写。
    :param model_path: 保存的微调模型的路径。
    :param tokenizer_path: 分词器的路径，应与训练时相同。
    :param max_length: 生成文本的最大长度。
    :return: 模型生成的文本。
    """

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 准备输入数据
    inputs = tokenizer(text, return_tensors="pt")

    # 在不进行梯度更新的环境下执行推理
    with torch.no_grad():
        # 生成文本
        generated_ids = model.generate(inputs['input_ids'], max_length=max_length)

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

text = "你好，请介绍一下你自己。"
# model_path = './lora-shepherd/num_clients/adapter_model.bin'  #
model_path = 'meta-llama/Llama-2-7b-hf'  # 您保存模型的路径
tokenizer_path = 'meta-llama/Llama-2-7b-hf'  # 指定模型和分词器的路径

# 执行推理
generated_text = inference(text, model_path, tokenizer_path)
print(generated_text)
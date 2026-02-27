# import json
import os
# import shutil
# # from unsloth import FastLanguageModel
# from transformers import DataCollatorForLanguageModeling, TrainerCallback
import torch
# from datasets import load_dataset
# from unsloth.chat_templates import standardize_sharegpt
# import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 使用GPU 0: os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="/media/user01/date/Projects/LC/qwen3/sft/output/merged_cot-epoch3",
#     max_seq_length=4096,  # 控制上下文长度 虽然Qwen3支持40960，但建议测试时使用2048。
#     dtype=torch.float16,  # 启用4位量化，减少微调时内存使用量至原来的1/4，适用于16GB GPU
# )

tokenizer = AutoTokenizer.from_pretrained("/media/user01/date/Projects/LC/qwen3/sft/output/merged_cot-epoch3", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/media/user01/date/Projects/LC/qwen3/sft/output/merged_cot-epoch3",
    device_map="auto",  # 自动分配 GPU
    torch_dtype=torch.float16  # 或 bfloat16 / float32 取决于你的硬件
)
model.eval()

# messages = [
#     {"role" : "user", "content" : "请根据以下放射学检查所见，生成检查结论：门脉血管显示清楚，胆囊不大，内未见异常信号影。肝内、外胆管未见扩张。胰腺形态及信号未见异常，胰管未见扩张。脾脏形态信号未见异常。左肾见数个长T1长T2信号灶，界清无强化。肾上腺未见明显异常。肝门区及腹膜后未见肿大淋巴结。无腹水征。"}
# ]

messages = [
    {"role" : "user", "content" : "请将以下放射学检查所见改为结构化描述：门脉血管显示清楚，胆囊不大，内未见异常信号影。肝内、外胆管未见扩张。胰腺形态及信号未见异常，胰管未见扩张。脾脏形态信号未见异常。左肾见数个长T1长T2信号灶，界清无强化。肾上腺未见明显异常。肝门区及腹膜后未见肿大淋巴结。无腹水征。"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # Disable thinking
)

from transformers import TextStreamer
# _ = model.generate(
#     **tokenizer(text, return_tensors = "pt").to("cuda"), # .to("cuda"),
#     max_new_tokens = 4096, # Increase for longer outputs!
#     temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
#     streamer = TextStreamer(tokenizer, skip_prompt = True),
# )

with torch.no_grad():
    outputs = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"), # .to("cuda"),
        max_new_tokens = 4096, # Increase for longer outputs!
        temperature = 0.9,
        top_p = 0.9,
        do_sample=True,  # Enable sampling for more diverse outputs
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True, skip_generation_prompt=True)
print(response)
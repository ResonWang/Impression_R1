# 加载模型
import json
import os
import shutil

from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling, TrainerCallback
import torch
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/first_stage_merged_weights",  # model weights path of from the first stage of training
    max_seq_length=4096,
    dtype=torch.float16,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

with open("qwen3_cot_dataset.json", "r", encoding="utf-8") as f:
    non_reasoning_dataset = json.load(f)

non_reasoning_conversations = tokenizer.apply_chat_template(
    non_reasoning_dataset,
    tokenize=False,
)

non_reasoning_subset = pd.Series(non_reasoning_conversations)

data = pd.concat([
    pd.Series(non_reasoning_subset)
])
data.name = "text"

from datasets import Dataset

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
# 随机打乱数据集
combined_dataset = combined_dataset.shuffle(seed=3407).map(
    batched=True,
    num_proc=8,
)

class CustomSaveCallback(TrainerCallback):
    def __init__(self, output_dir, max_step_checkpoints=5):
        """
        自定义保存回调，同时支持：
        1. 每个epoch保存（不清除）
        2. 限制steps保存的数量，超过自动清除最早的
        """
        self.output_dir = output_dir
        self.max_step_checkpoints = max_step_checkpoints
        self.step_checkpoints = []  # 跟踪所有step保存的路径

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch_output_dir = os.path.join(self.output_dir, 'epoch', f"epoch-{state.epoch}")
        model.save_pretrained(epoch_output_dir)
        print(f"Epoch {state.epoch} 模型已保存至: {epoch_output_dir}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if args.save_strategy == "steps" and state.global_step % args.save_steps == 0:
            step_output_dir = os.path.join(self.output_dir, 'step', f"step-{state.global_step}")
            model.save_pretrained(step_output_dir)
            print(f"Step {state.global_step} 模型已保存至: {step_output_dir}")

            self.step_checkpoints.append((state.global_step, step_output_dir))

            if len(self.step_checkpoints) > self.max_step_checkpoints:
                self._clean_old_step_checkpoints()

    def _clean_old_step_checkpoints(self):
        self.step_checkpoints.sort(key=lambda x: x[0])
        old_checkpoints = self.step_checkpoints[:-self.max_step_checkpoints]

        for step, path in old_checkpoints:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"已清除旧的 step checkpoint: {path}")
                self.step_checkpoints.remove((step, path))

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    eval_dataset=None,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[CustomSaveCallback(
        output_dir="output/cot_sft",
        max_step_checkpoints=5
    )],
    args=SFTConfig(
        num_train_epochs=3,
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        save_steps=100,
        save_strategy='steps',
        save_total_limit=5,
    ),
)
trainer_stats = trainer.train(resume_from_checkpoint = False)

model.save_pretrained("cot_sft")
tokenizer.save_pretrained("cot_sft")


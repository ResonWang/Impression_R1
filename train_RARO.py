import json
import os
import shutil
from unsloth import FastLanguageModel
from transformers import TrainerCallback
import torch
from datasets import  Dataset
import re
from RaTEScore.scorer import RaTEScore
from trl import GRPOConfig, GRPOTrainer


def extract_xml_answer(text: str) -> str:
    match = re.search(r"检查结论：([\s\S]*)", text)
    if match:
        content = re.sub(r"<\|.*?\|>", "", match.group(1))
        content = content.strip()
        content = content.replace('\n', '')
        return content
    else:
        return ""

def safe_compute_score(ratescore, pred_report, gt_report, fallback_score=0.5):
    def is_list_of_str(x):
        if not isinstance(x, list):
            return False
        return all(isinstance(i, str) for i in x)

    if not is_list_of_str(pred_report) or not is_list_of_str(gt_report):
        length = max(len(pred_report) if isinstance(pred_report, list) else 0,
                     len(gt_report) if isinstance(gt_report, list) else 0, 1)
        return [fallback_score] * length

    try:
        return ratescore.compute_score(pred_report, gt_report)
    except Exception as e:
        return [fallback_score] * len(pred_report)

def RateScore_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    question = prompts[0][-1]["content"]
    pred_ch = [extract_xml_answer(completion[0]['content']) for completion in completions]
    gt_ch = answer
    ratescore = RaTEScore()
    try:
        scores = safe_compute_score(ratescore, pred_ch, gt_ch, fallback_score=0.5)
    except Exception as e:
        print(f"[Error] safe_compute_score failed: {e}")
        scores = [0.5] * len(completions)

    if len(scores) != len(completions):
        print(
            f"\n[Warning] Score count {len(scores)} != Completions count {len(completions)}. Padding/filling fallback scores.")
        scores = scores + [0.5] * (len(completions) - len(scores))
        scores = scores[:len(completions)]
    return [6 * (s - 0.5) for s in scores]

def load_my_medical_json(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
        for session in sessions:
            prompt = [{'role': 'user', 'content': session[0]['content'].replace('\n', '')}]
            answer_raw = session[1]['content'].strip().replace('\n', '')
            if answer_raw.startswith("检查结论："):
                answer = answer_raw[len("检查结论："):].strip()
            elif answer_raw.startswith("检查结论："):
                answer = answer_raw[len("检查结论："):].strip()
            else:
                answer = answer_raw
            data.append({
                'prompt': prompt,
                'answer': answer
            })
    return Dataset.from_list(data)

max_seq_length = 8192
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="",
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    fast_inference=True,
)

non_reasoning_dataset = load_my_medical_json('./grpo_data_1224.json')
from tqdm import tqdm

def calc_token_lengths(dataset, tokenizer):
    lengths = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        messages = sample['prompt'] + [{'role': 'assistant', 'content': sample['answer']}]
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )
        lengths.append(len(tokens))
    return lengths

lengths = calc_token_lengths(non_reasoning_dataset, tokenizer)
print(f"最大长度: {max(lengths)}，最小长度: {min(lengths)}，平均长度: {sum(lengths)/len(lengths):.2f}")
print(f"最长的样本索引: {lengths.index(max(lengths))}")

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

class CustomSaveCallback(TrainerCallback):
    def __init__(self, output_dir, max_step_checkpoints=5):
        self.output_dir = output_dir
        self.max_step_checkpoints = max_step_checkpoints
        self.step_checkpoints = []

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

max_prompt_length = 1500
training_args = GRPOConfig(
    seed=3407,
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    num_generations = 8,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 10,
    # max_steps = 250,
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir="",
    save_strategy='steps',
    save_total_limit=5,
    beta = 0.005,
    use_vllm=True,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        RateScore_reward_func,
    ],
    callbacks=[CustomSaveCallback(
        output_dir="",
        max_step_checkpoints=5
    )],
    args = training_args,
    train_dataset = non_reasoning_dataset,
)

trainer_stats = trainer.train(resume_from_checkpoint = True)

model.save_pretrained("grpo_saved_lora")
tokenizer.save_pretrained("grpo_saved_lora")


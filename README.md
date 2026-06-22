<p align="center">
  <img src="https://github.com/user-attachments/assets/42a6f6da-6761-4e2c-a1c0-f68ba2b4a6dc" width="100" alt="Impression-R1 logo" />
</p>

<h1 align="center">Impression-R1</h1>

<p align="center">
  A domain-specialized large reasoning model for radiology impression generation
</p>

## Overview

Impression-R1 generates a radiology impression from imaging findings. This repository provides:

- the API-based inference example used to demonstrate the input and output format;
- the complete three-stage training pipeline: supervised fine-tuning (SFT), chain-of-thought supervised fine-tuning (CoT-SFT), and RaTEScore-guided reinforcement learning optimization (RARO); and
- representative, deidentified data examples for all three training stages.

The representative files illustrate the expected schemas and are not the complete training datasets. Model weights and nonpublic clinical datasets are not distributed without restriction; see [Availability and governance](#availability-and-governance).

## Repository structure

```text
Impression_R1/
|-- Inference.py
|-- RaTEScore/
|-- Training/
|   |-- train_sft.py
|   |-- sft_64.json
|   |-- train_cot_sft.py
|   |-- cot_sft_64.json
|   |-- train_RARO.py
|   `-- grpo_32.json
`-- README.md
```

The training examples contain 64 SFT records, 64 CoT-SFT records, and 32 RARO records.

## Model access and inference

### Hosted demonstration

A web demonstration is available at [http://www.radiology-llm.com](http://www.radiology-llm.com).

### API client

`Inference.py` is an OpenAI-compatible API client. It does not load the model weights locally. Before running it, configure the `base_url`, `api_key`, and `model` fields for an authorized Impression-R1 endpoint.

Requirements:

- Python 3.12
- `openai`

```bash
pip install openai
python Inference.py
```

The evaluation configuration used for Impression-R1 was temperature `0.6`, top-p `0.95`, maximum output length `8192` tokens, and the default repetition penalty (`1.0`, not explicitly modified). Runtime depends on the endpoint and network connection.

To evaluate additional cases, replace the example findings in `Inference.py` or extend the client to read a structured input file. The script should be used only with deidentified data and an endpoint approved by the relevant institution.

## Three-stage training pipeline

The scripts in `Training/` reproduce the three training stages described in the manuscript.

| Stage | Purpose | Script | Representative data | Input checkpoint | Default saved adapter |
|---|---|---|---|---|---|
| 1. SFT | Learn radiology impression generation from findings | `Training/train_sft.py` | `Training/sft_64.json` | Base Qwen3-8B checkpoint | `sft/` |
| 2. CoT-SFT | Learn structured clinical reasoning before impression generation | `Training/train_cot_sft.py` | `Training/cot_sft_64.json` | Merged Stage 1 checkpoint | `cot_sft/` |
| 3. RARO | Optimize generation with the RaTEScore reward | `Training/train_RARO.py` | `Training/grpo_32.json` | Merged Stage 2 checkpoint | `grpo/` |

### Reported training environment

The original experiments were conducted with Ubuntu 22.04, Python 3.12, an NVIDIA RTX A6000 GPU with 48 GB memory, and 256 GB system RAM. Hardware requirements may vary with sequence length, batch size, quantization, and vLLM configuration.

Core dependencies include:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers datasets pandas trl vllm
```

The RARO stage also imports the local `RaTEScore/` implementation. Install any additional dependencies listed by that module before starting Stage 3. Package and CUDA versions should be selected for the local GPU driver and platform.

### Preparation

The released scripts retain explicit checkpoint and output paths so that each training transition is visible. Before execution:

1. Set the base model path in `Training/train_sft.py`.
2. Point `Training/train_cot_sft.py` to the merged Stage 1 checkpoint and to `cot_sft_64.json` (or a full dataset with the same schema).
3. Point `Training/train_RARO.py` to the merged Stage 2 checkpoint, configure writable output directories, and verify that the local RaTEScore import resolves correctly.
4. Adjust batch size, gradient accumulation, sequence length, and vLLM settings for the available hardware.

### Run the representative examples

Run each stage from the `Training/` directory after updating the paths described above:

```bash
cd Training
python train_sft.py
python train_cot_sft.py
python train_RARO.py
```

The default scripts train Stage 1 for 3 epochs, Stage 2 for 3 epochs, and Stage 3 for 10 epochs. Outputs from each stage must be merged or otherwise prepared as the input checkpoint expected by the next stage. The small representative datasets are intended for code inspection and pipeline validation, not for reproducing the reported model performance.

## Data availability

- The 200 difficult cases used in the human-AI competition are available [here](https://drive.google.com/drive/folders/1woQNddoaGuOLE48WXAo_eFWCV9ylutZr?usp=sharing).
- The three public evaluation datasets are available [here](https://drive.google.com/drive/folders/1Gsm1mQ54TyvwYDgZVnPTiVARW1lWpAvh?usp=sharing).
- Representative, deidentified examples for each training stage are included in `Training/`.
- Nonpublic clinical datasets are not publicly released because the original governance approvals and data-use agreements do not permit unrestricted redistribution.

## Availability and governance

The full model weights and complete nonpublic training data are not provided as unrestricted downloads. This restriction reflects patient privacy, institutional governance, potential memorization and data leakage, regulatory and compliance obligations, and intellectual-property considerations.

Qualified researchers may request controlled access for justified noncommercial research purposes. Requests are subject to institutional review, applicable data-use agreements, information-security requirements, and any necessary ethics or legal approvals. Access is not guaranteed. Please use the contact information provided in the associated manuscript or repository to submit a request describing the research purpose, institution, requested artifacts, and proposed safeguards.

## References

- [Unsloth reinforcement learning guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#training-with-grpo)
- [Unsloth Qwen3 GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)

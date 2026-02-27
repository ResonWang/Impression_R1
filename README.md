<p align="center">
  <img src="https://github.com/user-attachments/assets/42a6f6da-6761-4e2c-a1c0-f68ba2b4a6dc" width="100" alt="Impression-R1 logo" />
</p>

<h1 align="center">Impression-R1</h1>

<p align="center">
  Domain-specialized large reasoning model for radiological impression generation
</p>

---
## Model inference
### 1. System Requirements

#### Operating System
- Tested on: Windows 10

#### Software Dependencies
- Python 3.12

Install dependency:

```bash
pip install openai
```
#### Non-standard Hardware
None required (API-based inference).

### 2. Installation Guide

Install Python 3.12

Install dependency

Typical installation time on a normal desktop computer:
Less than 5 minutes.

### 3. Demo
#### Run demo
```bash
python inference.py
```
#### Example
**Input findings**： "Strip- and patch-like hyperdense shadows are seen along the falx cerebri and tentorium cerebelli. No abnormal changes in structure or morphology are observed in the bilateral cerebral hemispheres, cerebellum, or brainstem. No abnormal density is detected within the brain parenchyma; the gray-white matter differentiation is clear. The ventricular system shows no significant dilatation, and the cerebral sulci, cisterns, and fissures appear normal. No midline shift is noted. The cranial bones show no abnormal changes. The mucosa of the bilateral paranasal sinuses shows no thickening."

Output:

#### Expected Runtime
Usuallly 10–20 seconds per case currently on a normal desktop computer (network dependent).

### 4. Instructions for Use
To run on your own data, modify the input inside `inference.py` or extend the script to load multiple cases.

## Model training
`Train_RARO.py` is the RARO RL training code.

### 1. System Requirements

#### Operating System
- Tested on: Ubuntu 22.04

#### Software Dependencies
- Python 3.12

Install dependency:

```bash
pip install unsloth
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
(pip install torch torchvision torchaudio \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121)
pip install vllm
```
#### Reference
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb

https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#training-with-grpo

## Dataset
The 200 difficult cases used for the human–AI competition can be found in: https://drive.google.com/drive/folders/1woQNddoaGuOLE48WXAo_eFWCV9ylutZr?usp=sharing

The three public datasets can be found in: https://drive.google.com/drive/folders/1Gsm1mQ54TyvwYDgZVnPTiVARW1lWpAvh?usp=sharing

The private datasets are not available due to privacy issues. 


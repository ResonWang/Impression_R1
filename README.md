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
**Input findings：**  "The bilateral breasts are generally symmetrical, with a balanced distribution of glandular tissue. Both breasts show mild background parenchymal enhancement (BPE). In the outer lower quadrant of the left breast, at approximately the 4–5 o’clock position, there is a mass measuring about 23 mm × 15 mm. It is oval in shape with indistinct margins, showing iso-signal intensity on T1WI and T2WI, with areas of long T1 and short T2 signal inside. The lesion demonstrates high signal on DWI and a corresponding high signal on the ADC map. Contrast-enhanced scan shows mild-to-moderate heterogeneous enhancement, and the dynamic enhancement time–signal intensity curve (TIC) is of a persistent (type I) pattern. In the outer upper quadrant of the right breast, at approximately the 11 o’clock position, a small oval mass is seen, measuring about 9 mm × 5 mm, with well-defined margins. It shows mixed iso- to slightly high signal on T2WI and slightly low signal on T1WI. On DWI, it appears slightly hyperintense, with mildly decreased signal on the ADC map. The lesion demonstrates early marked homogeneous enhancement, and the TIC is of a persistent (type I) pattern. Scattered punctate enhancements (<5 mm in diameter) are seen in both breasts. No obvious abnormalities are noted in the skin of either breast, and no nipple retraction is observed. No enlarged lymph nodes are seen in either axilla."


**Output:** "The impression of this report is:
1.Left breast mass, BI-RADS category 3–4A, likely fibroadenoma. Tissue biopsy is recommended for further evaluation.
2.Right breast mass, BI-RADS category 3, likely fibroadenoma. Six-month imaging follow-up is recommended.
3.Scattered punctate enhancements in both breasts, BI-RADS category 3. Six-month follow-up is recommended.
4.Bilateral fibrocystic changes."

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

https://github.com/MAGIC-AI4Med/RaTEScore

## Dataset
The 200 difficult cases used for the human–AI competition can be found in: https://drive.google.com/drive/folders/1woQNddoaGuOLE48WXAo_eFWCV9ylutZr?usp=sharing

The three public datasets can be found in: https://drive.google.com/drive/folders/1Gsm1mQ54TyvwYDgZVnPTiVARW1lWpAvh?usp=sharing

The private datasets are not available due to privacy issues. 


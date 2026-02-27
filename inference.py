# -*- coding: utf-8 -*-
from openai import OpenAI

client = OpenAI(
    base_url="http://llm_api.vip.cpolar.cn/v1",
    api_key="EMPTY"
)

System_prompt_zh = "请根据以下放射学检查所见，生成检查结论。"
System_prompt_en = "Please generate the impressions according to the following imaging findings:", # "Please generate an impression based on the following radiological findings."

Findings_zh = "大脑镰及小脑幕见条片状高密度影；双侧大脑半球、小脑及脑干结构、形态未见异常改变，脑实质未见异常密度影，灰白质界限清楚，脑室系统未见明显扩张，脑沟、脑池及脑裂均未见异常。中线结构无移位。颅板骨质未见异常改变。双侧副鼻窦粘膜无增厚。"
Findings_en = "The bilateral breasts are generally symmetrical, with a balanced distribution of glandular tissue. Both breasts show mild background parenchymal enhancement (BPE). In the outer lower quadrant of the left breast, at approximately the 4–5 o’clock position, there is a mass measuring about 23 mm × 15 mm. It is oval in shape with indistinct margins, showing iso-signal intensity on T1WI and T2WI, with areas of long T1 and short T2 signal inside. The lesion demonstrates high signal on DWI and a corresponding high signal on the ADC map. Contrast-enhanced scan shows mild-to-moderate heterogeneous enhancement, and the dynamic enhancement time–signal intensity curve (TIC) is of a persistent (type I) pattern. In the outer upper quadrant of the right breast, at approximately the 11 o’clock position, a small oval mass is seen, measuring about 9 mm × 5 mm, with well-defined margins. It shows mixed iso- to slightly high signal on T2WI and slightly low signal on T1WI. On DWI, it appears slightly hyperintense, with mildly decreased signal on the ADC map. The lesion demonstrates early marked homogeneous enhancement, and the TIC is of a persistent (type I) pattern. Scattered punctate enhancements (<5 mm in diameter) are seen in both breasts. No obvious abnormalities are noted in the skin of either breast, and no nipple retraction is observed. No enlarged lymph nodes are seen in either axilla."

response = client.chat.completions.create(
    model="Impression-R1-grpo2820",
    messages=[
        {"role": "user", "content": f'{System_prompt_en} {Findings_en}'}
    ],
    temperature=0.6,
    top_p=0.95,
)

print(response.choices[0].message.content)

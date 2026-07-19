import requests, json

OLLAMA_API_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'qwen2.5:7b'

def translate_batch_ollama(texts):
    numbered_texts = '\n'.join([f'{i+1}. {text}' for i, text in enumerate(texts)])
    system_prompt = """你是一个专业的视频字幕翻译员。
请将以下带有序号的外文字幕翻译成简体中文。
要求：
1. 必须保留原有的序号格式（例如 "1. 翻译内容"）。
2. 不要合并或删除任何行，必须一一对应。
3. 无论原句多么简短或难以理解，都必须将其翻译为中文（如语气词可译为“啊”、“嗯”等），绝对不能直接输出外文原文。
4. 不要输出任何额外的解释或废话。"""
    user_prompt = f"请翻译以下字幕：\n{numbered_texts}"
    payload = {
        'model': OLLAMA_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'stream': False,
        'options': {'temperature': 0.3}
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
    result_text = response.json()['message']['content']
    return result_text

texts = [
    'りょお',
    '教えてくれよ',
    '私は大丈夫',
    '見ないで',
    'こんなの全然平気だから'
]
res = translate_batch_ollama(texts)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(res)
print("Saved to output.txt")

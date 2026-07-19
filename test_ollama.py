import requests, json, re

OLLAMA_API_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'qwen2.5:7b'

def translate_batch_ollama(texts):
    numbered_texts = '\n'.join([f'{i+1}. {text}' for i, text in enumerate(texts)])
    system_prompt = """你是一个专业的视频字幕翻译员。
请将以下带有序号的外文字幕翻译成简体中文。
要求：
1. 必须保留原有的序号格式（例如 "1. 翻译内容"）。
2. 不要合并或删除任何行。
3. 不要输出任何额外的解释或废话。"""
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
    'リコに勉強を教えてくれてるんでしょ?',
    'まあ一応',
    '教えるの上手いの?',
    'そりゃそういえばいい大学してるんだけど',
    '確かに',
    '勝手に教授のバイトもやってるんだよね'
]
res = translate_batch_ollama(texts)
print('--- MODEL OUTPUT ---')
print(res)
print('--------------------')

# Now run the actual parsing logic to see what it produces:
translated_lines = []
for line in res.split('\n'):
    line = line.strip()
    if line:
        match = re.match(r'^\d+[\.、]\s*(.*)$', line)
        if match:
            translated_lines.append(match.group(1).strip())
        else:
            translated_lines.append(line)

print("Parsed lines:")
for i, line in enumerate(translated_lines):
    print(f"{i}: {line}")

import os
import re
import json
import requests
import site
from tqdm import tqdm
from faster_whisper import WhisperModel
from moviepy import VideoFileClip

# ==========================================
# 配置区域
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"  # 推荐使用 qwen 系列或 llama3，需先在本地 ollama run qwen2.5:7b
WHISPER_MODEL_SIZE = "large-v3"   # base, small, medium, large-v3

# [Windows 专属] 尝试自动加载通过 pip 安装的 CUDA DLL 路径
if os.name == 'nt':
    try:
        # 获取 site-packages 目录，如果是 user 安装则可能在 getusersitepackages
        site_packages = site.getsitepackages() + [site.getusersitepackages()]
        for sp in site_packages:
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            if os.path.exists(cublas_bin):
                os.environ["PATH"] = cublas_bin + os.pathsep + os.environ["PATH"]
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(cublas_bin)
            if os.path.exists(cudnn_bin):
                os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ["PATH"]
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(cudnn_bin)
    except Exception:
        pass

def extract_audio(video_path, audio_path):
    """1. 从视频中提取音频"""
    if os.path.exists(audio_path):
        print(f"[*] 检测到已存在的音频文件 {audio_path}，跳过提取。")
        return
        
    print(f"[*] 正在从 {video_path} 提取音频...")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        print("[+] 音频提取成功！")
    except Exception as e:
        print(f"[-] 提取音频失败: {e}")
        raise e

def transcribe_audio(audio_path, cache_file):
    """2. 使用 faster-whisper 进行语音识别提取带有时间戳的文本"""
    print("\n" + "="*50)
    print("=== 阶段 1/2: 语音识别提取 (Transcription) ===")
    print("="*50)
    if os.path.exists(cache_file):
        print(f"[*] 检测到识别缓存 {cache_file}，直接加载已识别时间戳...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print(f"[*] 正在加载 faster-whisper 模型 ({WHISPER_MODEL_SIZE})...")
    
    try:
        # 尝试使用自动检测（优先GPU）
        model = WhisperModel(WHISPER_MODEL_SIZE, device="auto", compute_type="default")
        print("[*] 正在识别音频...")
        segments, info = model.transcribe(audio_path, beam_size=5)
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "cudart" in str(e).lower():
            print("\n[!] 警告: 未检测到完整的 CUDA 驱动依赖 (缺失 cublas/cudnn dll)。")
            print("[!] 正在自动切换到 CPU 模式进行识别 (速度会变慢)...")
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("[*] 正在识别音频...")
            segments, info = model.transcribe(audio_path, beam_size=5)
        else:
            raise e
            
    print(f"[+] 检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"[*] 音频总长度: {info.duration:.2f} 秒")
    
    results = []
    with tqdm(total=round(info.duration, 2), desc="语音识别进度", unit="秒") as pbar:
        last_end = 0.0
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            pbar.update(round(segment.end - last_end, 2))
            last_end = segment.end
            
    # 保存缓存
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] 识别结果已缓存至 {cache_file}")
            
    return results

def translate_batch_ollama(texts):
    """调用本地 Ollama 进行批量翻译"""
    if not texts:
        return []

    # 构造带序号的提示词
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    
    system_prompt = """你是一个专业的视频字幕翻译员。
请将以下带有序号的外文字幕翻译成简体中文。
要求：
1. 必须保留原有的序号格式（例如 "1. 翻译内容"）。
2. 不要合并或删除任何行。
3. 不要输出任何额外的解释或废话。"""

    user_prompt = f"请翻译以下字幕：\n{numbered_texts}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3 # 低温保证输出格式稳定
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json()["message"]["content"]
        
        # 解析返回的文本，提取翻译结果
        translated_lines = []
        for line in result_text.split('\n'):
            line = line.strip()
            if line:
                # 使用正则匹配 "1. xxxxx" 或者 "1、xxxxx"
                match = re.match(r'^\d+[\.、]\s*(.*)$', line)
                if match:
                    translated_lines.append(match.group(1).strip())
                else:
                    # 如果模型没有严格按照序号返回，我们尽量保留整行
                    translated_lines.append(line)
        
        # 如果解析的数量和原文数量不一致，做简单容错补齐
        while len(translated_lines) < len(texts):
            translated_lines.append("[翻译缺失]")
            
        return translated_lines[:len(texts)] # 保证长度一致

    except Exception as e:
        print(f"[-] 请求本地大模型失败: {e}")
        print("请确认 Ollama 已经启动，并且安装了相应的模型！")
        return ["[翻译失败]"] * len(texts)

def translate_all_segments(segments, cache_file, batch_size=10):
    """3. 批量循环翻译所有片段 (带断点续传与句子级去重)"""
    print("\n" + "="*50)
    print("=== 阶段 2/2: 大模型本地翻译 (Translation) ===")
    print("="*50)
    translated_segments = []
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            translated_segments = json.load(f)
        print(f"[*] 检测到翻译缓存 {cache_file}，已恢复 {len(translated_segments)} 条翻译进度。")

    remaining_segments = segments[len(translated_segments):]
    if not remaining_segments:
        print("[*] 所有片段均已翻译完毕！")
        return translated_segments
        
    print(f"[*] 准备翻译，剩余 {len(remaining_segments)} 条字幕，每批次提取 {batch_size} 条...")
    
    # 构建历史翻译字典（用于去重重复句子）
    sentence_cache = {}
    for seg in translated_segments:
        if "text" in seg and "zh_text" in seg:
            sentence_cache[seg["text"]] = seg["zh_text"]
    
    total_batches = (len(remaining_segments) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="翻译进度", unit="批次") as pbar:
        for i in range(0, len(remaining_segments), batch_size):
            batch = remaining_segments[i:i+batch_size]
            
            # 过滤出没翻译过的且不重复的句子
            texts_to_translate = []
            for item in batch:
                txt = item["text"]
                if txt not in sentence_cache and txt not in texts_to_translate:
                    texts_to_translate.append(txt)
            
            # 调用大模型翻译未知句子
            if texts_to_translate:
                translated_texts = translate_batch_ollama(texts_to_translate)
                # 将新翻译结果存入字典
                for orig, trans in zip(texts_to_translate, translated_texts):
                    sentence_cache[orig] = trans
            
            # 组装完整的翻译后片段
            for item in batch:
                new_item = item.copy()
                new_item["zh_text"] = sentence_cache.get(item["text"], item["text"]) # 若翻译失败退回原文
                translated_segments.append(new_item)
                
            # 每处理完一个批次就保存一次，确保进度不丢失
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(translated_segments, f, ensure_ascii=False, indent=2)
                
            pbar.update(1)
            
    return translated_segments

def format_timestamp(seconds):
    """将秒转换为 SRT 时间戳格式 00:00:00,000"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    mills = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{mills:03d}"

def generate_srt(segments, output_srt):
    """4. 生成 SRT 文件"""
    print(f"[*] 正在生成 SRT 字幕文件: {output_srt}...")
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            zh_text = segment.get("zh_text", segment["text"])
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            # 这里同时输出中文和原文（双语字幕），您也可以将其修改为仅输出中文
            f.write(f"{zh_text}\n")
            f.write(f"{segment['text']}\n\n")
    print("[+] 生成完毕！")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="本地视频语音翻译程序 (基于 Faster-Whisper 和 Ollama)")
    parser.add_argument("video", help="输入的视频文件路径 (如: video.mp4)")
    parser.add_argument("--audio", default=None, help="中间音频文件存储路径 (默认存放在视频同级目录)")
    parser.add_argument("--output", default=None, help="输出的 srt 字幕路径 (默认与视频同名并存放在视频同级目录)")
    args = parser.parse_args()

    video_file = args.video
    
    # 派生同名文件路径
    base_name = os.path.splitext(video_file)[0]
    audio_file = args.audio if args.audio else f"{base_name}.wav"
    srt_file = args.output if args.output else f"{base_name}.srt"
    
    # 派生缓存文件名 (放在视频同目录下)
    transcription_cache = f"{base_name}.transcription.json"
    translation_cache = f"{base_name}.translation.json"

    if not os.path.exists(video_file):
        print(f"[-] 找不到视频文件: {video_file}")
        return

    success = False
    try:
        # 1. 提取音频 (带保留机制)
        extract_audio(video_file, audio_file)
        
        # 2. 语音识别 (带缓存)
        segments = transcribe_audio(audio_file, transcription_cache)
        
        # 3. 本地大语言模型翻译 (带断点续传)
        segments = translate_all_segments(segments, translation_cache, batch_size=10)
        
        # 4. 写入 SRT
        generate_srt(segments, srt_file)
        
        success = True
        print("\n[✔] 整个工作流执行成功！")

    except KeyboardInterrupt:
        print("\n[!] 用户手动中止了程序。")
    except Exception as e:
        print(f"\n[-] 发生错误: {e}")
    finally:
        # 清理临时文件
        if success:
            print("[*] 正在清理临时文件和进度缓存...")
            for temp_file in [audio_file, transcription_cache, translation_cache]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"    - 已清理: {temp_file}")
        else:
            print("\n[!] 提示: 由于执行未完全结束，临时音频和进度缓存已被保留。下次重新运行将自动恢复进度。")

if __name__ == "__main__":
    main()

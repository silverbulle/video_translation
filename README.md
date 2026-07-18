# Video Translation to Subtitles (本地视频翻译与双语字幕生成)

这是一个基于 Python 的本地化视频翻译程序。它可以通过提取视频中的音频，利用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 在本地进行极速离线语音识别，然后通过调用本地部署的 [Ollama](https://ollama.com/) 大语言模型（如 `qwen2.5` 或 `llama3`）进行高质量的批量上下文翻译，最终输出 `.srt` 格式的中英双语字幕文件。

整个流程**完全离线、免费，且能够保护数据隐私**。

## ✨ 功能特性

* **提取音频**：自动从常见视频格式（如 MP4）中无损分离音频轨道。
* **高精度时间戳提取**：基于 OpenAI Whisper 的 C++ 优化版 `faster-whisper`，拥有数倍于原版的推理速度，并完美支持 CPU 回退和自动加载 Windows CUDA DLL 机制。
* **本地大模型上下文翻译**：通过对接 Ollama 本地 API，按批次对字幕进行分组翻译，避免“逐句翻译”导致的上下文割裂和机翻感。支持自动正则格式对齐。
* **全流程进度展示**：内置 `tqdm` 进度条，终端运行状态一目了然。

## 🚀 快速开始

### 1. 安装基础环境

请确保您的电脑已安装 Python (推荐 3.10 以上版本)。克隆或下载本项目后，进入项目目录，执行以下命令安装必要的 Python 第三方库：

```bash
pip install -r requirements.txt
```

#### 🔌 (可选) Windows 下的 GPU 加速库配置
如果您使用的是带有 NVIDIA 显卡的 Windows 电脑，为了让 `faster-whisper` 能够跑满 GPU 算力，强烈建议您安装相关的 CUDA 支持库：
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
*(注意：代码中已内置自动寻址加载这些包的逻辑，您只需安装即可，无需手动配置环境变量)*

---

### 2. 配置本地大模型 (Ollama)

本程序的翻译模块依赖于 Ollama 本地服务。

1. **下载安装**：前往 [Ollama 官网](https://ollama.com/) 下载并安装对应系统的客户端。
2. **下载翻译模型**：打开新的命令行终端，输入以下命令拉取我们推荐的中英文翻译模型 `qwen2.5:7b`（体积约 4GB）：
   ```bash
   ollama run qwen2.5:7b
   ```
   *(下载完成后您可以输入 `/bye` 退出对话模式，只要 Ollama 在后台运行即可)*

---

### 3. 运行程序

准备一个测试视频文件（例如 `test_video.mp4`），然后在项目目录下运行：

```bash
python video_translator.py test_video.mp4
```

您也可以通过可选参数指定生成的字幕名称或临时音频名称：

```bash
python video_translator.py test_video.mp4 --output "双语字幕.srt"
```

运行结束后，您将在同目录下获得一个完美对齐好时间轴的 `.srt` 格式双语字幕文件！

## ⚙️ 进阶配置
您可以直接编辑 `video_translator.py` 头部的配置区域：
* `OLLAMA_MODEL`: 更改所使用的大模型名称（默认为 `qwen2.5:7b`）。
* `WHISPER_MODEL_SIZE`: 更改语音识别模型的规模（默认为 `small`，显存大于 8G 可尝试修改为 `large-v3` 获得极高精准度）。

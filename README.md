# PyVoiceTranslate: WeNet + NLLB 语音识别与翻译

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/Higasa-Yumetaka/PyVoiceTranslate)

## 简介

本项目是一个使用WeNet和NLLB的端到端语音识别与翻译的基于PyQt5的GUI应用程序。该应用程序支持语音识别和翻译功能，用户可以通过麦克风输入语音，或者通过文件导入语音，实现语音识别和翻译功能。

## 安装

### 1. 安装WeNet

```bash
git clone https://github.com/wenet-e2e/wenet.git
```

若要将WeNet作为库或命令行工具使用，可以使用以下命令安装WeNet：

```bash
cd wenet
pip install -e .
```

将WeNet作为库使用时，将其中的wenet目录拷贝到项目中即可。

### 2. 安装依赖

#### 2.1 安装PyQt5

```bash
pip install PyQt5
```

#### 2.2 安装PyTorch

请根据自己的环境安装对应的PyTorch版本，详见[PyTorch Start Locally](https://pytorch.org/get-started/locally/)。

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2.3 安装其他依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型

WeNet提供了多种模型，包括中文、英文、法文、西班牙文等多种语言的语音识别和翻译模型。用户可以根据自己的需求选择合适的模型。模型下载地址：[WeNet Models](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md)
或者使用以下命令下载wenetspeech模型：

```python
import wenet
model = wenet.load_model('chinese')
```

即可在目录.wenet/chinese中找到模型文件。需要下载英文模型时，将chinese替换为english即可。
NLLB模型支持多种语言的翻译，用只需要下载一个模型即可。模型下载地址：[NLLB Models](https://huggingface.co/facebook/nllb-200-distilled-600M)
也可以使用以下命令下载nllb模型：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
```

即可在目录.cache/huggingface/facebook/nllb-200-distilled-600M中找到模型文件。

### 4. 修改配置文件

在config.json中修改模型路径，可将模型放入项目文件中，也可以使用绝对路径。
默认项目目录组织如下：

```
.
├── language_models
│   ├── chinese
│   │   ├── final.zip
│   │   ├── ...
│   ├── english
│   │   ├── final.zip
│   │   ├── ...
├── log
│   ├── ...
├── nllb-200-distilled-600M
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── ...
├── wenet
|   ├── ...
├── config.yaml
├── app.py
```

### 5. 运行应用程序

```bash
python app.py
```

## 致谢

本项目使用了WeNet和NLLB的模型，感谢WeNet和NLLB的开发者们为我们提供开发平台和模型。
本项目使用了以下开源项目：

- [WeNet](https://github.com/wenet-e2e/wenet) - 一个端到端语音识别和语音翻译的开源工具包。[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/wenet-e2e/wenet/blob/main/LICENSE)
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Facebook AI Research的NLLB模型。[![License](https://img.shields.io/badge/License-MIT%202.0-brightgreen.svg)](https://github.com/facebookresearch/fairseq/blob/nllb/LICENSE)

## 协议

### 本项目基于[Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)协议进行分发
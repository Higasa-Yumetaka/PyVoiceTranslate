import os
import sys
import wave
import yaml

import soundfile
import torch
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from loguru import logger as log
from pyaudio import PyAudio, paInt16
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import wenet


def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


config = load_config()

# 日志配置
log_level = config['log']['log_level']
log_dir = config['log']['log_dir']
log_stdout = config['log']['log_stdout']
log_file = config['log']['log_file']
log_std_level = config['log']['log_std_level']

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log.remove()
if log_stdout:
    log.add(sys.stdout, level=log_std_level)
if log_file:
    log.add(log_dir + "/{time:YYYY-MM-DD}.log",
            level=log_level,
            rotation='1 day',
            retention='7 days')

log.debug("Starting application")

log.debug(f"Loading config file from {os.path.abspath('config.yaml')}")


def getLanguageCode(text):
    for lang in config['languages']:
        if lang['name'] == text:
            return lang['code'], lang['model_dir']
    return None, None


# 录音线程：录音、保存录音文件
class RecorderThread(QtCore.QThread):
    recordingStarted = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.isRecording = False
        self.audioBuffer = []

    def run(self):
        framerate = 16000
        num_samples = 2000
        channels = 1

        pa = PyAudio()
        stream = pa.open(format=paInt16,
                         channels=channels,
                         rate=framerate,
                         input=True,
                         frames_per_buffer=num_samples)
        self.audioBuffer = []
        log.debug('正在录音...')

        self.recordingStarted.emit()

        while self.isRecording:
            string_audio_data = stream.read(num_samples)
            self.audioBuffer.append(string_audio_data)

        stream.close()
        pa.terminate()
        log.debug('录音结束.')

    def startRecording(self):
        self.isRecording = True
        self.start()

    def stopRecording(self):
        self.isRecording = False
        self.wait()

    def saveRecording(self, filepath):
        frameRate = 16000
        channels = 1
        sampWidth = 2

        try:
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(sampWidth)
            wf.setframerate(frameRate)
            wf.writeframes(b''.join(self.audioBuffer))
            wf.close()
            log.info(f'录音文件已保存为 {filepath}')
        except Exception as e:
            log.error(f"Save recording error: {e}")
            raise


# 使用WeNet进行语音识别
def SpeechRecognition(input_file, output_file, language, model_dir, gpu=0):
    try:
        if not os.path.exists(input_file):
            raise Exception(f"File {input_file} does not exist")

        if os.path.getsize(input_file) == 0:
            raise Exception(f"File {input_file} is empty")

        convert_audio(input_file, output_file)
        WeNet_model = wenet.load_model(language=language,
                                       model_dir=model_dir,
                                       gpu=gpu)
        result = WeNet_model.transcribe(output_file)
        log.info(result)
        recognized_text = result["text"].replace('▁', ' ').strip().lower()
        confidence = result["confidence"]
        return recognized_text, confidence
    except Exception as e:
        log.error(f"Speech recognition error: {e}")
        return "识别失败", 0.00


# 将语音文件转换成16k采样率的wav文件
def convert_audio(input_filepath, output_filepath):
    try:
        data, samplerate = soundfile.read(input_filepath)
        log.debug(f"Converting audio, samplerate: {samplerate}")
        soundfile.write(output_filepath, data, samplerate, subtype='PCM_16')
        log.debug("Converted audio")
    except Exception as e:
        log.error(f"Audio conversion error: {e}")
        raise


# 识别线程：调用WeNet进行语音识别
class RecognitionThread(QtCore.QThread):
    recognitionFinished = QtCore.pyqtSignal(str, float)

    def __init__(self, input_file, output_file, src_lang, model_dir):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.src_lang = src_lang
        self.model_dir = model_dir

    def run(self):
        recognized_text, confidence = SpeechRecognition(self.input_file,
                                                        self.output_file,
                                                        self.src_lang,
                                                        self.model_dir)
        self.recognitionFinished.emit(recognized_text, confidence)


# 翻译线程：调用NLLB进行翻译
class TranslationThread(QtCore.QThread):
    translationFinished = QtCore.pyqtSignal(str)

    def __init__(self, recognized_text, src_lang, tgt_lang, translator):
        super().__init__()
        self.recognized_text = recognized_text
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator = translator

    def run(self):
        translated_text = translate(self.recognized_text,
                                    self.src_lang,
                                    self.tgt_lang,
                                    self.translator)
        self.translationFinished.emit(translated_text)


# 调用NLLB进行翻译
def translate(text, src_lang, tgt_lang, translator):
    try:
        translated_text = translator(text,
                                     src_lang=src_lang,
                                     tgt_lang=tgt_lang)[0]['translation_text']
        log.info(f"{{'Translated text': '{translated_text}'}}")
        return translated_text
    except Exception as e:
        log.error(f"Translation error: {e}")
        return f"翻译失败:\n{e}"


# 模型加载线程
class ModelLoadingThread(QtCore.QThread):
    modelLoaded = QtCore.pyqtSignal(object, object, object)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # 默认使用GPU
        device = 0 if torch.cuda.is_available() else -1

        # 加载NLLB模型
        nllb_model_path = self.config['nllb_model']['path']
        tokenizer = AutoTokenizer.from_pretrained(nllb_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_path).to(device)
        translator = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=device
        )
        self.modelLoaded.emit(tokenizer, model, translator)


# 界面类
class TranslatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.translator = None  # 翻译器
        self.model = None  # NLLB模型
        self.tokenizer = None  # NLLB分词器
        self.recognitionThread = None  # 识别线程
        self.translationThread = None  # 翻译线程
        self.recognizedTextBox = None  # 识别文本框
        self.confidenceLabel = None  # 置信度标签
        self.promptText = None  # 提示文本
        self.translatedTextBox = None  # 翻译文本框
        self.tgtLangBox = None  # 翻译语言选择框
        self.srcLangBox = None  # 语音语言选择框
        self.uploadFileButton = None  # 上传文件按钮
        self.startRecordButton = None  # 开始录音按钮
        self.initUI()

        self.recorderThread = RecorderThread()
        self.recorderThread.recordingStarted.connect(self.onRecordingStarted)
        self.modelLoadingThread = ModelLoadingThread(config)
        self.modelLoadingThread.modelLoaded.connect(self.onModelLoaded)
        self.modelLoadingThread.start()

    def initUI(self):
        self.promptText = QtWidgets.QLabel('选择或录制语音文件', self)
        self.startRecordButton = QtWidgets.QPushButton('开始录音', self)
        self.uploadFileButton = QtWidgets.QPushButton('上传音频', self)
        self.srcLangBox = QtWidgets.QComboBox(self)
        self.tgtLangBox = QtWidgets.QComboBox(self)

        self.recognizedTextBox = QtWidgets.QTextEdit(self)
        self.recognizedTextBox.setReadOnly(True)

        self.confidenceLabel = QtWidgets.QLabel('置信度: 0.00', self)

        self.translatedTextBox = QtWidgets.QTextEdit(self)
        self.translatedTextBox.setReadOnly(True)

        for lang in config['languages']:
            self.srcLangBox.addItem(lang['name'])
            self.tgtLangBox.addItem(lang['name'])
        # 设置默认语翻译语言为English
        self.tgtLangBox.setCurrentText('English')

        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(QtWidgets.QLabel('', self))

        # 创建语言选择布局
        lang_layout = QtWidgets.QVBoxLayout()

        # 水平布局
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addWidget(QtWidgets.QLabel('语音语言', self))
        label_layout.addWidget(QtWidgets.QLabel('翻译语言', self))

        combo_layout = QtWidgets.QHBoxLayout()
        combo_layout.addWidget(self.srcLangBox)
        combo_layout.addWidget(self.tgtLangBox)

        lang_layout.addLayout(label_layout)
        lang_layout.addLayout(combo_layout)

        main_layout.addLayout(lang_layout)

        main_layout.addWidget(QtWidgets.QLabel('', self))
        main_layout.addWidget(self.promptText)
        main_layout.addWidget(self.startRecordButton)
        main_layout.addWidget(self.uploadFileButton)
        main_layout.addWidget(QtWidgets.QLabel('', self))
        main_layout.addWidget(QtWidgets.QLabel('识别结果', self))
        main_layout.addWidget(QtWidgets.QLabel('语音文本:', self))
        main_layout.addWidget(self.recognizedTextBox)
        main_layout.addWidget(self.confidenceLabel)
        main_layout.addWidget(QtWidgets.QLabel('', self))
        main_layout.addWidget(QtWidgets.QLabel('翻译结果:', self))
        main_layout.addWidget(self.translatedTextBox)
        main_layout.addWidget(QtWidgets.QLabel('', self))

        self.setLayout(main_layout)

        # 信号槽
        self.startRecordButton.clicked.connect(self.startRecording)
        self.uploadFileButton.clicked.connect(self.uploadFile)

        self.setWindowTitle('语音翻译')
        self.setGeometry(350, 350, 350, 400)

    def startRecording(self):
        if self.startRecordButton.text() == '开始录音':
            self.promptText.setText('正在录音...')
            self.startRecordButton.setText('等待录音...')
            self.startRecordButton.setDisabled(True)
            self.srcLangBox.setDisabled(True)
            self.tgtLangBox.setDisabled(True)
            self.uploadFileButton.setDisabled(True)
            self.recognizedTextBox.setTextColor(QtGui.QColor('gray'))
            self.recognizedTextBox.setText('正在识别......')
            self.translatedTextBox.setTextColor(QtGui.QColor('gray'))
            self.confidenceLabel.setText('置信度: 0.00')
            self.translatedTextBox.setText('正在翻译......')
            self.recorderThread.startRecording()
        else:
            self.promptText.setText('选择或录制语音文件')
            self.startRecordButton.setText('开始录音')
            self.recorderThread.stopRecording()
            self.startRecordButton.setDisabled(True)
            self.srcLangBox.setDisabled(True)
            self.tgtLangBox.setDisabled(True)
            self.uploadFileButton.setDisabled(True)
            input_file = 'recorded_audio_file.wav'
            try:
                self.recorderThread.saveRecording(input_file)
                output_file = 'processed_audio_file.wav'
                src_lang, model_dir = getLanguageCode(self.srcLangBox.currentText())
                self.recognitionThread = RecognitionThread(input_file,
                                                           output_file,
                                                           src_lang,
                                                           model_dir)
                self.recognitionThread.recognitionFinished.connect(self.onRecognitionFinished)
                self.recognitionThread.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.resetUI()

    def onRecordingStarted(self):
        self.startRecordButton.setText('停止录音')
        self.startRecordButton.setDisabled(False)
        self.srcLangBox.setDisabled(True)
        self.tgtLangBox.setDisabled(True)
        self.uploadFileButton.setDisabled(True)

    def onRecognitionFinished(self, recognized_text, confidence):
        self.recognizedTextBox.setTextColor(QtGui.QColor('black'))
        self.recognizedTextBox.setText(recognized_text)
        self.confidenceLabel.setText(f'置信度: {confidence:.2f}')
        tgt_lang = getLanguageCode(self.tgtLangBox.currentText())[0]
        self.translationThread = TranslationThread(recognized_text, self.srcLangBox.currentText(), tgt_lang,
                                                   self.translator)
        self.translationThread.translationFinished.connect(self.onTranslationFinished)
        self.translationThread.start()

    def onTranslationFinished(self, translated_text):
        self.translatedTextBox.setTextColor(QtGui.QColor('black'))
        self.translatedTextBox.setText(translated_text)
        self.resetUI()

    def uploadFile(self):
        self.promptText.setText('选择语音文件')
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "上传音频文件", "",
                                                  "Audio Files (*.wav *.mp3 *.m4a *.flac *.acc);;All Files (*)",
                                                  options=options)
        if filePath:
            self.promptText.setText('选择或录制语音文件')
            self.recognizedTextBox.setTextColor(QtGui.QColor('gray'))
            self.recognizedTextBox.setText('正在识别......')
            self.translatedTextBox.setTextColor(QtGui.QColor('gray'))
            self.translatedTextBox.setText('正在翻译......')
            self.uploadFileButton.setDisabled(True)
            self.startRecordButton.setDisabled(True)
            self.srcLangBox.setDisabled(True)
            self.tgtLangBox.setDisabled(True)
            QtCore.QCoreApplication.processEvents()
            output_file = 'audio_file.wav'
            src_lang, model_dir = getLanguageCode(self.srcLangBox.currentText())
            self.recognitionThread = RecognitionThread(filePath, output_file, src_lang, model_dir)
            self.recognitionThread.recognitionFinished.connect(self.onRecognitionFinished)
            self.recognitionThread.start()
        else:
            log.debug("未选择文件")
            self.promptText.setText('选择或录制语音文件')

    def resetUI(self):
        self.promptText.setText('选择或录制语音文件')
        self.startRecordButton.setText('开始录音')
        self.startRecordButton.setDisabled(False)
        self.srcLangBox.setDisabled(False)
        self.tgtLangBox.setDisabled(False)
        self.uploadFileButton.setDisabled(False)

    def onModelLoaded(self, tokenizer, model, translator):
        self.tokenizer = tokenizer
        self.model = model
        self.translator = translator
        log.info(f"NLLB model loaded from {config['nllb_model']['path']}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('./Icon.svg'))
    translatorApp = TranslatorApp()
    translatorApp.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

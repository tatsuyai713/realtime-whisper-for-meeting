from __future__ import annotations

import os
import platform
import queue
import signal
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioCapture, list_audio_devices, preferred_device_index
from .local_service import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WHISPER_MODEL,
    LOCAL_TRANSLATION_MODELS,
    LocalProcessor,
    check_ollama,
    local_engine_label,
    whisperkit_available,
)
from .models import (
    AudioChunk,
    AudioDevice,
    TranscriptSegment,
    merge_continued_transcript,
    split_transcript_sentences,
)
from .openai_service import TRANSCRIPTION_MODEL, TRANSLATION_MODEL, OpenAIProcessor
from .performance import resolve_local_performance
from .translation_client import check_translation_server
from .translation_protocol import (
    DEFAULT_TRANSLATION_SERVER_HOST,
    DEFAULT_TRANSLATION_SERVER_PORT,
)

LANGUAGES = [
    ("自動検出", None),
    ("英語", "en"),
    ("中国語", "zh"),
    ("韓国語", "ko"),
    ("フランス語", "fr"),
    ("ドイツ語", "de"),
    ("スペイン語", "es"),
    ("イタリア語", "it"),
    ("ポルトガル語", "pt"),
    ("ロシア語", "ru"),
    ("日本語", "ja"),
]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class AppSignals(QObject):
    transcript = Signal(object)
    translation = Signal(int, str)
    status = Signal(str)
    error = Signal(str)
    level = Signal(float)
    processor_ready = Signal()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Meeting Live Translator")
        self.resize(1240, 780)
        self.audio_queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=30)
        self.capture: AudioCapture | None = None
        self.processor: OpenAIProcessor | LocalProcessor | None = None
        self.devices: list[AudioDevice] = []
        self.running = False
        self.started_at: datetime | None = None
        self.is_macos = platform.system() == "Darwin"
        self.is_apple_silicon = _is_apple_silicon()
        self._original_blocks: list[str] = []
        self._translation_blocks: list[str] = []
        self._continues_by_sequence: dict[int, bool] = {}
        self._applying_performance_profile = False
        self.signals = AppSignals()
        self._connect_signals()
        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self.refresh_devices)

    def _connect_signals(self) -> None:
        self.signals.transcript.connect(self._append_transcript)
        self.signals.translation.connect(self._append_translation)
        self.signals.status.connect(self._set_status)
        self.signals.error.connect(self._show_runtime_error)
        self.signals.level.connect(self._set_level)
        self.signals.processor_ready.connect(self._start_audio_capture)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(16)

        title_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Meeting Live Translator")
        title.setObjectName("title")
        subtitle = QLabel("会議の音声をリアルタイムで書き起こしと日本語訳を行います")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_row.addLayout(title_box)
        title_row.addStretch()
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_label = QLabel("停止中")
        self.status_label.setObjectName("status")
        title_row.addWidget(self.status_dot)
        title_row.addWidget(self.status_label)
        layout.addLayout(title_row)

        controls = QFrame()
        controls.setObjectName("controls")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setSpacing(10)

        audio_row = QHBoxLayout()
        audio_row.setSpacing(12)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.refresh_button = QPushButton("再読込")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.language_combo = QComboBox()
        for label, code in LANGUAGES:
            self.language_combo.addItem(label, code)
        audio_row.addWidget(QLabel("取得する音声"))
        audio_row.addWidget(self.device_combo, 1)
        audio_row.addWidget(self.refresh_button)
        audio_row.addWidget(QLabel("発話言語"))
        audio_row.addWidget(self.language_combo)
        controls_layout.addLayout(audio_row)

        processing_row = QHBoxLayout()
        processing_row.setSpacing(12)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("OpenAI（クラウド・高精度）", "openai")
        self.backend_combo.addItem("Local（PC内で処理）", "local")
        self.backend_combo.currentIndexChanged.connect(self._update_processing_options)

        self.translation_check = QCheckBox("日本語翻訳")
        self.translation_check.setChecked(True)
        self.translation_check.toggled.connect(self._update_processing_options)

        self.api_key_label = QLabel("APIキー")
        self.api_key_edit = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("OpenAI API key")
        self.api_key_edit.setMinimumWidth(220)

        self.local_model_label = QLabel("音声モデル")
        self.local_engine_label = QLabel("高速化")
        self.local_engine_combo = QComboBox()
        if self.is_apple_silicon:
            self.local_engine_combo.addItem("NPU（Neural Engine）", "whisperkit")
            self.local_engine_combo.addItem("GPU（Metal）", "mlx")
        else:
            self.local_engine_combo.addItem("自動", "auto")
            self.local_engine_combo.addItem("GPU（NVIDIA CUDA）", "cuda")
            self.local_engine_combo.addItem("CPU", "cpu")
        self.local_engine_combo.currentIndexChanged.connect(self._select_custom_performance)

        self.local_model_combo = QComboBox()
        for label, model in (
            ("tiny", "tiny"),
            ("base", "base"),
            ("small", "small"),
            ("small 圧縮 216MB", "small-fast"),
            ("turbo 4bit 632MB", "turbo"),
            ("medium", "medium"),
            ("large-v3 圧縮 626MB", "large-v3-compressed"),
            ("large-v3", "large-v3"),
        ):
            self.local_model_combo.addItem(label, model)
        model_index = self.local_model_combo.findData(
            os.environ.get("LOCAL_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
        )
        self.local_model_combo.setCurrentIndex(max(0, model_index))
        self.local_model_combo.currentIndexChanged.connect(self._select_custom_performance)

        self.performance_profile_label = QLabel("Mac最適化")
        self.performance_profile_combo = QComboBox()
        if self.is_apple_silicon:
            self.performance_profile_combo.addItem("高速（圧縮small）", "mac-fast")
            self.performance_profile_combo.addItem("バランス（Turbo 4bit）", "mac-balanced")
            self.performance_profile_combo.addItem("高精度（large-v3）", "mac-accuracy")
        self.performance_profile_combo.addItem("カスタム", "custom")
        default_profile = os.environ.get(
            "LOCAL_PERFORMANCE_PROFILE",
            "mac-fast" if self.is_apple_silicon else "custom",
        )
        profile_index = self.performance_profile_combo.findData(default_profile)
        self.performance_profile_combo.setCurrentIndex(max(0, profile_index))

        self.compute_profile_label = QLabel("Core ML配置")
        self.compute_profile_combo = QComboBox()
        self.compute_profile_combo.addItem("Neural Engine", "ane")
        self.compute_profile_combo.addItem("ANE + GPU", "hybrid")
        self.compute_profile_combo.addItem("GPU", "gpu")
        compute_index = self.compute_profile_combo.findData(
            os.environ.get("WHISPERKIT_COMPUTE_PROFILE", "ane")
        )
        self.compute_profile_combo.setCurrentIndex(max(0, compute_index))
        self.compute_profile_combo.currentIndexChanged.connect(self._select_custom_performance)
        self.performance_profile_combo.currentIndexChanged.connect(
            self._apply_performance_profile
        )
        self.ollama_model_label = QLabel("翻訳モデル")
        self.ollama_model_edit = QComboBox()
        self.ollama_model_edit.setEditable(True)
        self.ollama_model_edit.setMinimumWidth(220)
        for label, model in LOCAL_TRANSLATION_MODELS:
            self.ollama_model_edit.addItem(label, model)
        configured_translation_model = os.environ.get(
            "OLLAMA_TRANSLATION_MODEL", DEFAULT_OLLAMA_MODEL
        )
        configured_model_index = self.ollama_model_edit.findData(configured_translation_model)
        if configured_model_index >= 0:
            self.ollama_model_edit.setCurrentIndex(configured_model_index)
        else:
            self.ollama_model_edit.setEditText(configured_translation_model)
        self.ollama_model_edit.currentTextChanged.connect(self._update_processing_options)

        self.translation_target_label = QLabel("翻訳実行")
        self.translation_target_combo = QComboBox()
        self.translation_target_combo.addItem("このPC（Ollama）", "local")
        self.translation_target_combo.addItem("別PC（翻訳サーバー）", "server")
        target = os.environ.get("TRANSLATION_TARGET", "local")
        self.translation_target_combo.setCurrentIndex(1 if target == "server" else 0)

        self.translation_server_host_label = QLabel("IP / ホスト")
        self.translation_server_host_edit = QLineEdit(
            os.environ.get("TRANSLATION_SERVER_HOST", DEFAULT_TRANSLATION_SERVER_HOST)
        )
        self.translation_server_host_edit.setPlaceholderText("192.168.1.100")
        self.translation_server_host_edit.setMinimumWidth(150)
        self.translation_server_host_edit.textChanged.connect(self._update_processing_options)
        self.translation_server_port_label = QLabel("ポート")
        self.translation_server_port_spin = QSpinBox()
        self.translation_server_port_spin.setRange(1, 65535)
        try:
            server_port = int(
                os.environ.get(
                    "TRANSLATION_SERVER_PORT",
                    str(DEFAULT_TRANSLATION_SERVER_PORT),
                )
            )
        except ValueError:
            server_port = DEFAULT_TRANSLATION_SERVER_PORT
        self.translation_server_port_spin.setValue(server_port)
        self.translation_server_port_spin.valueChanged.connect(self._update_processing_options)
        self.translation_target_combo.currentIndexChanged.connect(
            self._update_processing_options
        )

        self.start_button = QPushButton("開始")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.toggle_capture)

        processing_row.addWidget(QLabel("処理"))
        processing_row.addWidget(self.backend_combo)
        processing_row.addWidget(self.translation_check)
        processing_row.addWidget(self.api_key_label)
        processing_row.addWidget(self.api_key_edit, 1)
        processing_row.addWidget(self.local_engine_label)
        processing_row.addWidget(self.local_engine_combo)
        processing_row.addWidget(self.local_model_label)
        processing_row.addWidget(self.local_model_combo)
        processing_row.addWidget(self.ollama_model_label)
        processing_row.addWidget(self.ollama_model_edit)
        processing_row.addStretch()
        processing_row.addWidget(self.start_button)
        controls_layout.addLayout(processing_row)

        performance_row = QHBoxLayout()
        performance_row.setSpacing(12)
        performance_row.addWidget(self.performance_profile_label)
        performance_row.addWidget(self.performance_profile_combo)
        performance_row.addWidget(self.compute_profile_label)
        performance_row.addWidget(self.compute_profile_combo)
        performance_row.addStretch()
        controls_layout.addLayout(performance_row)

        translation_row = QHBoxLayout()
        translation_row.setSpacing(12)
        translation_row.addWidget(self.translation_target_label)
        translation_row.addWidget(self.translation_target_combo)
        translation_row.addWidget(self.translation_server_host_label)
        translation_row.addWidget(self.translation_server_host_edit, 1)
        translation_row.addWidget(self.translation_server_port_label)
        translation_row.addWidget(self.translation_server_port_spin)
        translation_row.addStretch()
        controls_layout.addLayout(translation_row)
        layout.addWidget(controls)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(4)
        layout.addWidget(self.level_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_edit, self.original_panel, self.original_caption = self._make_text_panel(
            splitter, "原文", f"高精度文字起こし · {TRANSCRIPTION_MODEL}"
        )
        self.translation_edit, self.translation_panel, self.translation_caption = (
            self._make_text_panel(splitter, "日本語訳", f"高品質翻訳 · {TRANSLATION_MODEL}")
        )
        splitter.setSizes([600, 600])
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.hint_label = QLabel("音声デバイスを選び、「開始」を押してください")
        self.hint_label.setObjectName("hint")
        clear_button = QPushButton("クリア")
        clear_button.clicked.connect(self.clear_text)
        export_button = QPushButton("保存")
        export_button.clicked.connect(self.export_text)
        self.permission_button = QPushButton("Mac音声権限")
        self.permission_button.setToolTip("macOSの「画面収録とシステムオーディオ」設定を開きます")
        self.permission_button.clicked.connect(self.open_macos_audio_permissions)
        self.permission_button.setVisible(self.is_macos)
        footer.addWidget(self.hint_label)
        footer.addStretch()
        footer.addWidget(self.permission_button)
        footer.addWidget(clear_button)
        footer.addWidget(export_button)
        layout.addLayout(footer)
        self.setCentralWidget(root)
        self._apply_performance_profile()

    def _apply_performance_profile(self) -> None:
        profile = self.performance_profile_combo.currentData()
        self._applying_performance_profile = True
        try:
            if profile != "custom":
                performance = resolve_local_performance(
                    profile,
                    custom_model=str(self.local_model_combo.currentData()),
                    custom_engine=str(self.local_engine_combo.currentData()),
                    custom_compute_profile=str(self.compute_profile_combo.currentData()),
                )
                for combo, value in (
                    (self.local_model_combo, performance.model),
                    (self.local_engine_combo, performance.engine),
                    (self.compute_profile_combo, performance.compute_profile),
                ):
                    index = combo.findData(value)
                    if index >= 0:
                        combo.setCurrentIndex(index)
        finally:
            self._applying_performance_profile = False
        self._update_processing_options()

    def _select_custom_performance(self) -> None:
        if self._applying_performance_profile:
            return
        custom_index = self.performance_profile_combo.findData("custom")
        if custom_index >= 0 and self.performance_profile_combo.currentIndex() != custom_index:
            self.performance_profile_combo.setCurrentIndex(custom_index)
        else:
            self._update_processing_options()

    def _make_text_panel(
        self, parent: QSplitter, heading: str, caption: str
    ) -> tuple[QPlainTextEdit, QFrame, QLabel]:
        panel = QFrame(parent)
        panel.setObjectName("panel")
        box = QVBoxLayout(panel)
        box.setContentsMargins(18, 16, 18, 18)
        header = QLabel(heading)
        header.setObjectName("panelHeading")
        note = QLabel(caption)
        note.setObjectName("panelCaption")
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlaceholderText("ここに表示されます…")
        editor.setFrameShape(QFrame.Shape.NoFrame)
        text_font = QFont()
        text_font.setPointSize(14)
        editor.setFont(text_font)
        box.addWidget(header)
        box.addWidget(note)
        box.addWidget(editor, 1)
        return editor, panel, note

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f8; color: #17212b; }
            QLabel { background: transparent; }
            QLabel#title { font-size: 25px; font-weight: 700; }
            QLabel#subtitle, QLabel#hint, QLabel#panelCaption { color: #687684; }
            QLabel#statusDot { color: #9aa5af; font-size: 18px; }
            QLabel#status { font-weight: 600; }
            QFrame#controls, QFrame#panel {
                background: white; border: 1px solid #dfe5ea; border-radius: 10px;
            }
            QLabel#panelHeading { font-size: 18px; font-weight: 700; }
            QPlainTextEdit { background: white; selection-background-color: #b8d8ff; }
            QComboBox, QLineEdit, QSpinBox, QPushButton {
                min-height: 34px; padding: 0 10px; border: 1px solid #ccd5dc;
                border-radius: 7px; background: white;
            }
            QPushButton:hover { background: #f0f5f8; }
            QPushButton#startButton {
                background: #1769e0; color: white; border: none;
                min-width: 76px; font-weight: 700;
            }
            QPushButton#startButton:hover { background: #0f5bc8; }
            QProgressBar { border: none; background: #e2e7eb; border-radius: 2px; }
            QProgressBar::chunk { background: #27a96b; border-radius: 2px; }
            QSplitter::handle { width: 12px; background: transparent; }
            """
        )

    def refresh_devices(self) -> None:
        if self.running:
            return
        self.device_combo.clear()
        try:
            self.devices = list_audio_devices()
        except Exception as exc:
            self.devices = []
            self._show_runtime_error(f"音声デバイスを取得できません: {exc}")
        for device in self.devices:
            suffix = "（システム音声）" if device.is_loopback else "（マイク）"
            self.device_combo.addItem(f"{device.name}{suffix}")
        preferred = preferred_device_index(self.devices)
        if preferred >= 0:
            self.device_combo.setCurrentIndex(preferred)
        if not self.devices:
            self.device_combo.addItem("利用できる音声デバイスがありません")

    def toggle_capture(self) -> None:
        if self.running:
            self.stop_capture()
        else:
            self.start_capture()

    def start_capture(self) -> None:
        key = self.api_key_edit.text().strip()
        index = self.device_combo.currentIndex()
        backend = self.backend_combo.currentData()
        translation_enabled = self.translation_check.isChecked()
        translation_target = self.translation_target_combo.currentData()
        performance = resolve_local_performance(
            str(self.performance_profile_combo.currentData()),
            custom_model=str(self.local_model_combo.currentData()),
            custom_engine=str(self.local_engine_combo.currentData()),
            custom_compute_profile=str(self.compute_profile_combo.currentData()),
        )
        if backend == "openai" and not key:
            QMessageBox.warning(self, "APIキーが必要です", "OpenAI APIキーを入力してください。")
            return
        if not self.devices or index < 0 or index >= len(self.devices):
            QMessageBox.warning(self, "音声デバイス", "利用する音声デバイスを選択してください。")
            return
        if translation_enabled and (backend == "local" or translation_target == "server"):
            ollama_model = self._selected_ollama_model()
            if translation_target == "server":
                ollama_ready, ollama_error = check_translation_server(
                    self.translation_server_host_edit.text(),
                    self.translation_server_port_spin.value(),
                    ollama_model,
                )
                recovery = (
                    "別PCで翻訳サーバーとOllamaを起動し、IPとポートを確認するか、"
                    "日本語翻訳をOFFにしてください。"
                )
            else:
                ollama_ready, ollama_error = check_ollama(ollama_model)
                recovery = (
                    "Ollamaを起動してモデルを準備するか、日本語翻訳をOFFにしてください。"
                )
            if not ollama_ready:
                QMessageBox.critical(
                    self,
                    "Local翻訳を開始できません",
                    ollama_error + "\n\n" + recovery,
                )
                self.translation_edit.appendPlainText(f"⚠ {ollama_error}\n")
                return
        if (
            backend == "local"
            and performance.engine == "whisperkit"
            and not whisperkit_available()
        ):
            QMessageBox.critical(
                self,
                "Neural Engineを使用できません",
                "WhisperKitがありません。ターミナルで "
                "`brew install whisperkit-cli` を実行するか、GPU（Metal）を選んでください。",
            )
            return

        self.audio_queue = queue.Queue(maxsize=30)
        callbacks = {
            "audio_queue": self.audio_queue,
            "on_transcript": self.signals.transcript.emit,
            "on_translation": self.signals.translation.emit,
            "on_status": self.signals.status.emit,
            "on_error": self.signals.error.emit,
        }
        if backend == "openai":
            self.processor = OpenAIProcessor(
                api_key=key,
                language=self.language_combo.currentData(),
                translation_enabled=translation_enabled,
                translation_target="server" if translation_target == "server" else "openai",
                ollama_model=self._selected_ollama_model(),
                translation_server_host=self.translation_server_host_edit.text().strip(),
                translation_server_port=self.translation_server_port_spin.value(),
                **callbacks,
            )
        else:
            self.processor = LocalProcessor(
                language=self.language_combo.currentData(),
                translation_enabled=translation_enabled,
                whisper_model=performance.model,
                ollama_model=self._selected_ollama_model(),
                local_engine=performance.engine,
                whisperkit_compute_profile=performance.compute_profile,
                translation_target=translation_target,
                translation_server_host=self.translation_server_host_edit.text().strip(),
                translation_server_port=self.translation_server_port_spin.value(),
                on_ready=self.signals.processor_ready.emit,
                **callbacks,
            )
        self.capture = AudioCapture(
            device=self.devices[index],
            output_queue=self.audio_queue,
            on_level=self.signals.level.emit,
            on_error=self.signals.error.emit,
            max_segment_seconds=(
                performance.max_segment_seconds if backend == "local" else 8.0
            ),
            overlap_seconds=performance.overlap_seconds if backend == "local" else 0.8,
        )
        self.running = True
        self.processor.start()
        if backend == "openai":
            self.capture.start()
        self.started_at = datetime.now()
        self.start_button.setText("停止")
        self.start_button.setStyleSheet("background: #d64545;")
        self.device_combo.setEnabled(False)
        self.language_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.api_key_edit.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.translation_check.setEnabled(False)
        self.local_model_combo.setEnabled(False)
        self.local_engine_combo.setEnabled(False)
        self.performance_profile_combo.setEnabled(False)
        self.compute_profile_combo.setEnabled(False)
        self.ollama_model_edit.setEnabled(False)
        self.translation_target_combo.setEnabled(False)
        self.translation_server_host_edit.setEnabled(False)
        self.translation_server_port_spin.setEnabled(False)
        self.status_dot.setStyleSheet("color: #27a96b;")
        if backend == "openai":
            self._set_status("音声を待っています…")

    def _start_audio_capture(self) -> None:
        if not self.running or self.capture is None:
            return
        self.capture.start()
        self._set_status("音声を待っています…")

    def stop_capture(self) -> None:
        self.running = False
        if self.capture:
            self.capture.stop()
        if self.processor:
            self.processor.stop()
        self.capture = None
        self.processor = None
        self.start_button.setText("開始")
        self.start_button.setStyleSheet("")
        self.device_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.api_key_edit.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.translation_check.setEnabled(True)
        self.local_model_combo.setEnabled(True)
        self.local_engine_combo.setEnabled(True)
        self.performance_profile_combo.setEnabled(True)
        self.compute_profile_combo.setEnabled(True)
        self.ollama_model_edit.setEnabled(True)
        self.translation_target_combo.setEnabled(True)
        self.translation_server_host_edit.setEnabled(True)
        self.translation_server_port_spin.setEnabled(True)
        self.status_dot.setStyleSheet("color: #9aa5af;")
        self._set_status("停止中")
        self.level_bar.setValue(0)
        self._update_processing_options()

    def _update_processing_options(self) -> None:
        is_openai = self.backend_combo.currentData() == "openai"
        translation_enabled = self.translation_check.isChecked()
        uses_translation_server = self.translation_target_combo.currentData() == "server"
        performance = resolve_local_performance(
            str(self.performance_profile_combo.currentData()),
            custom_model=str(self.local_model_combo.currentData()),
            custom_engine=str(self.local_engine_combo.currentData()),
            custom_compute_profile=str(self.compute_profile_combo.currentData()),
        )
        self.translation_target_combo.setItemText(
            0, "OpenAI（クラウド）" if is_openai else "このPC（Ollama）"
        )
        self.api_key_label.setVisible(is_openai)
        self.api_key_edit.setVisible(is_openai)
        self.local_model_label.setVisible(not is_openai)
        self.local_model_combo.setVisible(not is_openai)
        self.local_engine_label.setVisible(not is_openai)
        self.local_engine_combo.setVisible(not is_openai)
        show_mac_options = not is_openai and self.is_apple_silicon
        self.performance_profile_label.setVisible(show_mac_options)
        self.performance_profile_combo.setVisible(show_mac_options)
        show_core_ml_options = (
            not is_openai and self.is_apple_silicon and performance.engine == "whisperkit"
        )
        self.compute_profile_label.setVisible(show_core_ml_options)
        self.compute_profile_combo.setVisible(show_core_ml_options)
        if not self.running:
            self.local_model_combo.setEnabled(not is_openai)
            self.local_engine_combo.setEnabled(not is_openai)
            self.compute_profile_combo.setEnabled(not is_openai)
        show_ollama_model = translation_enabled and (
            not is_openai or uses_translation_server
        )
        self.ollama_model_label.setVisible(show_ollama_model)
        self.ollama_model_edit.setVisible(show_ollama_model)
        self.translation_target_label.setVisible(translation_enabled)
        self.translation_target_combo.setVisible(translation_enabled)
        self.translation_server_host_label.setVisible(
            translation_enabled and uses_translation_server
        )
        self.translation_server_host_edit.setVisible(
            translation_enabled and uses_translation_server
        )
        self.translation_server_port_label.setVisible(
            translation_enabled and uses_translation_server
        )
        self.translation_server_port_spin.setVisible(
            translation_enabled and uses_translation_server
        )
        self.translation_panel.setVisible(translation_enabled)
        if is_openai:
            self.original_caption.setText(f"高精度文字起こし · {TRANSCRIPTION_MODEL}")
            if uses_translation_server:
                server = (
                    f"{self.translation_server_host_edit.text()}:"
                    f"{self.translation_server_port_spin.value()}"
                )
                self.translation_caption.setText(
                    f"別PC翻訳 · {server} · Ollama {self._selected_ollama_model()}"
                )
            else:
                self.translation_caption.setText(f"高品質翻訳 · {TRANSLATION_MODEL}")
        else:
            self.original_caption.setText(
                "完全ローカル文字起こし · "
                f"{local_engine_label(performance.engine)} · "
                f"{self.local_model_combo.currentText()}"
            )
            if uses_translation_server:
                server = (
                    f"{self.translation_server_host_edit.text()}:"
                    f"{self.translation_server_port_spin.value()}"
                )
                self.translation_caption.setText(
                    f"別PC翻訳 · {server} · Ollama {self._selected_ollama_model()}"
                )
            else:
                self.translation_caption.setText(
                    f"完全ローカル翻訳 · Ollama {self._selected_ollama_model()}"
                )

    def _selected_ollama_model(self) -> str:
        return str(self.ollama_model_edit.currentData() or self.ollama_model_edit.currentText()).strip() or DEFAULT_OLLAMA_MODEL

    def _append_transcript(self, segment: TranscriptSegment) -> None:
        self._continues_by_sequence[segment.sequence] = segment.continues_previous
        if segment.continues_previous and self._original_blocks:
            self._original_blocks[-1] = merge_continued_transcript(
                self._original_blocks[-1],
                segment.original,
            )
        else:
            self._original_blocks.append(segment.original)
        self._render_blocks(self.original_edit, self._original_blocks)

    def _append_translation(self, sequence: int, text: str) -> None:
        if self._continues_by_sequence.get(sequence, False) and self._translation_blocks:
            self._translation_blocks[-1] = merge_continued_transcript(
                self._translation_blocks[-1],
                text,
            )
        else:
            self._translation_blocks.append(text)
        self._render_blocks(self.translation_edit, self._translation_blocks)

    @staticmethod
    def _render_blocks(editor: QPlainTextEdit, blocks: list[str]) -> None:
        paragraphs = [
            sentence for block in blocks for sentence in split_transcript_sentences(block)
        ]
        editor.setPlainText("\n\n".join(paragraphs))
        scrollbar = editor.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.hint_label.setText(text)

    def _set_level(self, value: float) -> None:
        self.level_bar.setValue(int(value * 100))

    def _show_runtime_error(self, message: str) -> None:
        self.hint_label.setText(message)
        self.status_label.setText("エラー")
        if "翻訳" in message or "Ollama" in message:
            self.translation_edit.appendPlainText(f"⚠ {message}\n")
        else:
            self.original_edit.appendPlainText(f"⚠ {message}\n")

    def clear_text(self) -> None:
        self._original_blocks.clear()
        self._translation_blocks.clear()
        self._continues_by_sequence.clear()
        self.original_edit.clear()
        self.translation_edit.clear()

    def open_macos_audio_permissions(self) -> None:
        if platform.system() != "Darwin":
            return
        subprocess.Popen(
            [
                "open",
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
            ]
        )

    def export_text(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "文字起こしを保存", "meeting-transcript.txt", "Text (*.txt)"
        )
        if not path:
            return
        content = (
            "【原文】\n"
            + self.original_edit.toPlainText().strip()
            + "\n\n【日本語訳】\n"
            + self.translation_edit.toPlainText().strip()
            + "\n"
        )
        with open(path, "w", encoding="utf-8") as output:
            output.write(content)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.running:
            self.stop_capture()
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Meeting Live Translator")
    window = MainWindow()

    # Qt's event loop can otherwise keep Python from dispatching SIGINT until
    # another Python callback runs. A lightweight timer gives the interpreter
    # regular opportunities to run the signal handler.
    interrupt_timer = QTimer()
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)

    def stop_from_terminal(_signum: int, _frame: object) -> None:
        window.close()

    signal.signal(signal.SIGINT, stop_from_terminal)
    signal.signal(signal.SIGTERM, stop_from_terminal)
    window.show()
    return application.exec()

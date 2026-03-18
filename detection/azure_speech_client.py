"""
Azure Speech Service 快速转录客户端（Fast Transcription API）

使用 Azure Cognitive Services 的 Fast Transcription REST API，支持：
    - 快速批量转录（比实时音频处理更快）
    - 说话人分离（Speaker Diarization，区分多个说话人）
    - 多语言支持（zh-CN, ja-JP, en-US 等）
    - 自动语言识别（传入多个候选语言）

API 文档：
    https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/fast-transcription-create

前提：
    - 已在 Azure 创建 Speech 服务资源
    - 获取 Subscription Key 和 Region（如 japaneast）

用法：
    client = AzureSpeechClient(subscription_key="xxx", region="japaneast")
    result = client.transcribe("audio.wav", locales=["zh-CN"], max_speakers=2)
    for phrase in result.phrases:
        print(f"[Speaker {phrase.speaker}] {phrase.text}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# API 配置
# ─────────────────────────────────────────────

API_VERSION = "2024-11-15"
API_PATH = "speechtotext/transcriptions:transcribe"

# 超时设置（Fast Transcription 一般在数秒内返回）
REQUEST_TIMEOUT_SECONDS = 120


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class TranscriptionPhrase:
    """单段转录结果（对应一句话）"""
    offset_ms: int          # 相对音频开始的偏移（毫秒）
    duration_ms: int        # 本段持续时长（毫秒）
    text: str               # 转录文本
    speaker: Optional[int]  # 说话人编号（启用 diarization 时有值，从 1 开始）
    locale: str             # 识别语言（如 zh-CN）
    confidence: float       # 置信度（0.0~1.0）


@dataclass
class TranscriptionResult:
    """完整转录结果"""
    duration_ms: int                            # 音频总时长（毫秒）
    full_text: str                              # 所有说话人合并后的完整文本
    phrases: list[TranscriptionPhrase]          # 逐句结果（含说话人标签）
    speaker_count: int = 0                      # 检测到的说话人数量
    raw_response: dict = field(default_factory=dict)  # 原始 API 响应（调试用）

    def format_transcript(self) -> str:
        """
        格式化为可读的对话文本，按说话人分组显示。

        示例输出：
            [说话人 1 | 0.0s] 你好，请问您今天想了解哪类产品？
            [说话人 2 | 3.2s] 我对基金比较感兴趣。
        """
        lines = []
        for p in self.phrases:
            offset_sec = p.offset_ms / 1000
            speaker_label = f"说话人 {p.speaker}" if p.speaker is not None else "未知说话人"
            lines.append(f"[{speaker_label} | {offset_sec:.1f}s] {p.text}")
        return "\n".join(lines)

    def get_speakers(self) -> set[int]:
        """返回所有检测到的说话人编号集合"""
        return {p.speaker for p in self.phrases if p.speaker is not None}


# ─────────────────────────────────────────────
# Azure Speech 客户端
# ─────────────────────────────────────────────

class AzureSpeechClient:
    """
    Azure Speech Service 快速转录客户端。

    用法：
        client = AzureSpeechClient(
            subscription_key="your_key",
            region="japaneast",
        )
        result = client.transcribe(
            audio_path="conversation.wav",
            locales=["zh-CN"],
            max_speakers=2,
        )
    """

    def __init__(
        self,
        subscription_key: str,
        region: str = "japaneast",
        api_version: str = API_VERSION,
    ):
        """
        Args:
            subscription_key: Azure Speech 服务的 Subscription Key
            region:           Azure 区域（如 japaneast、eastus）
            api_version:      API 版本（默认 2024-11-15）
        """
        self.subscription_key = subscription_key
        self.region = region
        self.api_version = api_version
        self.endpoint = (
            f"https://{region}.api.cognitive.microsoft.com"
            f"/{API_PATH}?api-version={api_version}"
        )
        logger.info(f"AzureSpeechClient 初始化，region={region}, api_version={api_version}")

    def transcribe(
        self,
        audio_path: str | Path,
        locales: list[str] | None = None,
        max_speakers: int = 2,
        enable_diarization: bool = True,
        profanity_filter: str = "None",
    ) -> TranscriptionResult:
        """
        发送音频文件进行快速转录。

        Args:
            audio_path:         音频文件路径（支持 WAV/MP3/OGG/FLAC/M4A 等）
            locales:            目标语言列表（如 ["zh-CN"]，多个则触发语言识别）
                                支持：zh-CN, ja-JP, en-US, ko-KR, de-DE 等
            max_speakers:       说话人分离的最大说话人数（2~36，仅单声道有效）
            enable_diarization: 是否启用说话人分离（默认开启）
            profanity_filter:   敏感词过滤模式（"None"/"Masked"/"Removed"）

        Returns:
            TranscriptionResult，含逐句文本和说话人标签

        Raises:
            requests.HTTPError:  API 调用失败
            FileNotFoundError:   音频文件不存在
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在：{audio_path}")

        if locales is None:
            locales = ["zh-CN"]

        # 构建 definition JSON（API 的配置参数）
        definition: dict = {
            "locales": locales,
            "profanityFilterMode": profanity_filter,
        }
        if enable_diarization:
            definition["diarization"] = {
                "enabled": True,
                "maxSpeakers": max_speakers,
            }

        logger.info(
            f"发送转录请求：{audio_path.name}，语言：{locales}，"
            f"说话人分离：{enable_diarization}（最多 {max_speakers} 人）"
        )

        # multipart/form-data 请求
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                url=self.endpoint,
                headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
                files={
                    "audio": (audio_path.name, audio_file, _get_mime_type(audio_path)),
                    "definition": (None, json.dumps(definition, ensure_ascii=False), "application/json"),
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        response.raise_for_status()
        raw = response.json()
        logger.info(f"转录完成，HTTP {response.status_code}")

        return _parse_response(raw)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        locales: list[str] | None = None,
        max_speakers: int = 2,
        enable_diarization: bool = True,
    ) -> TranscriptionResult:
        """
        发送音频字节流进行快速转录（适用于内存中的音频数据）。

        Args:
            audio_bytes: 音频文件的字节内容
            filename:    虚拟文件名（用于 MIME 类型推断）
            其余参数同 transcribe()
        """
        if locales is None:
            locales = ["zh-CN"]

        definition: dict = {
            "locales": locales,
        }
        if enable_diarization:
            definition["diarization"] = {
                "enabled": True,
                "maxSpeakers": max_speakers,
            }

        logger.info(f"发送转录请求（字节流）：{filename}，语言：{locales}")

        mime_type = _get_mime_type(Path(filename))
        response = requests.post(
            url=self.endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
            files={
                "audio": (filename, audio_bytes, mime_type),
                "definition": (None, json.dumps(definition, ensure_ascii=False), "application/json"),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        raw = response.json()
        return _parse_response(raw)


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def make_azure_speech_client(
    subscription_key: str,
    region: str = "japaneast",
) -> AzureSpeechClient:
    """
    创建 AzureSpeechClient 实例（与 make_bedrock_client 风格一致）。

    Args:
        subscription_key: Azure Speech Subscription Key
        region:           Azure 区域（默认 japaneast）
    """
    return AzureSpeechClient(subscription_key=subscription_key, region=region)


# ─────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────

def _get_mime_type(path: Path) -> str:
    """根据文件扩展名返回 MIME 类型"""
    ext = path.suffix.lower()
    return {
        ".wav":  "audio/wav",
        ".mp3":  "audio/mpeg",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".m4a":  "audio/mp4",
        ".aac":  "audio/aac",
        ".webm": "audio/webm",
    }.get(ext, "audio/wav")  # 默认 wav


def _parse_response(raw: dict) -> TranscriptionResult:
    """
    解析 Azure Fast Transcription API 响应 JSON，转换为 TranscriptionResult。

    API 响应格式：
        {
          "durationMilliseconds": 182439,
          "combinedPhrases": [{"text": "全文..."}],
          "phrases": [
            {
              "offsetMilliseconds": 0,
              "durationMilliseconds": 1490,
              "text": "你好",
              "locale": "zh-CN",
              "confidence": 0.95,
              "speaker": 1       ← 仅当 diarization 启用时存在
            }
          ]
        }
    """
    duration_ms = raw.get("durationMilliseconds", 0)

    # 完整文本（所有说话人合并）
    combined = raw.get("combinedPhrases", [])
    full_text = combined[0]["text"] if combined else ""

    # 逐句解析
    phrases: list[TranscriptionPhrase] = []
    for p in raw.get("phrases", []):
        phrases.append(TranscriptionPhrase(
            offset_ms=p.get("offsetMilliseconds", 0),
            duration_ms=p.get("durationMilliseconds", 0),
            text=p.get("text", ""),
            speaker=p.get("speaker"),          # 可能为 None（未启用 diarization）
            locale=p.get("locale", ""),
            confidence=p.get("confidence", 0.0),
        ))

    # 统计说话人数量
    speakers = {p.speaker for p in phrases if p.speaker is not None}
    speaker_count = len(speakers)

    logger.info(
        f"解析完成：{len(phrases)} 句，时长 {duration_ms / 1000:.1f}s，"
        f"检测到 {speaker_count} 个说话人：{sorted(speakers)}"
    )

    return TranscriptionResult(
        duration_ms=duration_ms,
        full_text=full_text,
        phrases=phrases,
        speaker_count=speaker_count,
        raw_response=raw,
    )

"""
AWS Transcribe 流式转录客户端

封装 amazon-transcribe SDK，提供与 AzureSpeechClient 一致的接口。
支持：
    - WAV 文件转录（跳过 44 字节 WAV header，仅发送 PCM）
    - 字节流转录（适用于浏览器上传的音频数据）
    - 说话人分离（ShowSpeakerLabel，单声道）
    - 多语言（zh-CN / ja-JP / en-US 等）

依赖：
    pip install amazon-transcribe

音频格式要求：
    - 编码：PCM 16-bit little-endian（media_encoding="pcm"）
    - 采样率：16000 Hz（推荐）或 8000 Hz
    - 声道：单声道（mono）

说话人重建策略：
    逐字（item）级别拥有 speaker 字段，本模块将相邻同 speaker 的词语
    合并为短语（phrase），保持与 Azure 返回格式一致。
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

SAMPLE_RATE    = 16000
MEDIA_ENCODING = "pcm"
CHUNK_SIZE     = 1024 * 8   # 8 KB / chunk
WAV_HEADER_LEN = 44         # 标准 WAV header 长度（RIFF 标记）


# ─────────────────────────────────────────────
# 共享数据结构（与 azure_speech_client 格式一致）
# ─────────────────────────────────────────────

@dataclass
class TranscriptionPhrase:
    """单段转录结果（一个说话人的一句话）"""
    offset_ms:   int            # 相对音频开始的偏移（毫秒）
    duration_ms: int            # 本段持续时长（毫秒）
    text:        str            # 转录文本
    speaker:     Optional[int]  # 说话人编号（从 0 开始；Azure 从 1 开始，注意映射）
    locale:      str            # 识别语言
    confidence:  float          # 置信度（0.0~1.0）


@dataclass
class TranscriptionResult:
    """完整转录结果"""
    duration_ms:   int
    full_text:     str
    phrases:       list[TranscriptionPhrase]
    speaker_count: int = 0
    raw_response:  dict = field(default_factory=dict)

    def format_transcript(self) -> str:
        """格式化为可读对话文本：[说话人 N | X.Xs] 文本"""
        lines = []
        for p in self.phrases:
            offset_sec   = p.offset_ms / 1000
            speaker_label = f"说话人 {p.speaker}" if p.speaker is not None else "未知说话人"
            lines.append(f"[{speaker_label} | {offset_sec:.1f}s] {p.text}")
        return "\n".join(lines)

    def get_speakers(self) -> set[int]:
        return {p.speaker for p in self.phrases if p.speaker is not None}


# ─────────────────────────────────────────────
# AWS Transcribe 流式客户端
# ─────────────────────────────────────────────

class AWSSpeechClient:
    """
    AWS Transcribe 流式转录客户端（同步接口，内部使用 asyncio）。

    用法：
        client = AWSSpeechClient(
            access_key_id="AKIA...",
            secret_access_key="...",
            region="ap-northeast-1",
        )
        result = client.transcribe("conversation.wav", locales=["zh-CN"])
        print(result.format_transcript())
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str = "ap-northeast-1",
        sample_rate: int = SAMPLE_RATE,
    ):
        self.access_key_id     = access_key_id
        self.secret_access_key = secret_access_key
        self.region            = region
        self.sample_rate       = sample_rate

    # ── 公开同步接口 ──────────────────────────

    def transcribe(
        self,
        audio_path: str | Path,
        locales: list[str] | None = None,
        max_speakers: int = 2,
        enable_diarization: bool = True,
    ) -> TranscriptionResult:
        """
        转录 WAV/PCM 音频文件。

        Args:
            audio_path:         WAV 文件路径（16kHz 单声道 PCM）
            locales:            目标语言列表，取第一个（AWS 流式不支持自动检测）
            max_speakers:       说话人数上限（AWS 会自动判断，此参数预留）
            enable_diarization: 是否启用说话人分离

        Returns:
            TranscriptionResult
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在：{audio_path}")

        language_code = (locales or ["zh-CN"])[0]
        logger.info(f"开始转录文件：{audio_path.name}，语言：{language_code}")

        with open(audio_path, "rb") as f:
            raw = f.read()

        # 跳过 WAV header
        pcm_data = _strip_wav_header(raw)
        return asyncio.run(
            self._transcribe_pcm(pcm_data, language_code, enable_diarization)
        )

    def transcribe_pcm(
        self,
        pcm_data: bytes,
        language_code: str,
        enable_diarization: bool = True,
    ) -> TranscriptionResult:
        """
        直接转录裸 PCM 数据（16kHz 单声道）。

        用于会话级累积音频的场景：上层代码自行维护完整会话的 PCM 缓冲区，
        每次将「截至当前」的全部 PCM 发送给 AWS，以获得基于全局上下文的
        说话人分离和按停顿分段的结果。
        """
        logger.info(
            f"开始转录 PCM 流，长度={len(pcm_data)} bytes，语言：{language_code}"
        )
        return asyncio.run(
            self._transcribe_pcm(pcm_data, language_code, enable_diarization)
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        locales: list[str] | None = None,
        max_speakers: int = 2,
        enable_diarization: bool = True,
    ) -> TranscriptionResult:
        """
        转录音频字节流（适用于浏览器上传的音频数据）。

        音频需为 PCM 格式（WAV 容器或裸 PCM），16kHz 单声道。
        浏览器如使用 WebM/Opus，须在后端先转为 PCM（见 convert_to_pcm 辅助函数）。
        """
        language_code = (locales or ["zh-CN"])[0]
        logger.info(f"开始转录字节流：{filename}，语言：{language_code}")

        pcm_data = _strip_wav_header(audio_bytes)
        return asyncio.run(
            self._transcribe_pcm(pcm_data, language_code, enable_diarization)
        )

    # ── 内部异步实现 ──────────────────────────

    async def _transcribe_pcm(
        self,
        pcm_data: bytes,
        language_code: str,
        enable_diarization: bool,
    ) -> TranscriptionResult:
        """核心异步转录逻辑，复用 test_transcribe.py 验证过的流程。"""
        try:
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
        except ImportError:
            raise ImportError("缺少依赖：请先运行 pip install amazon-transcribe")

        # TranscribeStreamingClient 通过环境变量读取凭证
        os.environ["AWS_ACCESS_KEY_ID"]     = self.access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key
        os.environ["AWS_DEFAULT_REGION"]    = self.region

        # 收集所有 final 结果的 item 列表
        collected_items: list[dict] = []
        total_duration_ms: list[float] = [0.0]  # 用 list 绕过 closure 赋值限制

        class _Handler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, event: TranscriptEvent):
                for result in event.transcript.results:
                    if result.is_partial:
                        continue
                    for alt in result.alternatives:
                        if not alt.items:
                            continue
                        for item in alt.items:
                            end_time = float(getattr(item, "end_time", 0) or 0)
                            if end_time > total_duration_ms[0]:
                                total_duration_ms[0] = end_time
                            collected_items.append({
                                "type":       getattr(item, "type", "pronunciation"),
                                "content":    getattr(item, "content", ""),
                                "start_time": float(getattr(item, "start_time", 0) or 0),
                                "end_time":   end_time,
                                "speaker":    getattr(item, "speaker", None),
                                "confidence": float(getattr(item, "confidence", 1.0) or 1.0),
                            })

        client = TranscribeStreamingClient(region=self.region)
        stream = await client.start_stream_transcription(
            language_code=language_code,
            media_sample_rate_hz=self.sample_rate,
            media_encoding=MEDIA_ENCODING,
            show_speaker_label=enable_diarization,
        )
        handler = _Handler(stream.output_stream)

        async def _send():
            buf = io.BytesIO(pcm_data)
            chunk_count = 0
            while True:
                chunk = buf.read(CHUNK_SIZE)
                if not chunk:
                    break
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
                chunk_count += 1
            await stream.input_stream.end_stream()
            logger.debug(f"音频发送完成，共 {chunk_count} chunks")

        await asyncio.gather(_send(), handler.handle_events())

        return _build_result(collected_items, total_duration_ms[0], language_code)


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def make_aws_speech_client(
    access_key_id: str,
    secret_access_key: str,
    region: str = "ap-northeast-1",
) -> AWSSpeechClient:
    """创建 AWSSpeechClient 实例（与 make_bedrock_client 风格一致）。"""
    return AWSSpeechClient(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region,
    )


def make_aws_speech_client_from_csv(
    csv_path: str | Path,
    region: str = "ap-northeast-1",
) -> AWSSpeechClient:
    """从 AWS CSV 凭证文件创建 AWSSpeechClient。"""
    from detection.aws_client import load_credentials_from_csv
    access_key_id, secret_access_key = load_credentials_from_csv(csv_path)
    logger.info(f"已从 CSV 加载凭证，Access Key ID: {access_key_id[:8]}...")
    return make_aws_speech_client(access_key_id, secret_access_key, region)


# ─────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────

def _strip_wav_header(data: bytes) -> bytes:
    """
    检测并跳过 WAV/RIFF header（44 字节），返回裸 PCM 数据。
    若不是 RIFF 格式，假定已是裸 PCM，原样返回。
    """
    if data[:4] == b"RIFF":
        logger.debug("检测到 WAV 格式，跳过 44 字节 header")
        return data[WAV_HEADER_LEN:]
    return data


def _build_result(
    items: list[dict],
    total_duration_sec: float,
    language_code: str,
) -> TranscriptionResult:
    """
    将 AWS item 列表重建为 TranscriptionResult。

    策略：将相邻 speaker 相同的 pronunciation 词语合并为一个 phrase，
    punctuation 附加到前一个词语尾部。
    """
    if not items:
        return TranscriptionResult(
            duration_ms=int(total_duration_sec * 1000),
            full_text="",
            phrases=[],
            speaker_count=0,
        )

    phrases: list[TranscriptionPhrase] = []
    cur_speaker  = items[0].get("speaker")
    cur_words:   list[str]  = []
    cur_start    = items[0]["start_time"]
    cur_end      = items[0]["end_time"]
    cur_confs:   list[float] = []

    def _flush():
        if not cur_words:
            return
        text = " ".join(cur_words)
        # AWS speaker ID 是字符串 "0"/"1"，转为 int；None 保持 None
        speaker_int = int(cur_speaker) if cur_speaker is not None else None
        avg_conf = sum(cur_confs) / len(cur_confs) if cur_confs else 1.0
        phrases.append(TranscriptionPhrase(
            offset_ms   = int(cur_start * 1000),
            duration_ms = int((cur_end - cur_start) * 1000),
            text        = text,
            speaker     = speaker_int,
            locale      = language_code,
            confidence  = round(avg_conf, 3),
        ))

    for item in items:
        item_type = item.get("type", "pronunciation")
        speaker   = item.get("speaker")
        content   = item.get("content", "")

        if item_type == "punctuation":
            # 标点附加到当前词尾
            if cur_words:
                cur_words[-1] += content
            continue

        # 说话人切换 → flush 并开启新 phrase
        if speaker != cur_speaker:
            _flush()
            cur_speaker = speaker
            cur_words   = []
            cur_confs   = []
            cur_start   = item["start_time"]

        cur_words.append(content)
        cur_end = item["end_time"]
        cur_confs.append(item["confidence"])

    _flush()

    full_text = " ".join(p.text for p in phrases)
    speakers  = {p.speaker for p in phrases if p.speaker is not None}

    logger.info(
        f"转录完成：{len(phrases)} 个短语，时长 {total_duration_sec:.1f}s，"
        f"说话人 {sorted(speakers)}"
    )

    return TranscriptionResult(
        duration_ms   = int(total_duration_sec * 1000),
        full_text     = full_text,
        phrases       = phrases,
        speaker_count = len(speakers),
    )

"""
声纹锚定工具（本地 embedding + 质量检查）

优先使用 SpeechBrain ECAPA（若环境可用），否则回退到轻量频谱 embedding。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_ANCHOR_SECONDS = 3.0
MIN_SIGNAL_RMS = 0.005
MIN_SNR_DB = 8.0


@dataclass
class AnchorQuality:
    ok: bool
    reason: str  # "ok" | "too_short" | "low_snr"
    duration_sec: float
    snr_db: float


class VoiceEmbedder:
    """
    声纹 embedding 提取器。

    - 优先 ECAPA（SpeechBrain）
    - 不可用时自动回退到频谱 embedding（保证功能可用）
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, enable_ecapa: bool = True):
        self.sample_rate = sample_rate
        self._ecapa = None
        self._ecapa_ready = False
        if enable_ecapa:
            self._try_init_ecapa()

    def _try_init_ecapa(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except Exception as e:
            logger.warning(f"SpeechBrain 不可用，回退频谱 embedding：{e}")
            return

        try:
            cache_dir = Path("/tmp/ecapa_spkrec").as_posix()
            self._ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=cache_dir,
                run_opts={"device": "cpu"},
            )
            self._ecapa_ready = True
            logger.info("ECAPA 声纹模型加载成功")
        except Exception as e:
            logger.warning(f"ECAPA 模型加载失败，回退频谱 embedding：{e}")
            self._ecapa = None
            self._ecapa_ready = False

    def check_quality(self, pcm_data: bytes) -> AnchorQuality:
        samples = pcm_to_float32(pcm_data)
        if samples.size == 0:
            return AnchorQuality(False, "too_short", 0.0, 0.0)

        duration_sec = samples.size / self.sample_rate
        if duration_sec < MIN_ANCHOR_SECONDS:
            return AnchorQuality(False, "too_short", duration_sec, 0.0)

        rms = float(np.sqrt(np.mean(samples**2)))
        snr_db = _estimate_snr_db(samples)
        if rms < MIN_SIGNAL_RMS or snr_db < MIN_SNR_DB:
            return AnchorQuality(False, "low_snr", duration_sec, snr_db)

        return AnchorQuality(True, "ok", duration_sec, snr_db)

    def extract_embedding_from_pcm(self, pcm_data: bytes) -> Optional[np.ndarray]:
        samples = pcm_to_float32(pcm_data)
        return self.extract_embedding_from_waveform(samples)

    def extract_embedding_from_waveform(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        if waveform.size < int(self.sample_rate * 0.25):
            return None

        if self._ecapa_ready and self._ecapa is not None:
            try:
                with torch.no_grad():
                    wav = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
                    emb = self._ecapa.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
                    return _l2_normalize(emb)
            except Exception as e:
                logger.warning(f"ECAPA 推理失败，回退频谱 embedding：{e}")

        return self._extract_spectral_embedding(waveform)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _extract_spectral_embedding(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        if waveform.size < 320:
            return None
        with torch.no_grad():
            wav = torch.from_numpy(waveform.astype(np.float32))
            win = torch.hann_window(400)
            spec = torch.stft(
                wav,
                n_fft=512,
                hop_length=160,
                win_length=400,
                window=win,
                return_complex=True,
            )
            mag = torch.log1p(torch.abs(spec))
            mean = mag.mean(dim=1)
            std = mag.std(dim=1)
            emb = torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)
        return _l2_normalize(emb)


def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    if not pcm_data:
        return np.array([], dtype=np.float32)
    i16 = np.frombuffer(pcm_data, dtype="<i2")
    if i16.size == 0:
        return np.array([], dtype=np.float32)
    return (i16.astype(np.float32) / 32768.0).copy()


def build_speaker_waveform_map(
    pcm_data: bytes,
    phrases: list,
    sample_rate: int = SAMPLE_RATE,
    min_phrase_ms: int = 120,
) -> dict[str, np.ndarray]:
    """
    按 phrase 时间戳裁剪音频并聚合同一 speaker 的波形。
    """
    samples = pcm_to_float32(pcm_data)
    total = samples.size
    if total == 0:
        return {}

    min_len = int(sample_rate * (min_phrase_ms / 1000))
    per_spk: dict[str, list[np.ndarray]] = {}

    for p in phrases:
        spk = getattr(p, "speaker", None)
        if spk is None:
            continue
        offset_ms = int(getattr(p, "offset_ms", 0) or 0)
        duration_ms = int(getattr(p, "duration_ms", 0) or 0)
        if duration_ms <= 0:
            continue

        start = int(max(0, offset_ms) * sample_rate / 1000)
        end = int(min(total, start + duration_ms * sample_rate / 1000))
        if end - start < min_len:
            continue

        spk_key = str(spk)
        per_spk.setdefault(spk_key, []).append(samples[start:end])

    merged: dict[str, np.ndarray] = {}
    for spk, chunks in per_spk.items():
        if not chunks:
            continue
        merged_wave = np.concatenate(chunks)
        if merged_wave.size >= int(sample_rate * 0.3):
            merged[spk] = merged_wave
    return merged


def _estimate_snr_db(samples: np.ndarray) -> float:
    if samples.size < 320:
        return 0.0
    frame_size = 320  # 20ms @16kHz
    n = samples.size // frame_size
    if n <= 1:
        return 0.0
    frames = samples[: n * frame_size].reshape(n, frame_size)
    power = np.mean(frames**2, axis=1) + 1e-9
    speech = float(np.percentile(power, 90))
    noise = float(np.percentile(power, 20))
    return 10.0 * math.log10(speech / (noise + 1e-9))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x) + 1e-9
    return (x / denom).astype(np.float32)

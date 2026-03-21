from __future__ import annotations

import numpy as np
import pytest

import detection.session_analyzer as sa
from detection.session_analyzer import (
    SessionState,
    _backfill_recent_unknown_roles,
    _resolve_roles_with_anchor,
)
from detection.voice_anchor import VoiceEmbedder, build_speaker_waveform_map


@pytest.fixture(autouse=True)
def use_spectral_embedder():
    """
    强制全局 embedder 使用频谱模式（不依赖 ECAPA 模型），保证测试确定性。
    锚点和 chunk embedding 均使用同一后端，维度一致。
    """
    original = sa._voice_embedder
    sa._voice_embedder = VoiceEmbedder(enable_ecapa=False)
    yield
    sa._voice_embedder = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def _new_session(anchor_embedding: list[float] | None = None) -> SessionState:
    sess = SessionState(
        session_id="sess_test",
        language="ja-JP",
        max_speakers=2,
        turns=[],
        agent_speaker_id=None,
    )
    if anchor_embedding is not None:
        sess.agent_anchor_embedding = anchor_embedding
        sess.anchor_ready = True
    return sess


def _make_tone_pcm(freq: float, duration: float = 3.0, amp: float = 0.3, sr: int = 16000) -> bytes:
    """固定频率正弦波 + 静音前缀，用于模拟人声（通过质量检查）。"""
    silence = np.zeros(sr, dtype=np.float32)
    tone = amp * np.sin(2 * np.pi * freq * np.linspace(0, duration, int(sr * duration), endpoint=False))
    wave = np.concatenate([silence, tone.astype(np.float32)])
    return _to_pcm16_bytes(wave)


# ---------------------------------------------------------------------------
# 声纹质量门限
# ---------------------------------------------------------------------------

def test_anchor_quality_thresholds() -> None:
    embedder = VoiceEmbedder(enable_ecapa=False)
    sr = 16000

    # 时长不足
    short = 0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, int(sr), endpoint=False))
    q_short = embedder.check_quality(_to_pcm16_bytes(short))
    assert not q_short.ok
    assert q_short.reason == "too_short"

    # SNR 过低（纯噪声）
    noisy = np.random.normal(0.0, 0.02, int(sr * 4.0)).astype(np.float32)
    q_noisy = embedder.check_quality(_to_pcm16_bytes(noisy))
    assert not q_noisy.ok
    assert q_noisy.reason == "low_snr"

    # 正常信号
    tone = 0.2 * np.sin(2 * np.pi * 180 * np.linspace(0, 2.0, int(sr * 2.0), endpoint=False))
    silence = np.zeros(int(sr), dtype=np.float32)
    clean = np.concatenate([silence, tone.astype(np.float32), silence]).astype(np.float32)
    q_clean = embedder.check_quality(_to_pcm16_bytes(clean))
    assert q_clean.ok
    assert q_clean.reason == "ok"


# ---------------------------------------------------------------------------
# 两人场景：相对比较（核心逻辑）
# ---------------------------------------------------------------------------

def test_two_speakers_relative_comparison() -> None:
    """
    锚点由 Speaker 0 建立，Speaker 0 的 embedding 与锚点相似度应高于 Speaker 1。
    期望：Speaker 0 = agent，Speaker 1 = customer。
    """
    embedder = VoiceEmbedder(enable_ecapa=False)
    sr = 16000

    # 用不同频率模拟两个说话人的声纹差异
    agent_pcm = _make_tone_pcm(freq=180.0, duration=3.0, amp=0.3)
    customer_pcm = _make_tone_pcm(freq=400.0, duration=3.0, amp=0.3)

    anchor_emb = embedder.extract_embedding_from_pcm(agent_pcm)
    assert anchor_emb is not None

    sess = _new_session(anchor_embedding=anchor_emb.tolist())

    # 构造含两个说话人的 chunk PCM（先 agent 后 customer，各 3 秒）
    agent_wave = np.frombuffer(agent_pcm, dtype="<i2").astype(np.float32) / 32768.0
    customer_wave = np.frombuffer(customer_pcm, dtype="<i2").astype(np.float32) / 32768.0
    chunk_wave = np.concatenate([agent_wave, customer_wave]).astype(np.float32)
    chunk_pcm = _to_pcm16_bytes(chunk_wave)

    # 构造 phrases：Speaker "0" 占前半段，Speaker "1" 占后半段
    agent_dur_ms = int(len(agent_wave) / sr * 1000)
    customer_dur_ms = int(len(customer_wave) / sr * 1000)

    class FakePhrase:
        def __init__(self, speaker, offset_ms, duration_ms, text="x"):
            self.speaker = speaker
            self.offset_ms = offset_ms
            self.duration_ms = duration_ms
            self.text = text

    phrases = [
        FakePhrase("0", 0, agent_dur_ms),
        FakePhrase("1", agent_dur_ms, customer_dur_ms),
    ]

    role_map = _resolve_roles_with_anchor(sess, chunk_pcm, phrases)

    assert role_map.get("0") == "agent", f"Expected agent but got {role_map}"
    assert role_map.get("1") == "customer", f"Expected customer but got {role_map}"
    assert sess.agent_speaker_id == "0"


def test_two_speakers_clear_difference() -> None:
    """
    两人声纹差异明显时，相对比较正确区分角色，不产生 unknown。
    """
    embedder = VoiceEmbedder(enable_ecapa=False)
    sr = 16000

    anchor_pcm = _make_tone_pcm(freq=180.0, duration=3.0)
    anchor_emb = embedder.extract_embedding_from_pcm(anchor_pcm)
    sess = _new_session(anchor_embedding=anchor_emb.tolist())

    # 差异明显的两个频率（锚点 180Hz vs 客户 600Hz）
    pcm_a = _make_tone_pcm(freq=180.0, duration=3.0)
    pcm_b = _make_tone_pcm(freq=600.0, duration=3.0)
    wave_a = np.frombuffer(pcm_a, dtype="<i2").astype(np.float32) / 32768.0
    wave_b = np.frombuffer(pcm_b, dtype="<i2").astype(np.float32) / 32768.0
    chunk_wave = np.concatenate([wave_a, wave_b]).astype(np.float32)
    chunk_pcm = _to_pcm16_bytes(chunk_wave)
    dur_a = int(len(wave_a) / sr * 1000)
    dur_b = int(len(wave_b) / sr * 1000)

    class FakePhrase:
        def __init__(self, speaker, offset_ms, duration_ms, text="x"):
            self.speaker = speaker
            self.offset_ms = offset_ms
            self.duration_ms = duration_ms
            self.text = text

    phrases = [FakePhrase("0", 0, dur_a), FakePhrase("1", dur_a, dur_b)]
    role_map = _resolve_roles_with_anchor(sess, chunk_pcm, phrases)

    # 差异大时：一个 agent 一个 customer，无 unknown
    roles = set(role_map.values())
    assert "agent" in roles
    assert "unknown" not in roles


def test_diarization_error_both_agent() -> None:
    """
    AWS Transcribe 误拆同一人声音为两个 speaker 时，两者都应标为 agent。
    （两个 speaker 分数差 < MIN_SCORE_DIFF → 判为误拆）
    """
    embedder = VoiceEmbedder(enable_ecapa=False)
    sr = 16000

    # 锚点和两段音频使用相同或极接近的频率，模拟同一人被误拆
    anchor_pcm = _make_tone_pcm(freq=200.0, duration=3.0)
    anchor_emb = embedder.extract_embedding_from_pcm(anchor_pcm)
    sess = _new_session(anchor_embedding=anchor_emb.tolist())

    pcm_a = _make_tone_pcm(freq=200.0, duration=3.0)
    pcm_b = _make_tone_pcm(freq=205.0, duration=3.0)  # 极接近，模拟同一人
    wave_a = np.frombuffer(pcm_a, dtype="<i2").astype(np.float32) / 32768.0
    wave_b = np.frombuffer(pcm_b, dtype="<i2").astype(np.float32) / 32768.0
    chunk_wave = np.concatenate([wave_a, wave_b]).astype(np.float32)
    chunk_pcm = _to_pcm16_bytes(chunk_wave)
    dur_a = int(len(wave_a) / sr * 1000)
    dur_b = int(len(wave_b) / sr * 1000)

    class FakePhrase:
        def __init__(self, speaker, offset_ms, duration_ms, text="x"):
            self.speaker = speaker
            self.offset_ms = offset_ms
            self.duration_ms = duration_ms
            self.text = text

    phrases = [FakePhrase("0", 0, dur_a), FakePhrase("1", dur_a, dur_b)]
    role_map = _resolve_roles_with_anchor(sess, chunk_pcm, phrases)

    # 误拆时：两者均为 agent
    assert role_map.get("0") == "agent"
    assert role_map.get("1") == "agent"
    assert "unknown" not in role_map.values()


# ---------------------------------------------------------------------------
# 单说话人：沿用上次已知角色
# ---------------------------------------------------------------------------

def test_single_speaker_inherits_last_role() -> None:
    """
    第一个 chunk 已经确认 Speaker 0 = agent。
    第二个 chunk 只有 Speaker 0 出现且无足够音频提取 embedding，
    应沿用上次已知角色 = agent。
    """
    embedder = VoiceEmbedder(enable_ecapa=False)
    anchor_pcm = _make_tone_pcm(freq=200.0, duration=3.0)
    anchor_emb = embedder.extract_embedding_from_pcm(anchor_pcm)
    sess = _new_session(anchor_embedding=anchor_emb.tolist())
    sess.speaker_last_role["0"] = "agent"
    sess.agent_speaker_id = "0"

    # 极短 chunk，音频不够提取 embedding
    silence_pcm = _to_pcm16_bytes(np.zeros(160, dtype=np.float32))  # 10ms

    class FakePhrase:
        def __init__(self, speaker, offset_ms, duration_ms, text="x"):
            self.speaker = speaker
            self.offset_ms = offset_ms
            self.duration_ms = duration_ms
            self.text = text

    phrases = [FakePhrase("0", 0, 10)]
    role_map = _resolve_roles_with_anchor(sess, silence_pcm, phrases)
    assert role_map.get("0") == "agent"


# ---------------------------------------------------------------------------
# 首说话人约定
# ---------------------------------------------------------------------------

def test_first_speaker_convention() -> None:
    """
    无任何历史记录时，第一个出现的 speaker 应被标记为营业员。
    """
    embedder = VoiceEmbedder(enable_ecapa=False)
    anchor_pcm = _make_tone_pcm(freq=200.0, duration=3.0)
    anchor_emb = embedder.extract_embedding_from_pcm(anchor_pcm)
    sess = _new_session(anchor_embedding=anchor_emb.tolist())
    # 无历史，无 agent_speaker_id

    silence_pcm = _to_pcm16_bytes(np.zeros(160, dtype=np.float32))

    class FakePhrase:
        def __init__(self, speaker, offset_ms, duration_ms, text="x"):
            self.speaker = speaker
            self.offset_ms = offset_ms
            self.duration_ms = duration_ms
            self.text = text

    phrases = [FakePhrase("0", 0, 10)]
    role_map = _resolve_roles_with_anchor(sess, silence_pcm, phrases)
    assert role_map.get("0") == "agent"
    assert sess.agent_speaker_id == "0"


# ---------------------------------------------------------------------------
# unknown 回填（安全网，正常流程极少触发）
# ---------------------------------------------------------------------------

def test_unknown_backfill_with_recent_window() -> None:
    sess = _new_session()
    sess.speaker_last_role["0"] = "agent"

    sess.turns = [
        {"turn": 1, "speaker": "0", "role": "unknown", "text_ja": "a", "text": "a", "offset_ms": 0},
        {"turn": 2, "speaker": "1", "role": "unknown", "text_ja": "b", "text": "b", "offset_ms": 100},
        {"turn": 3, "speaker": "0", "role": "agent",   "text_ja": "c", "text": "c", "offset_ms": 200},
    ]

    # window=2 只覆盖 turn2~3，turn1 不在范围内
    changed = _backfill_recent_unknown_roles(sess, window_size=2)
    assert changed is False

    # window=3 覆盖全部，turn1 应被回填
    changed2 = _backfill_recent_unknown_roles(sess, window_size=3)
    assert changed2 is True
    assert sess.turns[0]["role"] == "agent"
    assert sess.turns[1]["role"] == "unknown"  # speaker "1" 无历史，不回填

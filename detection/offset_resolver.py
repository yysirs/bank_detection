"""
字符偏移计算模块

将 LLM 返回的 fragment 文本，定位到 text_ja 中的精确字符偏移 [start, end)。

两阶段策略：
1. 精确匹配：str.find()，O(n)，最快
2. 模糊匹配：difflib.SequenceMatcher 滑动窗口，处理全半角/标点差异，O(n*m)
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Optional


@dataclass
class OffsetResult:
    fragment: str   # 原始（或修正后）fragment 文本
    start: int      # text_ja 中的起始字符索引（包含）
    end: int        # text_ja 中的结束字符索引（不包含），即 text_ja[start:end] == fragment
    match_type: str  # "exact" | "fuzzy" | "not_found"
    score: float    # 匹配得分（精确=1.0，模糊=相似度，未找到=0.0）


def normalize_ja(text: str) -> str:
    """
    全角/半角归一化，用于提升模糊匹配精度。
    NFKC 将全角字母数字转为半角，并统一标点形式。
    """
    return unicodedata.normalize("NFKC", text)


def resolve_offset(text_ja: str, fragment: str, fuzzy_threshold: float = 0.85) -> OffsetResult:
    """
    计算 fragment 在 text_ja 中的字符偏移。

    Args:
        text_ja: 原始日文句子
        fragment: LLM 返回的违规片段
        fuzzy_threshold: 模糊匹配最低相似度（0~1），低于此值视为未找到

    Returns:
        OffsetResult，match_type 为 "exact"/"fuzzy"/"not_found"
    """
    if not fragment or not text_ja:
        return OffsetResult(fragment=fragment, start=0, end=0, match_type="not_found", score=0.0)

    # ── 阶段1：精确匹配 ──
    idx = text_ja.find(fragment)
    if idx != -1:
        return OffsetResult(
            fragment=fragment,
            start=idx,
            end=idx + len(fragment),
            match_type="exact",
            score=1.0,
        )

    # ── 阶段2：全角/半角归一化后再精确匹配 ──
    norm_text = normalize_ja(text_ja)
    norm_frag = normalize_ja(fragment)
    idx = norm_text.find(norm_frag)
    if idx != -1:
        # 用归一化后的索引对应回原始文本（字符数一致，NFKC 不改变字符个数）
        original_fragment = text_ja[idx : idx + len(fragment)]
        return OffsetResult(
            fragment=original_fragment,
            start=idx,
            end=idx + len(fragment),
            match_type="exact",
            score=1.0,
        )

    # ── 阶段3：滑动窗口模糊匹配 ──
    frag_len = len(fragment)
    best_score = 0.0
    best_idx = -1

    # 允许 fragment 长度在 ±3 个字符范围内浮动
    for window_size in range(max(1, frag_len - 3), frag_len + 4):
        for i in range(len(text_ja) - window_size + 1):
            candidate = text_ja[i : i + window_size]
            score = SequenceMatcher(None, norm_frag, normalize_ja(candidate)).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
                best_window = window_size

    if best_score >= fuzzy_threshold and best_idx != -1:
        matched_fragment = text_ja[best_idx : best_idx + best_window]
        return OffsetResult(
            fragment=matched_fragment,
            start=best_idx,
            end=best_idx + best_window,
            match_type="fuzzy",
            score=best_score,
        )

    # ── 未找到 ──
    return OffsetResult(fragment=fragment, start=0, end=0, match_type="not_found", score=0.0)


def resolve_all_offsets(
    text_ja: str,
    fragments: list[str],
    fuzzy_threshold: float = 0.85,
) -> list[OffsetResult]:
    """
    对一句话中的多个 fragment 批量计算偏移。
    """
    return [resolve_offset(text_ja, frag, fuzzy_threshold) for frag in fragments]


def build_violation_offsets(
    text_ja: str,
    fragment: str,
    fuzzy_threshold: float = 0.85,
) -> Optional[list[dict]]:
    """
    计算单个 fragment 的偏移，返回符合测试数据格式的 violation_offsets 列表。
    若未找到则返回 None。

    返回格式：
        [{"fragment": "...", "start": 0, "end": 10}]
    """
    result = resolve_offset(text_ja, fragment, fuzzy_threshold)
    if result.match_type == "not_found":
        return None
    return [{"fragment": result.fragment, "start": result.start, "end": result.end}]

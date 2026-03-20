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


def _strip_spaces_index_map(text: str) -> tuple[str, list[int]]:
    """
    去除文本中的空格，同时返回「去空格后索引 → 原始索引」的映射表。

    例：text = "山田 様" → stripped = "山田様", map = [0, 1, 3]
    """
    stripped_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch != ' ':
            stripped_chars.append(ch)
            index_map.append(i)
    return "".join(stripped_chars), index_map


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

    # ── 阶段3：去除空格后精确匹配（AWS 转录词间有空格，LLM fragment 无空格）──
    stripped_text, idx_map = _strip_spaces_index_map(text_ja)
    stripped_frag = fragment.replace(' ', '')
    norm_stripped  = normalize_ja(stripped_text)
    norm_sfrag     = normalize_ja(stripped_frag)
    sidx = norm_stripped.find(norm_sfrag)
    if sidx != -1:
        orig_start = idx_map[sidx]
        orig_end   = idx_map[sidx + len(stripped_frag) - 1] + 1
        original_fragment = text_ja[orig_start:orig_end]
        return OffsetResult(
            fragment=original_fragment,
            start=orig_start,
            end=orig_end,
            match_type="exact",
            score=1.0,
        )

    # ── 阶段4：fragment 比 text 长时，找 fragment 与 text 的最长公共子串 ──
    # 场景：LLM 把多个 turn 内容拼成一个 fragment，或 text 被截断
    stripped_frag_n = normalize_ja(fragment.replace(' ', ''))
    stripped_text_n = normalize_ja(stripped_text)
    lcs = _longest_common_substring(stripped_frag_n, stripped_text_n)
    if lcs and len(lcs) >= 4:  # 至少4个字符才有意义
        sidx = stripped_text_n.find(lcs)
        if sidx != -1:
            orig_start = idx_map[sidx]
            orig_end   = idx_map[sidx + len(lcs) - 1] + 1
            original_fragment = text_ja[orig_start:orig_end]
            score = len(lcs) / max(len(stripped_frag_n), len(stripped_text_n))
            if score >= fuzzy_threshold * 0.7:
                return OffsetResult(
                    fragment=original_fragment,
                    start=orig_start,
                    end=orig_end,
                    match_type="fuzzy",
                    score=round(score, 3),
                )


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


def _longest_common_substring(s1: str, s2: str) -> str:
    """返回 s1 和 s2 的最长公共子串。"""
    if not s1 or not s2:
        return ""
    m, n = len(s1), len(s2)
    best_len = 0
    best_end = 0
    # dp[j] = 以 s1[i-1], s2[j-1] 结尾的公共子串长度
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
                if dp[j] > best_len:
                    best_len = dp[j]
                    best_end = j
            else:
                dp[j] = 0
            prev = temp
    return s2[best_end - best_len: best_end]


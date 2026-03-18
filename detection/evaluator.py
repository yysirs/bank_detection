"""
评估工具（Evaluator）

将检测结果与 Ground Truth 对比，计算以下指标：
- 句子级召回率 / 精确率 / F1
- 类型准确率（violation_type + sub_category 完全匹配）
- Fragment 字符级重叠 F1
- 偏移误差（|start_pred - start_gt| 平均字符数）
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class TurnMetrics:
    """单轮次评估结果"""
    turn: int
    gt_is_violation: bool
    pred_is_violation: bool
    # 类型匹配：(gt_sub_category, pred_sub_category, is_match)
    type_matches: list[tuple[str, str, bool]] = field(default_factory=list)
    # Fragment 重叠 F1（对每对 gt/pred fragment）
    fragment_f1_scores: list[float] = field(default_factory=list)
    # 偏移误差（字符数）
    offset_errors: list[int] = field(default_factory=list)


@dataclass
class SessionMetrics:
    """会话级汇总指标"""
    session_id: str

    # 句子级
    turn_tp: int = 0   # 真正例：GT 违规 & 预测违规
    turn_fp: int = 0   # 假正例：GT 合规 & 预测违规
    turn_fn: int = 0   # 假负例：GT 违规 & 预测合规
    turn_tn: int = 0   # 真负例：GT 合规 & 预测合规

    # 类型匹配
    type_correct: int = 0
    type_total_gt: int = 0

    # Fragment F1
    fragment_f1_sum: float = 0.0
    fragment_f1_count: int = 0

    # 偏移误差
    offset_error_sum: int = 0
    offset_error_count: int = 0

    @property
    def turn_precision(self) -> float:
        denom = self.turn_tp + self.turn_fp
        return self.turn_tp / denom if denom > 0 else 0.0

    @property
    def turn_recall(self) -> float:
        denom = self.turn_tp + self.turn_fn
        return self.turn_tp / denom if denom > 0 else 0.0

    @property
    def turn_f1(self) -> float:
        p, r = self.turn_precision, self.turn_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / self.type_total_gt if self.type_total_gt > 0 else 0.0

    @property
    def fragment_f1_avg(self) -> float:
        return (
            self.fragment_f1_sum / self.fragment_f1_count
            if self.fragment_f1_count > 0
            else 0.0
        )

    @property
    def offset_error_avg(self) -> float:
        return (
            self.offset_error_sum / self.offset_error_count
            if self.offset_error_count > 0
            else 0.0
        )


def _char_f1(s1: str, s2: str) -> float:
    """计算两个字符串的字符级 F1（基于最长公共子序列）"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    matcher = SequenceMatcher(None, s1, s2)
    matching_chars = sum(triple.size for triple in matcher.get_matching_blocks())
    precision = matching_chars / len(s2)
    recall = matching_chars / len(s1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_fragment_match(
    gt_fragment: str,
    pred_fragments: list[str],
) -> tuple[float, int]:
    """
    找出与 gt_fragment 最匹配的 pred_fragment。
    返回 (最高 F1, 最佳 pred 索引)。
    """
    if not pred_fragments:
        return 0.0, -1
    scores = [_char_f1(gt_fragment, pf) for pf in pred_fragments]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return scores[best_idx], best_idx


def evaluate_session(gt_session: dict, pred_session: dict) -> SessionMetrics:
    """
    对单个会话进行评估。

    Args:
        gt_session:   Ground Truth 会话 JSON（测试数据原始格式）
        pred_session: 预测结果 JSON（与 GT 格式相同）

    Returns:
        SessionMetrics
    """
    session_id = gt_session.get("session_id", "unknown")
    metrics = SessionMetrics(session_id=session_id)

    gt_turns = {t["turn"]: t for t in gt_session.get("dialogue", [])}
    pred_turns = {t["turn"]: t for t in pred_session.get("dialogue", [])}

    for turn_id, gt_turn in gt_turns.items():
        if gt_turn.get("role") != "agent":
            continue

        pred_turn = pred_turns.get(turn_id, {})
        gt_violations = gt_turn.get("violations", [])
        pred_violations = pred_turn.get("violations", [])

        gt_is_viol = len(gt_violations) > 0
        pred_is_viol = len(pred_violations) > 0

        # ── 句子级 TP/FP/FN/TN ──
        if gt_is_viol and pred_is_viol:
            metrics.turn_tp += 1
        elif not gt_is_viol and pred_is_viol:
            metrics.turn_fp += 1
        elif gt_is_viol and not pred_is_viol:
            metrics.turn_fn += 1
        else:
            metrics.turn_tn += 1

        if not gt_violations:
            continue

        # ── 类型匹配 & Fragment 评估 ──
        # 收集预测的 sub_category 列表 & fragment 列表
        pred_sub_cats = [v.get("sub_category", "") for v in pred_violations]
        pred_fragments_all: list[str] = []
        pred_starts: list[int] = []
        for pv in pred_violations:
            for offset in pv.get("violation_offsets", []):
                pred_fragments_all.append(offset.get("fragment", ""))
                pred_starts.append(offset.get("start", -1))

        for gt_v in gt_violations:
            gt_sub_cat = gt_v.get("sub_category", "")
            metrics.type_total_gt += 1

            # 类型是否命中
            if gt_sub_cat in pred_sub_cats:
                metrics.type_correct += 1

            # Fragment F1
            for gt_offset in gt_v.get("violation_offsets", []):
                gt_frag = gt_offset.get("fragment", "")
                gt_start = gt_offset.get("start", -1)

                if pred_fragments_all:
                    f1, best_idx = _best_fragment_match(gt_frag, pred_fragments_all)
                    metrics.fragment_f1_sum += f1
                    metrics.fragment_f1_count += 1

                    # 偏移误差（只在有匹配时计算）
                    if best_idx >= 0 and gt_start >= 0 and pred_starts[best_idx] >= 0:
                        error = abs(gt_start - pred_starts[best_idx])
                        metrics.offset_error_sum += error
                        metrics.offset_error_count += 1
                else:
                    metrics.fragment_f1_sum += 0.0
                    metrics.fragment_f1_count += 1

    return metrics


def evaluate_all(
    gt_sessions: list[dict],
    pred_sessions: list[dict],
) -> dict:
    """
    对多个会话批量评估，返回汇总指标。

    Args:
        gt_sessions:   GT 会话列表
        pred_sessions: 预测结果列表（与 GT 一一对应）

    Returns:
        包含各项指标的 dict
    """
    # 建立 session_id → pred 映射
    pred_map = {s.get("session_id", ""): s for s in pred_sessions}

    all_metrics: list[SessionMetrics] = []
    for gt in gt_sessions:
        sid = gt.get("session_id", "")
        pred = pred_map.get(sid)
        if pred is None:
            print(f"警告：找不到 session_id={sid} 的预测结果，跳过")
            continue
        all_metrics.append(evaluate_session(gt, pred))

    # 汇总
    total_tp = sum(m.turn_tp for m in all_metrics)
    total_fp = sum(m.turn_fp for m in all_metrics)
    total_fn = sum(m.turn_fn for m in all_metrics)
    total_tn = sum(m.turn_tn for m in all_metrics)

    total_type_correct = sum(m.type_correct for m in all_metrics)
    total_type_gt = sum(m.type_total_gt for m in all_metrics)

    frag_f1_scores = [
        m.fragment_f1_sum / m.fragment_f1_count
        for m in all_metrics
        if m.fragment_f1_count > 0
    ]
    offset_errors = [
        m.offset_error_sum / m.offset_error_count
        for m in all_metrics
        if m.offset_error_count > 0
    ]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "sessions_evaluated": len(all_metrics),
        "sentence_level": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "type_accuracy": round(
            total_type_correct / total_type_gt if total_type_gt > 0 else 0.0, 4
        ),
        "fragment_f1_avg": round(
            sum(frag_f1_scores) / len(frag_f1_scores) if frag_f1_scores else 0.0, 4
        ),
        "offset_error_avg_chars": round(
            sum(offset_errors) / len(offset_errors) if offset_errors else 0.0, 2
        ),
        "per_session": [
            {
                "session_id": m.session_id,
                "turn_precision": round(m.turn_precision, 4),
                "turn_recall": round(m.turn_recall, 4),
                "turn_f1": round(m.turn_f1, 4),
                "type_accuracy": round(m.type_accuracy, 4),
                "fragment_f1_avg": round(m.fragment_f1_avg, 4),
                "offset_error_avg": round(m.offset_error_avg, 2),
            }
            for m in all_metrics
        ],
    }


def print_report(metrics: dict) -> None:
    """打印格式化评估报告"""
    print("\n" + "=" * 60)
    print("       合规检测算法评估报告")
    print("=" * 60)
    print(f"评估会话数：{metrics['sessions_evaluated']}")

    sl = metrics["sentence_level"]
    print(f"\n【句子级检测】")
    print(f"  TP={sl['TP']}  FP={sl['FP']}  FN={sl['FN']}  TN={sl['TN']}")
    print(f"  精确率 Precision : {sl['precision']:.2%}")
    print(f"  召回率 Recall    : {sl['recall']:.2%}")
    print(f"  F1               : {sl['f1']:.2%}")

    print(f"\n【违规类型准确率】: {metrics['type_accuracy']:.2%}")
    print(f"【Fragment 字符 F1】: {metrics['fragment_f1_avg']:.4f}")
    print(f"【偏移误差（平均字符数）】: {metrics['offset_error_avg_chars']:.1f} 字符")

    print(f"\n【逐会话明细】")
    print(f"{'Session ID':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TypeAcc':>8} {'FragF1':>7} {'OffErr':>7}")
    print("-" * 70)
    for s in metrics["per_session"]:
        print(
            f"{s['session_id']:<25} "
            f"{s['turn_precision']:>6.2%} "
            f"{s['turn_recall']:>6.2%} "
            f"{s['turn_f1']:>6.2%} "
            f"{s['type_accuracy']:>8.2%} "
            f"{s['fragment_f1_avg']:>7.4f} "
            f"{s['offset_error_avg']:>7.1f}"
        )
    print("=" * 60)

"""
违规分类体系（Compliance Violation Taxonomy）

从 10 组测试对话数据（PF_NG_FUND_01~10）归纳。
GT 字段为 Ground Truth 中对应的英文 sub_category 值，用于评估时对比。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SubCategory:
    name_zh: str       # 中文标签
    name_gt: str       # Ground Truth 英文值（测试数据中的 sub_category）
    description: str   # 简要定义
    examples: list[str]  # 典型日文例句（用于 Prompt 中的 few-shot）


@dataclass(frozen=True)
class ViolationType:
    type_id: str         # "Type 1" ~ "Type 6"
    name_zh: str         # 中文类型名
    description: str
    sub_categories: list[SubCategory]


# ─────────────────────────────────────────────
# Type 1 — 虚假收益/安全承诺
# ─────────────────────────────────────────────
TYPE_1 = ViolationType(
    type_id="Type 1",
    name_zh="虚假收益/安全承诺",
    description="营业员使用确定性语言承诺未来收益、本金安全或否定投资风险",
    sub_categories=[
        SubCategory(
            name_zh="未来确定性断言",
            name_gt="Certainty_Future",
            description="用「必ず」「確実に」「間違いなく」等断言未来收益必然发生",
            examples=[
                "これからも同じペースで必ず利益が出ることを保証できます",
                "今回も間違いなく同じように儲かりますよ",
                "将来は間違いなく儲かります",
            ],
        ),
        SubCategory(
            name_zh="本金保证",
            name_gt="Guarantee_Principal",
            description="声称本金不会减少或有保障，但投资产品不属于存款保险范围",
            examples=[
                "元本も保証されてますよ",
                "元本は減らないので安心です",
                "元本もほぼ保証されますから",
            ],
        ),
        SubCategory(
            name_zh="收益率保证",
            name_gt="Guarantee_Return",
            description="保证具体收益率或分配金金额，使用「確実」「必ず」等",
            examples=[
                "年率8%以上は確実ですよ",
                "来年も最低20%は確実に儲かりますから",
                "将来も必ず安定した収益が続きますから",
            ],
        ),
        SubCategory(
            name_zh="风险否定",
            name_gt="Risk_Nullification",
            description="直接否定投资风险，如「リスクゼロ」「下がらない」",
            examples=[
                "リスクゼロです",
                "損することはありません",
                "今の相場は下がらないから安心です",
            ],
        ),
        SubCategory(
            name_zh="绝对安全表述",
            name_gt="Absolute_Safety",
            description="使用「絶対安全」「これからも安全に増やせます」等绝对化安全表述",
            examples=[
                "絶対安全、元本も保証されるから",
                "これからも安全に増やせます",
                "絶対安全です",
            ],
        ),
        SubCategory(
            name_zh="收益夸大",
            name_gt="Exaggerated_Profit",
            description="用夸大的语言描述收益增长，如「複利効果が倍増」「ぐんぐん増える」",
            examples=[
                "回転売買をすれば複利効果が倍増します",
                "これで資産がぐんぐん増えますよ",
                "年率5％の利息に近い分配金が毎月入ります",
            ],
        ),
    ],
)

# ─────────────────────────────────────────────
# Type 2 — 虚假信息/机制误述
# ─────────────────────────────────────────────
TYPE_2 = ViolationType(
    type_id="Type 2",
    name_zh="虚假信息/机制误述",
    description="营业员对产品机制、历史数据或市场信息进行虚假或误导性描述",
    sub_categories=[
        SubCategory(
            name_zh="产品机制误述",
            name_gt="Misstate_Mechanism",
            description="将基金分配金误述为「利息」，混淆产品性质，使客户误解本金安全性",
            examples=[
                "分配金は利息と同じ仕組みで支払われます",
                "分配金は利息なので元本は保証",
                "利息として毎月5000円が入ってきます",
            ],
        ),
        SubCategory(
            name_zh="伪造内幕信息",
            name_gt="Fake_Internal_Info",
            description="捏造内幕消息或未公开信息来影响客户决策",
            examples=[
                "ここだけの話ですが、今週中に大手IT企業がこのファンドの主要保有株を買収すると噂が入ってます",
                "私の知り合いがそのIT企業のIR部にいるんです。絶対に内情です",
            ],
        ),
        SubCategory(
            name_zh="虚假产品特性",
            name_gt="False_Product_Feature",
            description="对产品特性进行虚假描述，或以虚假信息贬低竞争产品",
            examples=[
                "A社のファンドは分配金下げ続けているし、運用能力も低いですよ",
                "市場が悪くても分配金は守られます",
            ],
        ),
        SubCategory(
            name_zh="虚假历史业绩",
            name_gt="False_History",
            description="捏造或歪曲过去业绩数据",
            examples=[
                "去年の同ファンドは月2回売買で平均8%増えてます",
                "昨年加入した方は毎月2万円ずつ分配金（利息）をもらえてますから",
            ],
        ),
    ],
)

# ─────────────────────────────────────────────
# Type 3 — 不当交易行为
# ─────────────────────────────────────────────
TYPE_3 = ViolationType(
    type_id="Type 3",
    name_zh="不当交易行为",
    description="营业员推荐不适合客户的交易行为，或无视客户适宜性进行销售",
    sub_categories=[
        SubCategory(
            name_zh="频繁交易劝诱",
            name_gt="Churning",
            description="以获利为由劝诱客户频繁买卖（回転売買），实为增加手续费收入",
            examples=[
                "売って買って繰り返す回転売買が一番効率的ですよ",
                "今月中に3回売って買って、また売ればOK！",
                "月に2回売買するだけで安定して利益が出ます",
            ],
        ),
        SubCategory(
            name_zh="不适配销售",
            name_gt="Unsuitable_Sales",
            description="无视客户年龄、收入、风险承受度等属性，推荐不适合的高风险产品",
            examples=[
                "このファンドは誰でも大丈夫",
                "山田様のような方に最適ですよ",
                "高齢者こそ短期で稼ぐチャンスです",
            ],
        ),
    ],
)

# ─────────────────────────────────────────────
# Type 4 — 信息隐瞒/不披露
# ─────────────────────────────────────────────
TYPE_4 = ViolationType(
    type_id="Type 4",
    name_zh="信息隐瞒/不披露",
    description="营业员主动隐瞒、省略或拒绝提供客户应知悉的重要信息",
    sub_categories=[
        SubCategory(
            name_zh="风险未披露",
            name_gt="Non_Disclosure_Risk",
            description="主动拒绝/省略风险说明，或整个对话中介绍高风险产品后从未提及本金损失可能",
            examples=[
                "リスクの説明は長いから省きますね",
                # 缺失型：整场对话无风险说明（无明确例句，在批量模式下通过上下文检测）
            ],
        ),
        SubCategory(
            name_zh="合同条款回避",
            name_gt="Skip_Terms",
            description="客户要求查看合同时，营业员推迟或拒绝",
            examples=[
                "それは後でいいですよ",
            ],
        ),
        SubCategory(
            name_zh="费用隐瞒",
            name_gt="Hide_Fee",
            description="虚假否认或隐瞒实际存在的手续费",
            examples=[
                "ないですよ、全部含まれてます",
            ],
        ),
    ],
)

# ─────────────────────────────────────────────
# Type 5 — 强迫签约
# ─────────────────────────────────────────────
TYPE_5 = ViolationType(
    type_id="Type 5",
    name_zh="强迫签约",
    description="在客户理解产品前或未充分说明的情况下催促签字",
    sub_categories=[
        SubCategory(
            name_zh="强迫签约",
            name_gt="Pressure_Sign",
            description="使用「今サインしておけば」等语言催促客户立即签字",
            examples=[
                "今サインしておけばすぐ始められますよ！",
                "早くサインして始めましょう！",
            ],
        ),
    ],
)

# ─────────────────────────────────────────────
# Type 6 — 心理施压
# ─────────────────────────────────────────────
TYPE_6 = ViolationType(
    type_id="Type 6",
    name_zh="心理施压",
    description="通过虚假紧迫感、限额、截止时间或个人业绩压力等手段强迫客户快速决策",
    sub_categories=[
        SubCategory(
            name_zh="业绩压力施压",
            name_gt="Pressure_Tactic",
            description="将个人业绩压力转嫁给客户，制造客户负罪感",
            examples=[
                "今月ノルマが厳しいんで、このチャンスを逃したら後悔しますよ",
                "今日中に上司から1件取れってプレッシャーかかってて",
            ],
        ),
        SubCategory(
            name_zh="虚构限额紧张",
            name_gt="Fabricated_Scarcity",
            description="捏造产品名额有限，制造稀缺感",
            examples=[
                "今月中に加入なら優先枠があるんですが、残り少ないですよ",
                "今なら特別枠が残ってます！今週末までに申し込めば優先的に入れますよ",
            ],
        ),
        SubCategory(
            name_zh="虚假截止时间",
            name_gt="Fake_Deadline",
            description="捏造申购截止日期，迫使客户仓促决策",
            examples=[
                "今週中に募集終了する特別枠があるんです。今日決めないとチャンスなくなっちゃいます！",
            ],
        ),
        SubCategory(
            name_zh="虚假紧迫感",
            name_gt="False_Urgency",
            description="利用客户的情绪状态或外部因素制造虚假紧迫感",
            examples=[
                "急いでるんでしょ？今サインしておけばすぐ始められますよ！",
            ],
        ),
    ],
)


# ─────────────────────────────────────────────
# 全局分类体系
# ─────────────────────────────────────────────
ALL_TYPES: list[ViolationType] = [TYPE_1, TYPE_2, TYPE_3, TYPE_4, TYPE_5, TYPE_6]

# GT 英文名 → (ViolationType, SubCategory) 映射，用于评估时对比
GT_NAME_MAP: dict[str, tuple[ViolationType, SubCategory]] = {
    sub.name_gt: (vtype, sub)
    for vtype in ALL_TYPES
    for sub in vtype.sub_categories
}


def get_by_gt_name(gt_name: str) -> Optional[tuple[ViolationType, SubCategory]]:
    """通过 GT 英文 sub_category 名查找对应分类"""
    return GT_NAME_MAP.get(gt_name)


def build_taxonomy_prompt_section() -> str:
    """生成用于 Prompt 的违规分类说明文本"""
    lines = ["## 违规分类体系\n"]
    for vtype in ALL_TYPES:
        lines.append(f"### {vtype.type_id} — {vtype.name_zh}")
        lines.append(f"{vtype.description}\n")
        for sub in vtype.sub_categories:
            lines.append(f"**{sub.name_zh}**（{sub.name_gt}）")
            lines.append(f"定义：{sub.description}")
            lines.append("典型例句：")
            for ex in sub.examples:
                lines.append(f"  - 「{ex}」")
            lines.append("")
    return "\n".join(lines)

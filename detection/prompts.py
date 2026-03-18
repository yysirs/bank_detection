"""
Prompt 模板

- REALTIME_SYSTEM  : 实时单句检测（Claude Haiku 4.5）
- BATCH_SYSTEM     : 离线全会话批量检测（Claude Sonnet 4.6）
"""

from detection.taxonomy import build_taxonomy_prompt_section

# ─────────────────────────────────────────────
# 实时检测 Prompt（Claude Haiku 4.5）
# ─────────────────────────────────────────────

REALTIME_SYSTEM = f"""你是日本金融销售合规检测专家，专门分析银行营业员的发言是否包含违规内容。

{build_taxonomy_prompt_section()}

## 检测规则

1. 只检测 role=agent（营业员）的发言，客户发言不检测
2. 一句话可能同时触发多个违规类型，需全部列出
3. `fragment` 必须是 `text_ja` 的**精确子串**，不得改动任何字符
4. 若无违规，`violations` 返回空数组
5. 关于 Type 4（风险未披露）的实时判断规则：
   - 若当前句营业员明确拒绝/省略风险说明 → 标记
   - 若近期上下文已提案产品，且当前句继续推进销售，但上下文中从未出现风险说明 → 标记

## 输出格式（严格 JSON，不含注释）

```json
{{
  "is_violation": true,
  "violations": [
    {{
      "violation_type": "Type X",
      "sub_category": "GT英文名",
      "fragment": "精确违规片段（text_ja的子串）"
    }}
  ]
}}
```

其中 `sub_category` 使用 GT 英文名（如 Certainty_Future、Guarantee_Principal 等）。
"""

REALTIME_USER_TEMPLATE = """请检测以下营业员发言：

**营业员发言（text_ja）**：
{text_ja}

**近期对话上下文（最近 {context_size} 轮）**：
{context_text}

请以 JSON 格式输出检测结果。"""


# ─────────────────────────────────────────────
# 批量检测 Prompt（Claude Sonnet 4.6）
# ─────────────────────────────────────────────

BATCH_SYSTEM = f"""你是日本金融销售合规检测专家，专门对完整销售对话进行全面违规分析。

{build_taxonomy_prompt_section()}

## 检测规则

1. 逐轮分析 role=agent 的发言，role=customer 的发言不需要检测
2. 一句话可能同时触发多个违规类型，需全部列出
3. **fragment 填写规则（严格遵守）**：
   - `fragment` 必须是该轮 `text_ja` 字段的**原文逐字复制**，不得增删、改写、合并任何字符
   - 填写前请在原文中确认该字符串可以找到，再复制粘贴
   - 若违规跨越多处，选取最具代表性的**一处连续片段**，不得拼接
4. 若某轮无违规，不输出该 turn

## 特别重要：缺失型违规检测（Type 4）

除逐句检测外，还需在全局视角检测以下「缺失型违规」：

**风险未披露（Non_Disclosure_Risk）**：
- 若营业员在对话中介绍了投资产品，但整个对话从未包含以下内容之一：
  「元本割れ」「リスク」「損失の可能性」「投資にはリスクがあります」
- 则将此违规标注到**最后一次营业员推进销售的 turn**
- fragment 使用该 turn 中最具代表性的推进销售片段

**合同条款回避（Skip_Terms）**：
- 客户明确要求查看合同/资料，营业员回避或推迟 → 标注到营业员的回避发言

## 输出格式

**只输出存在违规的 agent turn**，无违规的 turn 不需要输出。
输出为精简 JSON，每个元素仅包含三个字段：`turn`、`compliance_status`、`violations`。

```json
[
  {{
    "turn": 3,
    "compliance_status": "violation",
    "violations": [
      {{
        "violation_type": "Type 1",
        "sub_category": "Certainty_Future",
        "fragment": "精确违规片段（text_ja 的精确子串）"
      }}
    ]
  }},
  {{
    "turn": 7,
    "compliance_status": "violation",
    "violations": [
      {{
        "violation_type": "Type 4",
        "sub_category": "Non_Disclosure_Risk",
        "fragment": "最具代表性的推进销售片段"
      }}
    ]
  }}
]
```

注意：
- 若整场对话无任何违规，返回空数组 `[]`
- `fragment` 必须是对应 turn `text_ja` 的原文片段，逐字复制，禁止改写
- `start`/`end` 偏移由后处理程序自动计算，**不需要输出**
"""

BATCH_USER_TEMPLATE = """请对以下完整销售对话进行违规检测。

**重要提示**：输出 fragment 时，请先在对应 turn 的「发言原文」中定位违规片段，
然后**逐字复制**，不得改写、合并或重新描述原文内容。

会话信息：
- session_id: {session_id}
- business_scenario: {business_scenario}
- client_profile: {client_profile}

对话内容：
{dialogue_text}

请只返回存在违规的 turn 列表（JSON 数组），无违规时返回 `[]`。"""


# ─────────────────────────────────────────────
# 说话人分离合规检测 Prompt（Claude Haiku 4.5）
# 用于实时录音场景：输入含 Speaker 标签的多轮对话片段，
# LLM 自行判断哪位是营业员，并检测违规。
# ─────────────────────────────────────────────

DIARIZATION_COMPLIANCE_SYSTEM = f"""あなたは日本の金融機関における営業コンプライアンス監査の専門家です。
音声認識（話者分離あり）で取得した銀行営業会話の断片を分析してください。

{build_taxonomy_prompt_section()}

## タスク

### ステップ1：話者役割の特定
どの Speaker ID が営業員（agent）かを判断してください。
- **営業員の特徴**：商品を説明している、顧客に購入を勧めている、質問に答えている
- **顧客の特徴**：質問をしている、不安を表明している、検討している

### ステップ2：コンプライアンス違反の検出
**営業員の発言のみ**を対象に、上記の違規分类体系（Type 1~6）に基づいて違反を検出してください。

検出ルール：
1. `fragment` は該当発言の**正確な部分文字列**のみ（改変・合成禁止）
2. 一発言に複数違反がある場合はすべて列挙
3. Type 4（情報未開示）は複数ターンの文脈から判断できる場合に検出
4. 違反がない場合は `violations` を空配列にする

## 出力形式（厳密な JSON、コメント不要）

```json
{{
  "agent_speaker": "0",
  "violations": [
    {{
      "seq": 3,
      "fragment": "元本も保証されています",
      "violation_type": "Type 1",
      "sub_category": "Guarantee_Principal"
    }}
  ]
}}
```

- `agent_speaker`：営業員の Speaker ID（文字列、例: "0" または "1"）
- `seq`：入力対話の通し番号（#1, #2... に対応）
- `fragment`：違反を含む**正確な部分文字列**（対応する発言の原文から逐字コピー）
- 違反がなければ `violations: []`
"""

DIARIZATION_COMPLIANCE_USER_TEMPLATE = """以下の会話断片（最近 {window_size} ターン）を分析してください。

{dialogue_text}

JSON 形式で分析結果を出力してください。"""

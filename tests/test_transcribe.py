"""
AWS Transcribe 连通性 + 流式转录测试

分两层验证：
    层1 — 凭证/权限验证：调用 list_transcription_jobs 确认账号有 Transcribe 权限
    层2 — 流式转录测试：将 WAV 文件流式发送，验证实时转录 + 说话人区分（ShowSpeakerLabel）

依赖安装：
    pip3 install amazon-transcribe

用法：
    # 仅验证连通性（不需要音频文件）
    python tests/test_transcribe.py --region ap-northeast-1

    # 完整流式转录测试（需要 WAV 文件，16kHz 单声道）
    python tests/test_transcribe.py --audio /path/to/sample.wav

    # 指定凭证 CSV
    python tests/test_transcribe.py --csv /path/to/accessKeys.csv --audio sample.wav

音频格式要求：
    - 格式：WAV 或 PCM
    - 采样率：16000 Hz（推荐）或 8000 Hz
    - 声道：单声道（mono）
    - 说话人区分建议：至少 30 秒、含双人对话的音频
"""

from __future__ import annotations
import os
import argparse
import asyncio
import logging
import struct
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.aws_client import load_credentials_from_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_transcribe")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CSV = PROJECT_ROOT / "poc_dev_accessKeys20260310.csv"

# 流式转录参数（语言通过 --lang 覆盖）
LANGUAGE_CODE  = "zh-CN"   # 默认中文，可改为 ja-JP / en-US 等
SAMPLE_RATE    = 16000      # Hz
MEDIA_ENCODING = "pcm"      # WAV PCM
CHUNK_SIZE     = 1024 * 8   # 8KB / chunk

# 常用语言代码参考
# zh-CN  简体中文
# ja-JP  日语
# en-US  英语（美国）


# ─────────────────────────────────────────────
# 层1：凭证 + 权限验证（boto3 同步）
# ─────────────────────────────────────────────

def test_credentials(access_key_id: str, secret_access_key: str, region: str) -> bool:
    """验证凭证有效且具备 Transcribe 权限。"""
    logger.info(f"[层1] 验证凭证，region={region}")
    try:
        client = boto3.client(
            "transcribe",
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        # 仅列出最近 1 个任务，验证权限即可
        client.list_transcription_jobs(MaxResults=1)
        logger.info("[层1] ✅ 凭证有效，Transcribe 权限正常")
        return True
    except Exception as e:
        logger.error(f"[层1] ❌ {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────
# 层2：流式转录测试（amazon-transcribe SDK）
# ─────────────────────────────────────────────

async def test_streaming(
    access_key_id: str,
    secret_access_key: str,
    region: str,
    audio_path: Path,
    language_code: str = LANGUAGE_CODE,
) -> bool:
    """
    流式发送 WAV 文件到 Transcribe，验证：
    - 实时转录（ja-JP）
    - 说话人区分（ShowSpeakerLabel=True）
    """
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.handlers import TranscriptResultStreamHandler
        from amazon_transcribe.model import TranscriptEvent
    except ImportError:
        logger.error("缺少依赖：请先运行 pip3 install amazon-transcribe")
        return False

    logger.info(f"[层2] 流式转录测试，音频：{audio_path.name}，语言：{LANGUAGE_CODE}")

    # 收集所有转录结果
    transcripts: list[str] = []
    speaker_labels: list[str] = []

    class BankConversationHandler(TranscriptResultStreamHandler):
        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            results = transcript_event.transcript.results
            for result in results:
                if result.is_partial:
                    continue  # 只处理最终结果
                for alt in result.alternatives:
                    text = alt.transcript
                    transcripts.append(text)
                    logger.info(f"  [转录] {text}")

                    # 提取说话人标签
                    if alt.items:
                        for item in alt.items:
                            if hasattr(item, "speaker") and item.speaker:
                                speaker_labels.append(item.speaker)

    try:
        # TranscribeStreamingClient 通过环境变量读取凭证
        os.environ["AWS_ACCESS_KEY_ID"] = access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_access_key
        os.environ["AWS_DEFAULT_REGION"] = region

        client = TranscribeStreamingClient(region=region)

        stream = await client.start_stream_transcription(
            language_code=language_code,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding=MEDIA_ENCODING,
            show_speaker_label=True,   # 单声道说话人区分，不传 number_of_channels
        )

        handler = BankConversationHandler(stream.output_stream)

        # 读取 WAV 文件并流式发送（跳过 WAV header，仅发送 PCM 数据）
        async def send_audio():
            with open(audio_path, "rb") as f:
                # 跳过 WAV header（44 字节）
                header = f.read(44)
                if header[:4] == b"RIFF":
                    logger.info("  检测到 WAV 格式，跳过 44 字节 header")
                else:
                    # 非标准 WAV，从头发送
                    f.seek(0)

                chunk_count = 0
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    chunk_count += 1
                await stream.input_stream.end_stream()
                logger.info(f"  音频发送完成，共 {chunk_count} 个 chunk")

        await asyncio.gather(send_audio(), handler.handle_events())

        # 汇总结果
        print("\n" + "=" * 50)
        print("转录结果汇总")
        print("=" * 50)
        if transcripts:
            for t in transcripts:
                print(f"  {t}")
        else:
            print("  （无转录文本）")

        if speaker_labels:
            unique_speakers = set(speaker_labels)
            print(f"\n检测到说话人：{unique_speakers}")
            print(f"说话人区分：{'✅ 正常' if len(unique_speakers) > 1 else '⚠️  仅检测到 1 个说话人（可能音频太短或单人）'}")
        else:
            print("\n说话人标签：未返回（可能模型判断无需区分）")

        logger.info("[层2] ✅ 流式转录测试完成")
        return True

    except Exception as e:
        logger.error(f"[层2] ❌ {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="AWS Transcribe 连通性 + 流式转录测试")
    parser.add_argument("--csv",    default=None, help=f"凭证 CSV（默认：{DEFAULT_CSV}）")
    parser.add_argument("--region", default="ap-northeast-1", help="AWS 区域（默认：ap-northeast-1）")
    parser.add_argument("--audio",  default=None, help="WAV 音频文件路径（16kHz 单声道），不指定则只跑层1")
    parser.add_argument("--lang",   default=LANGUAGE_CODE, help=f"语言代码（默认：{LANGUAGE_CODE}），如 ja-JP / zh-CN / en-US")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    region   = args.region

    if not csv_path.exists():
        logger.error(f"凭证 CSV 不存在：{csv_path}")
        sys.exit(1)

    access_key_id, secret_access_key = load_credentials_from_csv(csv_path)
    logger.info(f"凭证加载成功，Access Key ID: {access_key_id[:8]}...")

    results: dict[str, bool] = {}

    # ── 层1：权限验证 ──
    print("\n" + "=" * 60)
    print("层1：凭证 + Transcribe 权限验证")
    print("=" * 60)
    results["credentials"] = test_credentials(access_key_id, secret_access_key, region)

    # ── 层2：流式转录 ──
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error(f"音频文件不存在：{audio_path}")
            results["streaming"] = False
        else:
            print("\n" + "=" * 60)
            print("层2：流式转录 + 说话人区分")
            print("=" * 60)
            results["streaming"] = await test_streaming(
                access_key_id, secret_access_key, region, audio_path,
                language_code=args.lang,
            )
    else:
        print("\n⚠️  未指定 --audio，跳过流式转录测试")
        print("   如需完整测试，请提供 16kHz 单声道 WAV 文件：")
        print("   python tests/test_transcribe.py --audio /path/to/sample.wav")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        logger.info("全部通过！AWS Transcribe 链路正常。")
    else:
        logger.warning("部分失败，请根据上方 ❌ 行的错误信息排查。")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

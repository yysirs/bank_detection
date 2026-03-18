"""
基准测试脚本

对全部 10 组测试对话数据运行批量检测，与 Ground Truth 对比，输出评估报告。

用法：
    cd /Users/shuliu/Work/Code/bank_detection
    python -m tests.run_benchmark

    # 仅测试单组（调试用）
    python -m tests.run_benchmark --group 1

    # 保存预测结果
    python -m tests.run_benchmark --save-output ./output/

    # 指定 AWS 凭证 CSV（默认使用项目根目录下的 poc_dev_accessKeys20260310.csv）
    python -m tests.run_benchmark --csv /path/to/accessKeys.csv

    # 指定 AWS 区域（默认 ap-northeast-1）
    python -m tests.run_benchmark --region ap-northeast-1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.aws_client import make_bedrock_client_from_csv
from detection.batch_detector import BatchDetector
from detection.evaluator import evaluate_all, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "测试对话数据"
DEFAULT_CSV = PROJECT_ROOT / "poc_dev_accessKeys20260310.csv"
GROUP_TEMPLATE = "第{n}组/PF_NG_FUND_{n:02d}_原文.json"


def load_test_sessions(groups: list[int]) -> list[tuple[dict, str]]:
    """加载指定组的测试数据，返回 (session_dict, 文件stem) 列表"""
    sessions = []
    for n in groups:
        path = DATA_ROOT / GROUP_TEMPLATE.format(n=n)
        if not path.exists():
            logger.warning(f"文件不存在，跳过：{path}")
            continue
        with open(path, encoding="utf-8") as f:
            sessions.append((json.load(f), path.stem))  # stem = PF_NG_FUND_01_原文
        logger.info(f"已加载第 {n} 组：{path.name}")
    return sessions


def run_benchmark(groups, save_dir=None, csv_path=None, region="ap-northeast-1") -> dict:
    """
    运行基准测试。

    Args:
        groups:   要测试的组号列表（1~10）
        save_dir: 若指定，将预测结果 JSON 保存到此目录
        csv_path: AWS 凭证 CSV 文件路径（默认使用项目根目录下的文件）
        region:   AWS 区域（默认 us-east-1）

    Returns:
        汇总评估指标 dict
    """
    csv_file = Path(csv_path) if csv_path else DEFAULT_CSV
    if not csv_file.exists():
        logger.error(f"AWS 凭证 CSV 不存在：{csv_file}")
        sys.exit(1)

    logger.info(f"使用 AWS 凭证：{csv_file}，区域：{region}")
    client = make_bedrock_client_from_csv(csv_file, region=region)
    detector = BatchDetector(client=client)
    session_pairs = load_test_sessions(groups)

    if not session_pairs:
        logger.error("没有找到任何测试数据，退出")
        sys.exit(1)

    gt_sessions = [gt for gt, _ in session_pairs]
    pred_sessions = []
    total_start = time.time()

    for i, (gt, file_stem) in enumerate(session_pairs, 1):
        sid = gt.get("session_id", f"group_{i}")
        logger.info(f"[{i}/{len(session_pairs)}] 检测会话 {sid} ...")
        t0 = time.time()

        result = detector.detect_session(gt)
        pred_dict = detector.to_output_dict(result)
        pred_sessions.append(pred_dict)

        elapsed = time.time() - t0
        logger.info(f"  完成，耗时 {elapsed:.1f}s")

        # 输出文件名与原文件保持一致：PF_NG_FUND_01_原文 → PF_NG_FUND_01_pred
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            base = file_stem.replace("_原文", "")   # PF_NG_FUND_01
            out_path = save_dir / f"{base}_pred.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(pred_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"  已保存预测结果 → {out_path}")

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部 {len(gt_sessions)} 个会话检测完成，总耗时 {total_elapsed:.1f}s")

    # 评估
    metrics = evaluate_all(gt_sessions, pred_sessions)

    # 保存汇总报告
    if save_dir:
        report_path = save_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存 → {report_path}")

    print_report(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="合规检测基准测试")
    parser.add_argument(
        "--group",
        type=int,
        default=None,
        help="只测试指定组（1~10），不指定则测试全部",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="预测结果保存目录（可选）",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=f"AWS 凭证 CSV 文件路径（默认：{DEFAULT_CSV}）",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="ap-northeast-1",
        help="AWS 区域（默认：ap-northeast-1）",
    )
    args = parser.parse_args()

    groups = [args.group] if args.group else list(range(1, 11))
    save_dir = Path(args.save_output) if args.save_output else None

    run_benchmark(groups, save_dir, csv_path=args.csv, region=args.region)


if __name__ == "__main__":
    main()

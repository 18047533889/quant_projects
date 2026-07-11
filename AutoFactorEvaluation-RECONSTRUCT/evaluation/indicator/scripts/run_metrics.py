"""从 repo 根目录运行阶段 4-2 指标计算。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    读取默认 mock 输入目录，调用 `main_code.run_stage4_2`，并把输出写到
    `database/mock/evaluation/indicator/stage4_2_output/`，日志写到
    `database/log/evaluation/indicator/`。
"""

from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from main_code import run_stage4_2
from main_code.calculator import default_run_paths


def main() -> None:
    """按默认根目录路径运行阶段 4-2 并打印摘要。"""

    paths = default_run_paths(ROOT)
    result = run_stage4_2(paths.input_dir, paths.output_dir, paths.log_dir)
    print(f"Generated Stage 4-2 outputs in {result['output_dir']}")
    print(f"Metrics rows: {len(result['metrics_matrix'])}")
    print(f"Validation events: {len(result['validation_events'])}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "AutoFactorEvaluation-RECONSTRUCT" / "pipeline.py"
text = PATH.read_text(encoding="utf-8")

start = text.find("\ndef run_gtja185_batch_pipeline(")
end = text.find("\ndef _validate_all_paths", start)
if start < 0 or end < 0:
    raise RuntimeError(
        f"GTJA batch function anchors not found: start={start} end={end}"
    )
text = text[:start] + "\n" + text[end:]

text = "\n".join(
    line for line in text.splitlines() if "--gtja" not in line
) + "\n"

start = text.find("    if args.gtja185:")
end = text.find("    # 全流程启动前验证所有路径已就绪", start)
if start < 0 or end < 0:
    raise RuntimeError(f"GTJA CLI block anchors not found: start={start} end={end}")
text = text[:start] + text[end:]

remaining = [
    line for line in text.splitlines()
    if "gtja185" in line.lower() or "evaluation.gtja" in line.lower()
]
if remaining:
    raise RuntimeError(f"pipeline.py still contains GTJA coupling: {remaining[:10]}")

PATH.write_text(text, encoding="utf-8")
print("Removed GTJA-specific compatibility path from AutoFactorEvaluation pipeline")

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "AutoFactorEvaluation-RECONSTRUCT" / "pipeline.py"

text = PATH.read_text(encoding="utf-8")

start = text.find("\ndef run_gtja185_batch_pipeline(")
end = text.find("\ndef _validate_all_paths", start)
if start < 0 or end < 0:
    raise RuntimeError("GTJA batch function anchors not found")
text = text[:start] + "\n" + text[end:]

lines = []
for line in text.splitlines():
    if "--gtja" in line:
        continue
    lines.append(line)
text = "\n".join(lines) + "\n"

start = text.find("    if args.gtja185:")
end = text.find("    # 全流程启动前验证所有路径已就绪", start)
if start < 0 or end < 0:
    raise RuntimeError("GTJA CLI block anchors not found")
text = text[:start] + text[end:]

if "gtja185" in text.lower():
    raise RuntimeError("pipeline.py still contains GTJA-specific coupling")
PATH.write_text(text, encoding="utf-8")
print("Removed GTJA-specific compatibility path from AutoFactorEvaluation pipeline")

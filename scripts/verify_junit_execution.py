"""Fail CI when a supposedly executed gate collected no successful test bodies."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET


def verify(path: str) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    skipped = sum(c.find("skipped") is not None for c in cases)
    passed = len(cases) - failed - skipped
    result = {"collected": len(cases), "passed": passed, "failed": failed,
              "skipped": skipped, "status": "PASS" if passed > 0 and failed == 0 else "FAIL"}
    if result["status"] != "PASS":
        raise ValueError(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit")
    print(json.dumps(verify(parser.parse_args().junit)))

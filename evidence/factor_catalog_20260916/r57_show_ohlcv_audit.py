import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ohlcv_audit.json"
with open(path, encoding="utf-8") as fh:
    d = json.load(fh)

hdr = "{:<20s} {:>5s} {:<11s} {:<14s} {}".format(
    "canonical", "uses", "registered", "status", "param_names"
)
print(hdr)
print("-" * 118)
ok, bad = [], []
for r in d["rows"]:
    pn = r["param_names"]
    print("{:<20s} {:>5d} {:<11s} {:<14s} {}".format(
        r["canonical"], r["uses"], str(r["registered"]), str(r["status"]), pn
    ))
    if r["signature"]:
        print("{:<20s}   sig: {}".format("", r["signature"][:150]))
    # directory convention needs the operator to accept O,H,L,C,V as leading params
    if not r["registered"]:
        bad.append((r["canonical"], "not-registered"))
    elif not pn:
        bad.append((r["canonical"], "empty param_names"))
    else:
        head = [str(x).lower() for x in pn[:5]]
        want = ["open", "high", "low", "close", "volume"]
        if head[:5] == want:
            ok.append(r["canonical"])
        else:
            bad.append((r["canonical"], "head=%s" % head[:5]))

print()
print("=== conforms to the unified OHLCV signature (%d) ===" % len(ok))
print("   ", ok)
print()
print("=== does NOT conform (%d) ===" % len(bad))
for c, why in bad:
    print("   %-20s %s" % (c, why))

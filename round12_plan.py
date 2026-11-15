"""Round 12: FLOPs/token (model.flops_per_token, the value train.py logs) and the stable-token counts that put each run's
total training FLOPs (stable + 10% decay) within 0.5% of a target. H128k before its class files exist: the same formula
with the capacity-bound cluster size L = ceil(122,880 / 4,096) = 30 (asserted equal once the files exist).

python round12_plan.py  -> runs/round12/plan.json
"""
import json
import math
import os

from arms import ARMS
from model import flops_per_token

TPS = 64 * 1024
T575, T300 = 6.716844637929216e16, 3.504924836453376e16


def fpt(arm):
    cfg = ARMS[arm]
    try:
        return flops_per_token(cfg)["total"], flops_per_token(cfg)["head"], "logged formula"
    except FileNotFoundError:
        f = flops_per_token(ARMS["B"])
        body = f["total"] - f["head"]
        L = math.ceil((cfg.vocab - cfg.head_adapt) / cfg.head_C)
        head = 3 * (2 * (cfg.head_adapt + cfg.head_C + L) * cfg.d + cfg.vocab * cfg.d / 16384)
        return body + head, head, f"formula, L = {L} (class files not built yet)"


def match(f, target):
    S = round(target / (f * 1.1 * TPS))
    best = min(range(S - 50, S + 50), key=lambda s: abs(f * (s + round(0.1 * s)) * TPS - target))
    tot = f * (best + round(0.1 * best)) * TPS
    return {"stable_steps": best, "stable_tokens": best * TPS, "total_tokens": (best + round(0.1 * best)) * TPS,
            "flops": tot, "rel_err": tot / target - 1}


if __name__ == "__main__":
    out = {}
    for arm in ("B", "H3s384", "F8k384", "F128k384", "H128k384", "tF128k"):
        f, h, how = fpt(arm)
        out[arm] = {"fpt": f, "head": h, "head_share": h / f, "how": how,
                    "match_575": match(f, T575), "match_300": match(f, T300)}
    f = out["tF128k"]["fpt"]
    S = round(50e6 / TPS)
    out["tF128k"]["proxy"] = {"stable_steps": S, "stable_tokens": S * TPS, "total_tokens": (S + round(0.1 * S)) * TPS,
                              "flops": f * (S + round(0.1 * S)) * TPS}
    out["H128k384"]["match_575_charged"] = match(out["H128k384"]["fpt"], T575 - out["tF128k"]["proxy"]["flops"])
    json.dump(out, open("runs/round12/plan.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:9s} fpt {v['fpt'] / 1e6:7.2f}M head share {100 * v['head_share']:5.1f}%  575: S={v['match_575']['stable_tokens']} "
              f"({v['match_575']['rel_err']:+.2e})  300: S={v['match_300']['stable_tokens']} ({v['match_300']['rel_err']:+.2e})  [{v['how']}]")
    print("proxy", out["tF128k"]["proxy"], "charged", out["H128k384"]["match_575_charged"])
